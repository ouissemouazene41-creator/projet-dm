"""
Projet 19 : Efficacite Energetique des Centres de Donnees (Datacenters)
Architecture ML Hybride:
 1. LSTM : Prevision temporelle de la charge CPU des serveurs.
 2. CNN : Analyse spatiale des cartes thermiques 2D des baies (detection de hotspots).
 3. Random Forest : Pilotage dynamique et proactif des climatiseurs (HVAC).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
import os

# ==========================================
# 1. GENERATION / CHARGEMENT DES DONNEES
# ==========================================
def generate_synthetic_datacenter_data(n_samples=2000, grid_size=8, seed=42):
    """
    Genere un jeu de donnees realiste si les fichiers Kaggle ne sont pas encore presents :
    - Charge CPU (%) avec cycle journalier et bruit
    - Cartes thermiques 2D (8x8) des baies de serveurs
    - Parametres environnementaux (temperature ambiante, humidite)
    """
    np.random.seed(seed)
    time = np.linspace(0, 48 * np.pi, n_samples)
    
    # 1. Charge CPU (base cyclique + pics aleatoires de batch jobs)
    base_cpu = 45 + 25 * np.sin(time / 6) + 10 * np.cos(time / 2)
    noise = np.random.normal(0, 5, n_samples)
    spikes = (np.random.rand(n_samples) > 0.95) * np.random.uniform(10, 25, n_samples)
    cpu_load = np.clip(base_cpu + noise + spikes, 5, 100)
    
    # 2. Cartes thermiques 2D (ex: 8x8 baies dans la salle des serveurs)
    heatmaps = []
    hotspot_scores = []
    for cpu in cpu_load:
        # Temperature de base liee a la charge CPU globale (20 a 35 °C)
        base_temp = 20.0 + (cpu / 100.0) * 12.0
        rack_map = np.random.normal(base_temp, 1.5, (grid_size, grid_size))
        
        # Simuler un hotspot aleatoire lie aux racks sous forte charge
        if cpu > 60:
            rx, ry = np.random.randint(1, grid_size-1, 2)
            rack_map[rx-1:rx+2, ry-1:ry+2] += np.random.uniform(4.0, 8.0)
            
        heatmaps.append(rack_map)
        hotspot_scores.append(np.max(rack_map)) # Temperature maximale du point chaud

    heatmaps = np.array(heatmaps)
    hotspot_scores = np.array(hotspot_scores)
    
    # 3. Consigne HVAC optimale cible (label d'entrainement pour Random Forest)
    # Plus la charge predite et le hotspot sont hauts, plus la clim doit baisser la consigne et ventiler
    hvac_setpoint_temp = 24.0 - 0.08 * (cpu_load - 50) - 0.3 * (hotspot_scores - 28)
    hvac_setpoint_temp = np.clip(hvac_setpoint_temp, 16.0, 26.0)
    
    return cpu_load, heatmaps, hotspot_scores, hvac_setpoint_temp


# ==========================================
# 2. MODELE LSTM : PREVISION DE CHARGE CPU
# ==========================================
def create_sliding_windows(data, window_size=30):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def build_and_train_lstm(cpu_series, window_size=30, epochs=15):
    print("\n--- [1/3] Entrainement du modele LSTM (Charge CPU) ---")
    scaler = MinMaxScaler()
    scaled_cpu = scaler.fit_transform(cpu_series.reshape(-1, 1))
    
    X, y = create_sliding_windows(scaled_cpu, window_size)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"-> LSTM Test RMSE: {rmse:.2f}% CPU")
    return model, scaler, y_pred, y_true


# ==========================================
# 3. MODELE CNN : ANALYSE CARTES THERMIQUES
# ==========================================
def build_and_train_cnn(heatmaps, hotspot_targets, epochs=15):
    print("\n--- [2/3] Entrainement du modele CNN (Cartes 2D Baies) ---")
    # Reshape pour format image (Samples, H, W, Channels)
    X = heatmaps[..., np.newaxis]
    y = hotspot_targets
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(X.shape[1], X.shape[2], 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1) # Predire la severite / temperature max du hotspot
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"-> CNN Test MAE (Hotspot): {mae:.2f} °C")
    return model


# ==========================================
# 4. RANDOM FOREST : PILOTAGE DYNAMIQUE HVAC
# ==========================================
def build_and_train_rf(cpu_pred, hotspot_pred, hvac_targets):
    print("\n--- [3/3] Entrainement du Random Forest (Pilotage Climatiseurs) ---")
    # Fusion des signaux : Charge CPU anticipee (LSTM) + Hotspot spatial (CNN)
    X_features = np.column_stack([cpu_pred.flatten(), hotspot_pred.flatten()])
    y = hvac_targets[:len(X_features)]
    
    split = int(0.8 * len(X_features))
    X_train, X_test = X_features[:split], X_features[split:]
    y_train, y_test = y[:split], y[split:]
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    
    predictions = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"-> Random Forest HVAC Test RMSE: {rmse:.2f} °C de consigne")
    
    # 5. Calcul des economies d'energie vs consigne statique traditionnelle (ex: fixe a 18°C)
    # Regle empirique datacenter : Chaque +1°C de consigne economise ~4% a 5% de consommation HVAC
    static_setpoint = 18.0
    avg_dynamic_setpoint = np.mean(predictions)
    temp_delta = avg_dynamic_setpoint - static_setpoint
    estimated_energy_savings_pct = max(0, temp_delta * 4.5)
    
    print("\n" + "="*50)
    print(f"Consigne Statique Classique : {static_setpoint:.1f} °C")
    print(f"Consigne Dynamique Optimisee : {avg_dynamic_setpoint:.1f} °C")
    print(f"Gain energetique estime : ~{estimated_energy_savings_pct:.1f}% d'electricite economisee !")
    print("="*50)
    
    return rf


# ==========================================
# EXECUTION PRINCIPALE
# ==========================================
if __name__ == "__main__":
    print("Demarrage du pipeline Projet 19...")
    cpu_load, heatmaps, hotspot_scores, hvac_targets = generate_synthetic_datacenter_data()
    
    # 1. LSTM
    lstm_model, scaler, cpu_pred, cpu_true = build_and_train_lstm(cpu_load, window_size=30, epochs=10)
    
    # 2. CNN
    cnn_model = build_and_train_cnn(heatmaps, hotspot_scores, epochs=10)
    hotspot_features = cnn_model.predict(heatmaps[30:][..., np.newaxis])
    
    # Aligner dimensions pour Random Forest
    min_len = min(len(cpu_pred), len(hotspot_features))
    
    # 3. Random Forest
    rf_model = build_and_train_rf(cpu_pred[:min_len], hotspot_features[:min_len], hvac_targets[30:])
    print("\nPipeline termine avec succes !")

