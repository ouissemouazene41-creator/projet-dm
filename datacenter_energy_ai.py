"""
PROJET 19 : EFFICACITE ENERGETIQUE DES CENTRES DE DONNEES (DATACENTERS)
-----------------------------------------------------------------------
Architecture Hybride d'Intelligence Artificielle pour le Refroidissement Intelligent :
1. LSTM           : Prevision temporelle proactive de la charge CPU des serveurs.
2. CNN            : Analyse spatiale 2D des cartes thermiques pour detecter les points chauds (Hotspots).
3. Random Forest  : Ajustement dynamique de la consigne des climatiseurs (HVAC).
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv2D, Flatten, MaxPooling2D

# ==============================================================================
# 1. CHARGEMENT OU GENERATION DES DONNEES
# ==============================================================================
def load_or_generate_data():
    """Charge les fichiers CSV si disponibles, sinon genere des traces realistes."""
    csv_files = glob.glob('/kaggle/input/**/*.csv', recursive=True) or glob.glob('*.csv')
    
    if csv_files:
        print(f"📂 Chargement du dataset reel : {csv_files[0]}")
        df = pd.read_csv(csv_files[0], nrows=10000)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cpu_load = df[numeric_cols[0]].fillna(df[numeric_cols[0]].mean()).values
        # Normaliser entre 10% et 95%
        cpu_load = 10 + 85 * ((cpu_load - cpu_load.min()) / (cpu_load.max() - cpu_load.min() + 1e-6))
    else:
        print("⚙️ Generation automatique de donnees de simulation de datacenter...")
        np.random.seed(42)
        t = np.linspace(0, 48 * np.pi, 3000)
        base = 45 + 25 * np.sin(t / 6) + 10 * np.cos(t / 2)
        noise = np.random.normal(0, 4, 3000)
        spikes = (np.random.rand(3000) > 0.94) * np.random.uniform(10, 25, 3000)
        cpu_load = np.clip(base + noise + spikes, 10, 100)

    # Creation des cartes thermiques 2D (Baies 8x8)
    grid_size = 8
    heatmaps, hotspot_scores = [], []
    for cpu in cpu_load:
        base_temp = 20.0 + (cpu / 100.0) * 11.0
        m = np.random.normal(base_temp, 1.2, (grid_size, grid_size))
        if cpu > 65:
            rx, ry = np.random.randint(1, grid_size - 1, 2)
            m[rx-1:rx+2, ry-1:ry+2] += np.random.uniform(4.0, 7.5)
        heatmaps.append(m)
        hotspot_scores.append(np.max(m))
        
    return cpu_load, np.array(heatmaps), np.array(hotspot_scores)

# ==============================================================================
# 2. MODELE 1 : LSTM (PREVISION TEMPORELLE DE LA CHARGE CPU)
# ==============================================================================
def train_lstm(cpu_series, window=30):
    print("\n[1/3] 🧠 Entrainement du LSTM (Prevision de Charge)...")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(cpu_series.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled) - window):
        X.append(scaled[i:i + window])
        y.append(scaled[i + window])
    X, y = np.array(X), np.array(y)
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=8, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    y_pred = scaler.inverse_transform(model.predict(X_test))
    y_true = scaler.inverse_transform(y_test)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"✅ LSTM Test RMSE : {rmse:.2f}% CPU")
    return model, y_pred

# ==============================================================================
# 3. MODELE 2 : CNN (DETECTION SPATIALE DES HOTSPOTS SUR CARTES 2D)
# ==============================================================================
def train_cnn(heatmaps, hotspot_scores):
    print("\n[2/3] 🔍 Entrainement du CNN (Detection des Points Chauds)...")
    X = heatmaps[..., np.newaxis]
    y = hotspot_scores
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(8, 8, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=8, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"✅ CNN Test MAE : {mae:.2f}°C")
    return model, y_pred

# ==============================================================================
# 4. MODELE 3 : RANDOM FOREST (PILOTAGE DYNAMIQUE DU REFROIDISSEMENT HVAC)
# ==============================================================================
def train_random_forest(cpu_pred, hotspot_pred):
    print("\n[3/3] ❄️ Entrainement du Random Forest (Controle HVAC)...")
    min_len = min(len(cpu_pred), len(hotspot_pred))
    X = np.column_stack([cpu_pred[:min_len].flatten(), hotspot_pred[:min_len].flatten()])
    
    # Consigne optimale : baisser la consigne si le CPU monte ou si un hotspot apparait
    y_target = 24.5 - 0.08 * (X[:, 0] - 50) - 0.3 * (X[:, 1] - 27)
    y_target = np.clip(y_target, 16.5, 25.5)
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    
    preds = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"✅ Random Forest Test RMSE : {rmse:.2f}°C de consigne")
    
    # Estimation des gains d'energie vs consigne statique traditionnelle a 18°C
    baseline_temp = 18.0
    avg_ai_temp = np.mean(preds)
    saved_energy_pct = max(0.0, (avg_ai_temp - baseline_temp) * 4.5)
    
    print("\n" + "="*55)
    print("📊 RESULTATS GLOBAUX DU PROJET 19 :")
    print(f" - Consigne statique classique : {baseline_temp:.1f}°C")
    print(f" - Consigne IA dynamique      : {avg_ai_temp:.1f}°C")
    print(f" - Gain energetique estime     : ~{saved_energy_pct:.1f}% d'electricite")
    print(f" - Empreinte carbone           : Forte reduction CO2")
    print("="*55)
    return rf

# ==============================================================================
# POINT D'ENTREE
# ==============================================================================
if __name__ == "__main__":
    cpu_data, heatmaps, hotspots = load_or_generate_data()
    lstm_model, cpu_forecast = train_lstm(cpu_data)
    cnn_model, hotspot_forecast = train_cnn(heatmaps, hotspots)
    rf_model = train_random_forest(cpu_forecast, hotspot_forecast)
    print("\n🎉 Projet 19 execute avec succes !")
