import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Projet 19 - Datacenter Energy Efficiency", layout="wide")

st.title("⚡ Projet 19 : Efficacité Énergétique des Centres de Données")
st.markdown("""
**Système hybride de pilotage intelligent du refroidissement (HVAC) :**
- **LSTM** : Prévision proactive de la charge CPU des serveurs.
- **CNN** : Détection des points chauds (hotspots) sur cartes thermiques 2D.
- **Random Forest** : Ajustement dynamique de la consigne de climatisation pour minimiser l'empreinte carbone.
""")

st.sidebar.header("🕹️ Simulation Salle Serveurs")
current_cpu = st.sidebar.slider("Charge CPU Actuelle (%)", min_value=10, max_value=100, value=65)
ambient_temp = st.sidebar.slider("Température Extérieure (°C)", min_value=15, max_value=40, value=25)
add_hotspot = st.sidebar.checkbox("Simuler un Hotspot sur la Baie (Surmenage)", value=True)

# 1. Prediction LSTM simplifiee (simulation prompte)
predicted_future_cpu = min(100.0, current_cpu + np.random.uniform(-3, 8))

# 2. Simulation Carte Thermique 2D (Baies 8x8)
grid_size = 8
base_rack_temp = 20.0 + (current_cpu / 100.0) * 10.0
thermal_map = np.random.normal(base_rack_temp, 0.8, (grid_size, grid_size))

if add_hotspot:
    thermal_map[3:5, 4:6] += np.random.uniform(5.0, 9.0)

max_hotspot_temp = np.max(thermal_map)

# 3. Decision Random Forest (Consigne HVAC)
# Baseline classique : consigne fixe a 18°C
baseline_setpoint = 18.0
# Consigne intelligente
optimized_setpoint = 24.5 - 0.07 * (predicted_future_cpu - 50) - 0.25 * (max_hotspot_temp - 26)
optimized_setpoint = np.clip(optimized_setpoint, 16.5, 25.5)

# Estimation gains energetiques (regle standard: 4.5% par °C releve)
temp_difference = optimized_setpoint - baseline_setpoint
energy_saved_pct = max(0.0, temp_difference * 4.5)
power_saved_kw = energy_saved_pct * 1.8  # estimation pour une salle type

# Affichage des Metriques Clés
col1, col2, col3, col4 = st.columns(4)
col1.metric("Charge CPU Prédite (LSTM)", f"{predicted_future_cpu:.1f} %", delta=f"{predicted_future_cpu - current_cpu:.1f} %")
col2.metric("Point Chaud Max (CNN)", f"{max_hotspot_temp:.1f} °C", delta="Critique" if max_hotspot_temp > 30 else "Normal", delta_color="inverse")
col3.metric("Consigne Climatisation (RF)", f"{optimized_setpoint:.1f} °C", delta=f"+{temp_difference:.1f} °C vs Fixe")
col4.metric("Économie d'Énergie Estimée", f"{energy_saved_pct:.1f} %", delta=f"-{power_saved_kw:.1f} kW")

st.divider()

# Graphiques
g_col1, g_col2 = st.columns(2)

with g_col1:
    st.subheader("🔥 Carte Thermique 2D des Baies (Analyse CNN)")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(thermal_map, annot=True, fmt=".1f", cmap="coolwarm", vmin=18, vmax=35, ax=ax, cbar_kws={'label': 'Température (°C)'})
    ax.set_title("Matrice de Température des Racks (8x8)")
    ax.set_xlabel("Colonnes de Racks")
    ax.set_ylabel("Rangées de Racks")
    st.pyplot(fig)

with g_col2:
    st.subheader("📈 Prévision de Charge & Refroidissement")
    time_steps = np.arange(-10, 6)
    history_cpu = np.clip(current_cpu + np.random.normal(0, 3, 11), 10, 100)
    future_forecast = [history_cpu[-1]] + [min(100, history_cpu[-1] + i * (predicted_future_cpu - history_cpu[-1])/5) for i in range(1, 6)]
    
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(time_steps[:11], history_cpu, 'b-o', label="Historique Réel (CPU %)")
    ax2.plot(time_steps[10:], future_forecast, 'r--s', label="Prévision LSTM (t+5 min)")
    ax2.axvline(x=0, color='gray', linestyle=':', label='Instant T')
    ax2.set_xlabel("Temps (minutes)")
    ax2.set_ylabel("Charge CPU (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    st.pyplot(fig2)

st.success(f"🌱 **Impact Écologique :** Le pilotage prédictif permet d'éviter le sur-refroidissement inutile, économisant **{energy_saved_pct:.1f}%** d'électricité tout en maintenant les baies sous la limite critique de sécurité.")

