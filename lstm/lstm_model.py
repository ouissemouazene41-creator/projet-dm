"""
LSTM Model: Datacenter Server Workload (CPU) Forecasting
---------------------------------------------------------
This model forecasts future CPU utilization from historical workload traces
to enable proactive cooling in datacenter server rooms.
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ── 1. LOAD DATA ──────────────────────────────────────────────
def load_data():
    """Searches for Kaggle dataset CSV or generates synthetic traces."""
    csv_files = glob.glob('/kaggle/input/**/*.csv', recursive=True) or glob.glob('*.csv')
    
    if csv_files:
        print(f"📂 Loading dataset from: {csv_files[0]}")
        df = pd.read_csv(csv_files[0], nrows=50000)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        data = df[numeric_cols[:2]].dropna().values
    else:
        print("⚙️ Generating synthetic CPU workload trace...")
        np.random.seed(42)
        t = np.linspace(0, 48 * np.pi, 3000)
        base = 45 + 25 * np.sin(t / 6) + 10 * np.cos(t / 2)
        noise = np.random.normal(0, 4, 3000)
        spikes = (np.random.rand(3000) > 0.94) * np.random.uniform(10, 25, 3000)
        cpu = np.clip(base + noise + spikes, 10, 100)
        mem = np.clip(cpu * 0.8 + np.random.normal(0, 5, 3000), 10, 95)
        data = np.column_stack([cpu, mem])
        
    return data

# ── 2. PREPROCESSING & SLIDING WINDOWS ────────────────────────
def prepare_windows(data, window_size=30):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i + window_size])
        y.append(scaled_data[i + window_size, 0])  # Target: next CPU %
        
    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    return X[:split], X[split:], y[:split], y[split:], scaler

# ── 3. BUILD & TRAIN LSTM ─────────────────────────────────────
def train_lstm(X_train, y_train, X_test, y_test, window_size=30, n_features=2):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window_size, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    print("🚀 Training LSTM model...")
    model.fit(X_train, y_train, epochs=8, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    return model

# ── 4. EVALUATION ─────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test).flatten()
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    print(f"\n🎉 Test RMSE: {rmse:.4f} (Normalized Scale)")
    return predictions

if __name__ == "__main__":
    data = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_windows(data)
    model = train_lstm(X_train, y_train, X_test, y_test, window_size=30, n_features=X_train.shape[2])
    evaluate(model, X_test, y_test)
