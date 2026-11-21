"""
LSTM Model: Datacenter Server Workload (CPU) Forecasting
---------------------------------------------------------
Dataset: Cloud Workload Job Traces for Resource Forecasting (zoya77)
Features: ['Requested_CPUs', 'Used_CPUs', 'Requested_Memory(MB)', 'Used_Memory(MB)']
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# ── 1. LOAD DATA ──────────────────────────────────────────────
def load_dataset():
    # Exact Kaggle path from your session
    kaggle_path = '/kaggle/input/datasets/zoya77/cloud-workload-job-traces-for-resource-forecasting/cloud_workload_dataset.csv'
    
    if os.path.exists(kaggle_path):
        csv_file = kaggle_path
    else:
        # Fallback search
        found = glob.glob('/kaggle/input/**/cloud_workload_dataset.csv', recursive=True) or glob.glob('*.csv')
        csv_file = found[0] if found else None

    if csv_file:
        print(f"📂 Loading: {csv_file}")
        df = pd.read_csv(csv_file)
        # Select key workload features
        features = ['Requested_CPUs', 'Used_CPUs', 'Requested_Memory(MB)', 'Used_Memory(MB)']
        available = [c for c in features if c in df.columns]
        data = df[available].dropna().values
    else:
        print("⚙️ Fallback: Generating benchmark workload traces...")
        np.random.seed(42)
        t = np.linspace(0, 48 * np.pi, 3000)
        base = 45 + 25 * np.sin(t / 6)
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
        y.append(scaled_data[i + window_size, 1])  # Target: next Used_CPUs
        
    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    return X[:split], X[split:], y[:split], y[split:], scaler

# ── 3. BUILD & TRAIN LSTM ─────────────────────────────────────
def train_model(X_train, y_train, X_test, y_test, window_size=30, n_features=2):
    model = Sequential([
        Input(shape=(window_size, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    print(f"🚀 Training LSTM on {len(X_train)} samples...")
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), verbose=1)
    return model

# ── 4. EVALUATION & VISUALIZATION ─────────────────────────────
def evaluate_and_plot(model, X_test, y_test):
    pred = model.predict(X_test).flatten()
    rmse = np.sqrt(np.mean((pred - y_test) ** 2))
    print(f"\n🎉 SUCCESS! LSTM Trained. Test RMSE: {rmse:.4f}")
    
    # Plot first 100 timesteps
    plt.figure(figsize=(10, 4))
    plt.plot(y_test[:100], label='Actual Real CPU', color='blue', alpha=0.7)
    plt.plot(pred[:100], label='LSTM Predicted CPU', color='red', linestyle='--')
    plt.title('LSTM CPU Workload Prediction (First 100 timesteps)')
    plt.xlabel('Time (steps)')
    plt.ylabel('Normalized CPU Usage')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_curve.png')
    plt.show()
    return pred

if __name__ == "__main__":
    data = load_dataset()
    X_train, X_test, y_train, y_test, scaler = prepare_windows(data, window_size=30)
    model = train_model(X_train, y_train, X_test, y_test, window_size=30, n_features=X_train.shape[2])
    evaluate_and_plot(model, X_test, y_test)
