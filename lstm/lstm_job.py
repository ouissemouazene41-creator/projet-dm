"""
LSTM 1: Predict CPU Usage from Jobs Dataset (cloud_workload_dataset.csv)
"""

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# ── 1. LOAD & SORT DATA BY TIME ───────────────────────────────
files = glob.glob('/kaggle/input/**/cloud_workload_dataset.csv', recursive=True) or glob.glob('cloud_workload_dataset.csv')
df = pd.read_csv(files[0])

# Keep CPU and Memory columns, sorted by time so it forms a real timeline
features = [c for c in ['Requested_CPUs', 'Used_CPUs', 'Requested_Memory(MB)', 'Used_Memory(MB)'] if c in df.columns]
time_col = next((c for c in ['Submit_Time', 'Submit Time', 'Start_Time', 'Start Time'] if c in df.columns), None)
if time_col:
    df = df.sort_values(time_col)

# Smooth noisy per-job jumps into a clean workload trend
data = df[features].dropna().rolling(window=5, min_periods=1).mean().values
target_col = features.index('Used_CPUs') if 'Used_CPUs' in features else 0

# ── 2. SCALE & MAKE 30-STEP WINDOWS ───────────────────────────
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

WINDOW = 30
X, y = [], []
for i in range(len(scaled) - WINDOW):
    X.append(scaled[i:i + WINDOW])
    y.append(scaled[i + WINDOW, target_col])

X, y = np.array(X), np.array(y)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ── 3. BUILD & TRAIN LSTM 1 ───────────────────────────────────
model = Sequential([
    Input(shape=(WINDOW, X.shape[2])),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1),
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)

# ── 4. TEST ACCURACY & PLOT ───────────────────────────────────
pred = model.predict(X_test).flatten()
rmse = np.sqrt(np.mean((pred - y_test) ** 2))
mae = np.mean(np.abs(pred - y_test))
print(f"\nLSTM 1 (Jobs) -> Test RMSE: {rmse:.4f} | Test MAE: {mae:.4f}")

plt.figure(figsize=(10, 4))
plt.plot(y_test[:100], label='Actual CPU', color='blue')
plt.plot(pred[:100], label='LSTM 1 Predicted CPU', color='red', linestyle='--')
plt.title('LSTM 1 — Jobs Dataset Prediction')
plt.legend()
plt.grid(True)
plt.savefig('lstm1_jobs.png')
plt.show()

