"""
LSTM 2: Predict CPU Usage from Alibaba Cluster Trace (Past 30 Steps)
"""

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── 1. LOAD & SORT ALIBABA DATA ───────────────────────────────
all_csvs = sorted(glob.glob('/kaggle/input/**/*.csv', recursive=True) or glob.glob('*.csv'))
print("Found CSVs:", all_csvs)

candidates = (
    [f for f in all_csvs if 'machine_metric' in f.lower() or 'machine_usage' in f.lower()]
    + [f for f in all_csvs if 'sensor_table' in f.lower()]
    + [f for f in all_csvs if ('alibaba' in f.lower() or 'cluster' in f.lower()) and 'cloud_workload' not in f.lower()]
    + [f for f in all_csvs if 'cloud_workload' not in f.lower()]
)
# Deduplicate while keeping order
files = list(dict.fromkeys(candidates))
if not files:
    raise FileNotFoundError("No Alibaba CSV found in /kaggle/input! Attach the Alibaba dataset first.")

raw_cols = ['machine_id', 'time_stamp', 'cpu_util_percent', 'mem_util_percent', 'mem_gps', 'mkpi', 'net_in', 'net_out', 'disk_io_percent']

df = None
features = []
for path in files:
    temp = pd.read_csv(path, nrows=50000)
    if temp.shape[1] == 9 and 'cpu_util_percent' not in temp.columns:
        temp = pd.read_csv(path, header=None, names=raw_cols, nrows=50000)

    # Convert numeric-looking string columns to numbers
    for col in temp.columns:
        if temp[col].dtype == object:
            converted = pd.to_numeric(temp[col], errors='coerce')
            if converted.notna().sum() > len(temp) * 0.5:
                temp[col] = converted

    # Pick CPU first, then memory/network/disk columns (case-insensitive)
    cpu_cols = [c for c in temp.columns if 'cpu' in str(c).lower() and pd.api.types.is_numeric_dtype(temp[c])]
    other_cols = [c for c in temp.columns if any(k in str(c).lower() for k in ('mem', 'net', 'disk', 'util')) and c not in cpu_cols and pd.api.types.is_numeric_dtype(temp[c])]
    picked = cpu_cols[:1] + other_cols[:4]

    if not picked:
        picked = [c for c in temp.select_dtypes(include=[np.number]).columns if 'time' not in str(c).lower() and 'id' not in str(c).lower()][:5]

    if picked:
        df, features = temp, picked
        print(f"Loaded: {path}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Using features: {features}")
        break

if df is None or not features:
    raise ValueError(f"Could not find numeric CPU/telemetry columns in {files}. Check your dataset CSVs.")

ts_col = next(
    (c for c in df.columns if any(k in str(c).lower() for k in ('time_stamp', 'timestamp', 'start_time', 'time'))),
    None,
)
if ts_col:
    df = df.sort_values(ts_col)
    if df[ts_col].nunique() >= 500:
        df = df.groupby(ts_col, as_index=False)[features].mean()
    else:
        id_col = next((c for c in df.columns if any(k in str(c).lower() for k in ('machine', 'node'))), None)
        if id_col:
            top_machine = df[id_col].value_counts().idxmax()
            df = df[df[id_col] == top_machine].sort_values(ts_col)

data = df[features].dropna().values.astype(np.float32)
del df

# ── 2. SCALE & MAKE 10-STEP WINDOWS ───────────────────────────
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data).astype(np.float32)

WINDOW = 10
n_samples = len(scaled) - WINDOW
X = np.empty((n_samples, WINDOW, scaled.shape[1]), dtype=np.float32)
y = np.empty(n_samples, dtype=np.float32)
for i in range(n_samples):
    X[i] = scaled[i:i + WINDOW]
    y[i] = scaled[i + WINDOW, 0]  # Target: next CPU %

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ── 3. BUILD & TRAIN LSTM 2 (Conv1D + LSTM) ──────────────────
model = Sequential([
    Input(shape=(WINDOW, X.shape[2])),
    Conv1D(filters=32, kernel_size=3, padding='causal', activation='relu'),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1),
])

model.compile(optimizer='adam', loss='huber', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

# ── 4. TEST ACCURACY & PLOT ───────────────────────────────────
pred = model.predict(X_test).flatten()
rmse = np.sqrt(np.mean((pred - y_test) ** 2))
mae = np.mean(np.abs(pred - y_test))
print(f"\nLSTM 2 (Alibaba) -> Test RMSE: {rmse:.4f} | Test MAE: {mae:.4f}")

plt.figure(figsize=(10, 4))
plt.plot(y_test[:100], label='Actual CPU', color='blue')
plt.plot(pred[:100], label='LSTM 2 Predicted CPU', color='green', linestyle='--')
plt.title('LSTM 2 — Alibaba Cluster Prediction')
plt.legend()
plt.grid(True)
plt.savefig('lstm2_alibaba.png')
plt.show()

