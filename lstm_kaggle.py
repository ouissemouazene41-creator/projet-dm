import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ── 1. LOAD DATA ──────────────────────────────────────────────
alibaba = pd.read_csv('/kaggle/input/alibaba-cluster-trace/machine_usage.csv')
alibaba = alibaba[['timestamp', 'cpu_util_percent', 'mem_util_percent']].dropna()
alibaba['minute'] = (alibaba['timestamp'] // 60).astype(int)
alibaba = alibaba.groupby('minute').mean().reset_index()

# ── 2. NORMALIZE ──────────────────────────────────────────────
scaler = MinMaxScaler()
alibaba[['cpu_util_percent', 'mem_util_percent']] = scaler.fit_transform(
    alibaba[['cpu_util_percent', 'mem_util_percent']]
)

# ── 3. BUILD SLIDING WINDOWS (last 30 min → predict next) ────
def make_windows(data, window=30):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window][0])
    return np.array(X), np.array(y)

features = alibaba[['cpu_util_percent', 'mem_util_percent']].values
X, y = make_windows(features)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ── 4. BUILD & TRAIN LSTM ─────────────────────────────────────
model = Sequential([
    LSTM(64, input_shape=(30, 2)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# ── 5. EVALUATE ───────────────────────────────────────────────
pred = model.predict(X_test)

pred_real = scaler.inverse_transform(
    np.column_stack([pred, np.zeros(len(pred))])
)[:, 0]

y_real = scaler.inverse_transform(
    np.column_stack([y_test, np.zeros(len(y_test))])
)[:, 0]

rmse = np.sqrt(np.mean((pred_real - y_real) ** 2))
print(f"RMSE: {rmse:.2f}% CPU")

