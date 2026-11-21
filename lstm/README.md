# Datacenter Workload Forecasting with LSTM

This component implements the **LSTM neural network** for predicting future CPU and memory utilization in data center servers.

## 🎯 Objective
Accurately forecast server workload spikes (5 to 10 minutes ahead) based on historical time-series traces. This enables proactive cooling adjustments to prevent server overheating while saving energy.

## 🧠 Model Architecture
* **Input**: Sliding window of past 30 timesteps with 2 features (`CPU Usage %`, `Memory Usage %`).
* **LSTM Layer 1**: 64 units with Dropout (0.2).
* **LSTM Layer 2**: 32 units with Dropout (0.2).
* **Dense Layer**: 16 units (ReLU activation).
* **Output Layer**: 1 unit predicting the next CPU workload point.
* **Loss Function**: Mean Squared Error (MSE), Adam optimizer.

## 🚀 How to Run
```bash
python lstm_model.py
```
*(The script automatically detects Kaggle CSV datasets or generates a benchmark workload trace if no file is found).*
