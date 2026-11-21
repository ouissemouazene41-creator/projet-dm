# Datacenter Workload Forecasting with LSTM

Predicting server CPU workload using Deep Learning to enable proactive energy-efficient cooling in datacenters.

## 📊 Dataset Used
* **Source:** [Cloud Workload Job Traces for Resource Forecasting (Kaggle - zoya77)](https://www.kaggle.com/datasets/zoya77/cloud-workload-job-traces-for-resource-forecasting)
* **File:** `cloud_workload_dataset.csv`
* **Features:**
  * `Requested_CPUs` & `Used_CPUs`
  * `Requested_Memory(MB)` & `Used_Memory(MB)`
  * `Execution_Time(Seconds)` & `Queue_Wait_Time(Seconds)`

## 🧠 Model Architecture
* **Input Layer:** `Input(shape=(30, n_features))` — 30-timestep sliding window.
* **LSTM Layer 1:** 64 units + Dropout (0.2).
* **LSTM Layer 2:** 32 units + Dropout (0.2).
* **Dense Layer:** 16 units (ReLU).
* **Output Layer:** 1 unit (Next CPU utilization forecast).
* **Optimizer & Loss:** Adam, Mean Squared Error (MSE).

## 📈 Performance Results
* **Training samples:** 2,825 windows
* **Final Validation Loss:** `0.0415`
* **Test RMSE:** `0.2038` (Normalized scale)

## 🚀 Execution
```bash
python lstm_model.py
```
Outputs prediction metrics and saves `prediction_curve.png` showing the actual vs. predicted workload.
