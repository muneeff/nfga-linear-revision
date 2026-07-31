# su09_lstm.py

import os
import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_recall_fscore_support

warnings.filterwarnings("ignore")

# -------------------- Directories --------------------
BASE_DIR = "/content/drive/MyDrive/Colab_Projects/Cholera_Forecasting/data/processed"
READY_DIR = os.path.join(BASE_DIR, "ready_for_models")
ANOM_DIR = os.path.join(BASE_DIR, "anomaly_redesign")
OUT_DIR = os.path.join(BASE_DIR, "lstm_results")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------- Datasets & seeds --------------------
DATASETS = ["cholera", "ilinet", "electricity"]
SEEDS = [11, 22, 33, 44, 55]

# -------------------- Metrics --------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    out = np.zeros_like(y_true, dtype=float)
    mask = denom != 0
    out[mask] = np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return 100 * np.mean(out)

def forecast_metrics(y_true, y_pred):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

def anomaly_metrics(y_true_flags, y_pred_flags):
    p, r, f1, _ = precision_recall_fscore_support(y_true_flags, y_pred_flags, average="binary", zero_division=0)
    return p, r, f1

def load_anomaly_truth(dataset, scenario, test_len):
    if scenario == "real":
        path = os.path.join(ANOM_DIR, f"{dataset}_test_real_anomalies.csv")
        df = pd.read_csv(path)
        return df["anomaly_real_robust"].values[:test_len]
    elif scenario == "injected_5pct":
        path = os.path.join(ANOM_DIR, f"{dataset}_test_injected_5pct.csv")
        df = pd.read_csv(path)
        return df["anomaly_injected"].values[:test_len]
    elif scenario == "injected_10pct":
        path = os.path.join(ANOM_DIR, f"{dataset}_test_injected_10pct.csv")
        df = pd.read_csv(path)
        return df["anomaly_injected"].values[:test_len]
    else:
        raise ValueError("Unknown anomaly scenario")

def detect_anomalies(train_errors, test_errors):
    med = np.median(train_errors)
    mad = np.median(np.abs(train_errors - med))
    if mad == 0:
        threshold = np.mean(train_errors) + 3 * np.std(train_errors)
    else:
        threshold = med + 3.5 * mad
    flags = (np.abs(test_errors) > threshold).astype(int)
    return flags, threshold

# -------------------- Main Loop --------------------
all_results = []

for dataset in DATASETS:
    print(f"\n================ LSTM: {dataset.upper()} ================")

    # Load data
    data = np.load(os.path.join(READY_DIR, f"{dataset}_windows.npz"), allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test_scaled = data["y_test"]
    train_mean = float(data["train_mean"])
    train_std = float(data["train_std"])
    eval_len = len(y_test_scaled)

    y_test_original = y_test_scaled * train_std + train_mean

    # Reshape for LSTM [samples, timesteps, features]
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    for seed in SEEDS:
        tf.random.set_seed(seed)
        np.random.seed(seed)

        model = Sequential([
            LSTM(64, activation='tanh', return_sequences=True, input_shape=(X_train_lstm.shape[1],1)),
            Dropout(0.2),
            LSTM(32, activation='tanh'),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        start = time.time()
        model.fit(X_train_lstm, y_train, epochs=100, batch_size=16, validation_split=0.1, verbose=0, callbacks=[es])
        train_time = time.time() - start

        # Forecast
        pred_scaled = model.predict(X_test_lstm).flatten()
        pred_original = pred_scaled * train_std + train_mean

        # Training residuals for anomaly detection
        train_pred_scaled = model.predict(X_train_lstm).flatten()
        train_errors = np.abs((y_train * train_std + train_mean) - (train_pred_scaled * train_std + train_mean))
        test_errors = np.abs(y_test_original - pred_original)
        anomaly_pred, threshold = detect_anomalies(train_errors, test_errors)

        fm = forecast_metrics(y_test_original, pred_original)
        print(f"seed={seed} | RMSE={fm['RMSE']:.4f}, MAE={fm['MAE']:.4f}, sMAPE={fm['sMAPE']:.2f}%, R2={fm['R2']:.4f}, detected={int(anomaly_pred.sum())}")

        # Save results
        all_results.append({
            "dataset": dataset,
            "model": "LSTM",
            "seed": seed,
            "scenario": "forecast_original",
            **fm,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "avg_time_sec": train_time / max(1,len(y_test_original)),
            "error_threshold": threshold
        })

        # Evaluate anomalies
        for scenario in ["real", "injected_5pct", "injected_10pct"]:
            truth = load_anomaly_truth(dataset, scenario, eval_len)
            if truth.sum() == 0:
                print(f"Anomaly {scenario}: skipped because true anomalies = 0")
                continue
            p,r,f1 = anomaly_metrics(truth, anomaly_pred)
            all_results.append({
                "dataset": dataset,
                "model": "LSTM",
                "seed": seed,
                "scenario": scenario,
                "RMSE": np.nan,
                "MAE": np.nan,
                "sMAPE": np.nan,
                "R2": np.nan,
                "Precision": p,
                "Recall": r,
                "F1": f1,
                "avg_time_sec": train_time / max(1,len(y_test_original)),
                "error_threshold": threshold
            })

        # Save predictions
        pred_df = pd.DataFrame({
            "y_true": y_test_original,
            "y_pred": pred_original,
            "error_abs": test_errors,
            "anomaly_pred": anomaly_pred
        })

        pred_df.to_csv(os.path.join(OUT_DIR, f"{dataset}_lstm_predictions_seed_{seed}.csv"), index=False)

# Save raw and summary
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(OUT_DIR,"lstm_raw_results.csv"), index=False)

summary = results_df.groupby(["dataset","model","scenario"], dropna=False).agg({
    "RMSE":["mean","std"],
    "MAE":["mean","std"],
    "sMAPE":["mean","std"],
    "R2":["mean","std"],
    "Precision":["mean","std"],
    "Recall":["mean","std"],
    "F1":["mean","std"],
    "avg_time_sec":["mean","std"]
})
summary.to_csv(os.path.join(OUT_DIR,"lstm_summary.csv"))

print("\n===== LSTM completed =====")
print("Raw results saved to:", os.path.join(OUT_DIR,"lstm_raw_results.csv"))
print("Summary saved to:", os.path.join(OUT_DIR,"lstm_summary.csv"))