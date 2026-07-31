import os
import time
import warnings
import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings("ignore")

BASE_DIR = "/content/drive/MyDrive/Colab_Projects/Cholera_Forecasting/data/processed"
READY_DIR = os.path.join(BASE_DIR, "ready_for_models")
ANOM_DIR = os.path.join(BASE_DIR, "anomaly_redesign")
OUT_DIR = os.path.join(BASE_DIR, "baseline_fair_results")
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = ["cholera", "ilinet", "electricity"]
SEEDS = [11, 22, 33, 44, 55]

ARIMA_ORDERS = [
    (1, 0, 0), (1, 1, 0),
    (2, 1, 0), (1, 1, 1),
    (2, 1, 1), (3, 1, 0)
]


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
        "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    }


def anomaly_metrics(y_true_flags, y_pred_flags):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_flags,
        y_pred_flags,
        average="binary",
        zero_division=0
    )
    return {
        "Precision": p,
        "Recall": r,
        "F1": f1
    }


def load_train_test(dataset):
    train = pd.read_csv(os.path.join(READY_DIR, f"{dataset}_train_scaled.csv"), parse_dates=["ds"])
    test = pd.read_csv(os.path.join(READY_DIR, f"{dataset}_test_scaled.csv"), parse_dates=["ds"])
    return train, test


def select_arima_order(history):
    best_order = None
    best_aic = np.inf

    for order in ARIMA_ORDERS:
        try:
            fit = ARIMA(history, order=order).fit()
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_order = order
        except Exception:
            continue

    if best_order is None:
        best_order = (1, 1, 0)

    return best_order


def walk_forward_arima(train_y, test_y):
    history = list(train_y)
    preds = []
    times = []

    order = select_arima_order(history)

    for t in range(len(test_y)):
        start = time.time()

        try:
            model = ARIMA(history, order=order)
            fit = model.fit()
            yhat = fit.forecast()[0]
        except Exception:
            yhat = history[-1]

        preds.append(float(yhat))
        times.append(time.time() - start)

        history.append(float(test_y[t]))

    return np.array(preds), np.mean(times), order


def make_windows(series, window):
    X = []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
    return np.array(X)


def walk_forward_isolation_forest(train_y, test_y, window, seed=42, contamination=0.05):
    history = list(train_y)
    preds = []
    flags = []
    times = []

    for t in range(len(test_y)):
        start = time.time()

        if len(history) <= window + 5:
            yhat = history[-1]
            flag = 0
        else:
            X_train = make_windows(np.array(history), window)

            model = IsolationForest(
                contamination=contamination,
                random_state=seed
            )
            model.fit(X_train)

            current_window = np.array(history[-window:]).reshape(1, -1)
            pred_flag = model.predict(current_window)[0]
            flag = 1 if pred_flag == -1 else 0

            if flag == 0:
                yhat = history[-1]
            else:
                yhat = np.mean(history[-min(3, len(history)):])

        preds.append(float(yhat))
        flags.append(int(flag))
        times.append(time.time() - start)

        history.append(float(test_y[t]))

    return np.array(preds), np.array(flags), np.mean(times)


def residual_anomaly_flags(train_y, y_true, y_pred):
    naive_train_errors = np.abs(np.diff(train_y))
    mad = np.median(np.abs(naive_train_errors - np.median(naive_train_errors)))

    if mad == 0:
        threshold = np.mean(naive_train_errors) + 3 * np.std(naive_train_errors)
    else:
        threshold = 3.5 * mad

    errors = np.abs(y_true - y_pred)
    return (errors > threshold).astype(int)


def load_anomaly_truth(dataset, scenario, test_len):
    if scenario == "real":
        path = os.path.join(ANOM_DIR, f"{dataset}_test_real_anomalies.csv")
        df = pd.read_csv(path)
        return df["anomaly_real_robust"].values[:test_len]

    if scenario == "injected_5pct":
        path = os.path.join(ANOM_DIR, f"{dataset}_test_injected_5pct.csv")
        df = pd.read_csv(path)
        return df["anomaly_injected"].values[:test_len]

    if scenario == "injected_10pct":
        path = os.path.join(ANOM_DIR, f"{dataset}_test_injected_10pct.csv")
        df = pd.read_csv(path)
        return df["anomaly_injected"].values[:test_len]

    raise ValueError("Unknown anomaly scenario")


all_results = []

for dataset in DATASETS:
    print(f"\n================ {dataset.upper()} ================")

    train, test = load_train_test(dataset)

    train_y = train["y"].values.astype(float)
    test_y = test["y"].values.astype(float)

    meta = np.load(os.path.join(READY_DIR, f"{dataset}_windows.npz"), allow_pickle=True)
    window = int(meta["window_size"])

    # ---------------- ARIMA ----------------
    arima_pred, arima_time, arima_order = walk_forward_arima(train_y, test_y)
    fm = forecast_metrics(test_y, arima_pred)

    arima_anom_pred = residual_anomaly_flags(train_y, test_y, arima_pred)

    row = {
        "dataset": dataset,
        "model": f"ARIMA{arima_order}",
        "seed": "none",
        "scenario": "forecast_original",
        **fm,
        "Precision": np.nan,
        "Recall": np.nan,
        "F1": np.nan,
        "avg_time_sec": arima_time
    }
    all_results.append(row)

    print(f"ARIMA{arima_order}: RMSE={fm['RMSE']:.4f}, MAE={fm['MAE']:.4f}, sMAPE={fm['sMAPE']:.2f}%")

    # anomaly scenarios for ARIMA
    for scenario in ["real", "injected_5pct", "injected_10pct"]:
        y_true_flags = load_anomaly_truth(dataset, scenario, len(test_y))

        if y_true_flags.sum() == 0:
            print(f"ARIMA anomaly {scenario}: skipped because true anomalies = 0")
            continue

        am = anomaly_metrics(y_true_flags, arima_anom_pred)

        all_results.append({
            "dataset": dataset,
            "model": f"ARIMA{arima_order}",
            "seed": "none",
            "scenario": scenario,
            "RMSE": np.nan,
            "MAE": np.nan,
            "sMAPE": np.nan,
            "R2": np.nan,
            **am,
            "avg_time_sec": arima_time
        })

    # ---------------- Isolation Forest ----------------
    for seed in SEEDS:
        if_pred, if_flags, if_time = walk_forward_isolation_forest(
            train_y=train_y,
            test_y=test_y,
            window=window,
            seed=seed,
            contamination=0.05
        )

        fm = forecast_metrics(test_y, if_pred)

        all_results.append({
            "dataset": dataset,
            "model": "IsolationForest",
            "seed": seed,
            "scenario": "forecast_original",
            **fm,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "avg_time_sec": if_time
        })

        print(f"IF seed={seed}: RMSE={fm['RMSE']:.4f}, MAE={fm['MAE']:.4f}, sMAPE={fm['sMAPE']:.2f}%")

        for scenario in ["real", "injected_5pct", "injected_10pct"]:
            y_true_flags = load_anomaly_truth(dataset, scenario, len(test_y))

            if y_true_flags.sum() == 0:
                print(f"IF anomaly {scenario}: skipped because true anomalies = 0")
                continue

            am = anomaly_metrics(y_true_flags, if_flags)

            all_results.append({
                "dataset": dataset,
                "model": "IsolationForest",
                "seed": seed,
                "scenario": scenario,
                "RMSE": np.nan,
                "MAE": np.nan,
                "sMAPE": np.nan,
                "R2": np.nan,
                **am,
                "avg_time_sec": if_time
            })


results_df = pd.DataFrame(all_results)
results_path = os.path.join(OUT_DIR, "baseline_fair_raw_results.csv")
results_df.to_csv(results_path, index=False)

summary = (
    results_df
    .groupby(["dataset", "model", "scenario"], dropna=False)
    .agg({
        "RMSE": ["mean", "std"],
        "MAE": ["mean", "std"],
        "sMAPE": ["mean", "std"],
        "R2": ["mean", "std"],
        "Precision": ["mean", "std"],
        "Recall": ["mean", "std"],
        "F1": ["mean", "std"],
        "avg_time_sec": ["mean", "std"]
    })
)

summary_path = os.path.join(OUT_DIR, "baseline_fair_summary.csv")
summary.to_csv(summary_path)

print("\n===== Stage 3 completed =====")
print("Raw results saved to:", results_path)
print("Summary saved to:", summary_path)