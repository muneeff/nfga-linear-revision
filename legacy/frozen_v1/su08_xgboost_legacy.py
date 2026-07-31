# su08_xgboost.py

import os
import time
import warnings
import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_recall_fscore_support
)

warnings.filterwarnings("ignore")

BASE_DIR = "/content/drive/MyDrive/Colab_Projects/Cholera_Forecasting/data/processed"

READY_DIR = os.path.join(BASE_DIR, "ready_for_models")
ANOM_DIR = os.path.join(BASE_DIR, "anomaly_redesign")

OUT_DIR = os.path.join(BASE_DIR, "xgboost_results")
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = ["cholera", "ilinet", "electricity"]
SEEDS = [11, 22, 33, 44, 55]


# ==========================================================
# Metrics
# ==========================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def smape(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denom = (
        np.abs(y_true) + np.abs(y_pred)
    ) / 2

    out = np.zeros_like(y_true, dtype=float)

    mask = denom != 0

    out[mask] = (
        np.abs(
            y_true[mask] -
            y_pred[mask]
        ) / denom[mask]
    )

    return 100 * np.mean(out)


def forecast_metrics(y_true, y_pred):

    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }


def anomaly_metrics(
        y_true_flags,
        y_pred_flags):

    p, r, f1, _ = precision_recall_fscore_support(
        y_true_flags,
        y_pred_flags,
        average="binary",
        zero_division=0
    )

    return p, r, f1


# ==========================================================
# Anomaly Truth Loader
# ==========================================================

def load_anomaly_truth(
        dataset,
        scenario,
        test_len):

    if scenario == "real":

        path = os.path.join(
            ANOM_DIR,
            f"{dataset}_test_real_anomalies.csv"
        )

        df = pd.read_csv(path)

        return (
            df["anomaly_real_robust"]
            .values[:test_len]
        )

    if scenario == "injected_5pct":

        path = os.path.join(
            ANOM_DIR,
            f"{dataset}_test_injected_5pct.csv"
        )

        df = pd.read_csv(path)

        return (
            df["anomaly_injected"]
            .values[:test_len]
        )

    if scenario == "injected_10pct":

        path = os.path.join(
            ANOM_DIR,
            f"{dataset}_test_injected_10pct.csv"
        )

        df = pd.read_csv(path)

        return (
            df["anomaly_injected"]
            .values[:test_len]
        )

    raise ValueError(
        "Unknown anomaly scenario"
    )


# ==========================================================
# MAD anomaly detector
# ==========================================================

def detect_anomalies(
        train_errors,
        test_errors):

    med = np.median(train_errors)

    mad = np.median(
        np.abs(
            train_errors - med
        )
    )

    if mad == 0:

        threshold = (
            np.mean(train_errors)
            + 3 * np.std(train_errors)
        )

    else:

        threshold = med + 3.5 * mad

    flags = (
        test_errors > threshold
    ).astype(int)

    return flags, threshold


# ==========================================================
# Main Loop
# ==========================================================

all_results = []

for dataset in DATASETS:

    print(
        f"\n================ "
        f"XGBOOST: {dataset.upper()} "
        f"================"
    )

    data = np.load(
        os.path.join(
            READY_DIR,
            f"{dataset}_windows.npz"
        ),
        allow_pickle=True
    )

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_test = data["X_test"]
    y_test = data["y_test"]

    train_mean = float(
        data["train_mean"]
    )

    train_std = float(
        data["train_std"]
    )

    y_test_original = (
        y_test * train_std
        + train_mean
    )

    for seed in SEEDS:

        print(
            f"\n--- seed={seed} ---"
        )

        start = time.time()

        model = XGBRegressor(

            n_estimators=300,
            max_depth=4,

            learning_rate=0.05,

            subsample=0.8,
            colsample_bytree=0.8,

            objective="reg:squarederror",

            random_state=seed,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        train_time = (
            time.time() - start
        )

        pred_scaled = model.predict(
            X_test
        )

        pred_original = (
            pred_scaled
            * train_std
            + train_mean
        )

        fm = forecast_metrics(
            y_test_original,
            pred_original
        )

        train_pred_scaled = (
            model.predict(X_train)
        )

        train_errors = np.abs(

            (
                y_train
                * train_std
                + train_mean
            )

            -

            (
                train_pred_scaled
                * train_std
                + train_mean
            )
        )

        test_errors = np.abs(
            y_test_original
            - pred_original
        )

        anomaly_pred, threshold = (
            detect_anomalies(
                train_errors,
                test_errors
            )
        )

        avg_time = (
            train_time
            / len(y_test_original)
        )

        print(
            f"RMSE={fm['RMSE']:.4f}, "
            f"MAE={fm['MAE']:.4f}, "
            f"sMAPE={fm['sMAPE']:.2f}%, "
            f"R2={fm['R2']:.4f}, "
            f"detected={int(anomaly_pred.sum())}"
        )

        all_results.append({

            "dataset": dataset,
            "model": "XGBoost",

            "seed": seed,

            "scenario":
                "forecast_original",

            **fm,

            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,

            "avg_time_sec":
                avg_time,

            "error_threshold":
                threshold
        })

        for scenario in [

            "real",
            "injected_5pct",
            "injected_10pct"

        ]:

            truth = load_anomaly_truth(
                dataset,
                scenario,
                len(y_test_original)
            )

            if truth.sum() == 0:

                print(
                    f"Anomaly {scenario}: "
                    f"skipped because "
                    f"true anomalies = 0"
                )

                continue

            p, r, f1 = anomaly_metrics(
                truth,
                anomaly_pred
            )

            all_results.append({

                "dataset": dataset,
                "model": "XGBoost",

                "seed": seed,
                "scenario": scenario,

                "RMSE": np.nan,
                "MAE": np.nan,
                "sMAPE": np.nan,
                "R2": np.nan,

                "Precision": p,
                "Recall": r,
                "F1": f1,

                "avg_time_sec":
                    avg_time,

                "error_threshold":
                    threshold
            })

        pred_df = pd.DataFrame({

            "y_true":
                y_test_original,

            "y_pred":
                pred_original,

            "error_abs":
                test_errors,

            "anomaly_pred":
                anomaly_pred
        })

        pred_df.to_csv(

            os.path.join(
                OUT_DIR,
                f"{dataset}_xgb_predictions_seed_{seed}.csv"
            ),

            index=False
        )


# ==========================================================
# Save Results
# ==========================================================

results_df = pd.DataFrame(
    all_results
)

raw_path = os.path.join(
    OUT_DIR,
    "xgboost_raw_results.csv"
)

results_df.to_csv(
    raw_path,
    index=False
)

summary = (
    results_df
    .groupby(
        [
            "dataset",
            "model",
            "scenario"
        ],
        dropna=False
    )
    .agg({

        "RMSE":
            ["mean", "std"],

        "MAE":
            ["mean", "std"],

        "sMAPE":
            ["mean", "std"],

        "R2":
            ["mean", "std"],

        "Precision":
            ["mean", "std"],

        "Recall":
            ["mean", "std"],

        "F1":
            ["mean", "std"],

        "avg_time_sec":
            ["mean", "std"]
    })
)

summary_path = os.path.join(
    OUT_DIR,
    "xgboost_summary.csv"
)

summary.to_csv(
    summary_path
)

print(
    "\n===== XGBOOST completed ====="
)

print(
    "Raw results saved to:",
    raw_path
)

print(
    "Summary saved to:",
    summary_path
)