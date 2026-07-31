from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


DATASETS = ("cholera", "ilinet", "electricity")
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)
TUNING_SEED = 20260731
ROBUST_K = 3.5
EPS = 1e-12

MAX_EPOCHS = 150
PATIENCE = 15

PARAM_GRID = {
    "architecture": ("small", "medium", "stacked"),
    "batch_size": (16,),
    "learning_rate": (1e-3,),
}


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("_").lower()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    terms = np.zeros_like(y_true, dtype=float)
    mask = denominator > EPS
    terms[mask] = np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
    return float(100.0 * np.mean(terms))


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Forecast metric shape mismatch: {y_true.shape} vs {y_pred.shape}"
        )

    result = {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "sMAPE": smape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }

    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    numerator = float(np.sum((y_true - y_pred) ** 2))
    r2_manual = np.nan if denominator <= EPS else 1.0 - numerator / denominator

    if np.isfinite(r2_manual) and not np.isclose(
        result["R2"], r2_manual, rtol=1e-10, atol=1e-10
    ):
        raise AssertionError(
            f"R² consistency failure: sklearn={result['R2']}, manual={r2_manual}"
        )

    result["R2_manual"] = float(r2_manual)
    return result


def anomaly_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float, float, str]:
    truth = np.asarray(truth, dtype=int)
    predicted = np.asarray(predicted, dtype=int)

    if truth.shape != predicted.shape:
        raise ValueError(
            f"Anomaly metric shape mismatch: {truth.shape} vs {predicted.shape}"
        )

    if int(truth.sum()) == 0:
        return np.nan, np.nan, np.nan, "not_applicable_no_positive_reference"

    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        predicted,
        average="binary",
        zero_division=0,
    )
    return float(precision), float(recall), float(f1), "applicable"


def robust_residual_threshold(
    residuals: np.ndarray,
    *,
    k: float = ROBUST_K,
) -> tuple[float, float, float, str]:
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    if residuals.size < 3:
        raise ValueError("At least three finite calibration residuals are required.")

    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    robust_scale = 1.4826 * mad

    if not np.isfinite(robust_scale) or robust_scale <= EPS:
        std = float(np.std(residuals, ddof=0))
        threshold = median + 3.0 * std
        method = "median_plus_3std_fallback"
    else:
        threshold = median + k * robust_scale
        method = "median_plus_k_scaled_mad"

    if not np.isfinite(threshold):
        raise ValueError("Non-finite residual threshold.")

    return float(threshold), median, mad, method


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    validation_size = max(20, int(round(0.20 * n)))
    validation_size = min(validation_size, n - 20)

    if validation_size <= 0:
        raise ValueError(f"Not enough samples for chronological validation: n={n}")

    split = n - validation_size
    return X[:split], y[:split], X[split:], y[split:]


def architecture_spec(name: str) -> dict:
    specs = {
        "small": {
            "lstm_units": (32,),
            "dropout": 0.0,
            "dense_units": 16,
        },
        "medium": {
            "lstm_units": (64,),
            "dropout": 0.2,
            "dense_units": 16,
        },
        "stacked": {
            "lstm_units": (64, 32),
            "dropout": 0.2,
            "dense_units": 16,
        },
    }
    if name not in specs:
        raise ValueError(f"Unknown architecture: {name}")
    return specs[name]


def build_model(
    window_size: int,
    *,
    architecture: str,
    learning_rate: float,
) -> tf.keras.Model:
    spec = architecture_spec(architecture)

    inputs = tf.keras.Input(shape=(window_size, 1))
    x = inputs

    units = spec["lstm_units"]
    for layer_index, layer_units in enumerate(units):
        return_sequences = layer_index < len(units) - 1
        x = tf.keras.layers.LSTM(
            layer_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=return_sequences,
        )(x)

        if spec["dropout"] > 0:
            x = tf.keras.layers.Dropout(spec["dropout"])(x)

    x = tf.keras.layers.Dense(
        spec["dense_units"],
        activation="relu",
    )(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


def model_size_bytes(model: tf.keras.Model) -> int:
    return int(
        sum(
            np.asarray(weight).nbytes
            for weight in model.get_weights()
        )
    )


def parameter_candidates() -> list[dict]:
    keys = tuple(PARAM_GRID.keys())
    candidates: list[dict] = []

    for values in itertools.product(*(PARAM_GRID[key] for key in keys)):
        candidates.append(dict(zip(keys, values)))

    return candidates


def tune_configuration(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    window_size: int,
    train_std: float,
) -> tuple[dict, int, pd.DataFrame, np.ndarray]:
    X_fit, y_fit, X_val, y_val = chronological_split(X_train, y_train)

    X_fit_lstm = X_fit[..., np.newaxis]
    X_val_lstm = X_val[..., np.newaxis]

    rows: list[dict] = []
    best_config: dict | None = None
    best_epoch = 0
    best_rmse = np.inf
    best_residuals_original: np.ndarray | None = None

    for candidate_index, config in enumerate(parameter_candidates()):
        tf.keras.backend.clear_session()
        set_global_seed(TUNING_SEED)

        model = build_model(
            window_size,
            architecture=config["architecture"],
            learning_rate=config["learning_rate"],
        )

        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            mode="min",
        )

        started = time.perf_counter()
        history = model.fit(
            X_fit_lstm,
            y_fit,
            validation_data=(X_val_lstm, y_val),
            epochs=MAX_EPOCHS,
            batch_size=config["batch_size"],
            shuffle=False,
            verbose=0,
            callbacks=[callback],
        )
        fit_time = time.perf_counter() - started

        validation_prediction = model.predict(
            X_val_lstm,
            verbose=0,
        ).reshape(-1)

        validation_rmse_scaled = rmse(y_val, validation_prediction)
        residuals_original = (
            np.abs(y_val - validation_prediction) * train_std
        )

        best_epoch_candidate = (
            int(np.argmin(history.history["val_loss"])) + 1
        )

        rows.append(
            {
                "candidate_index": candidate_index,
                "architecture": config["architecture"],
                "batch_size": config["batch_size"],
                "learning_rate": config["learning_rate"],
                "best_epoch": best_epoch_candidate,
                "epochs_ran": len(history.history["loss"]),
                "validation_rmse_scaled": validation_rmse_scaled,
                "fit_time_sec": fit_time,
                "model_size_bytes": model_size_bytes(model),
                "n_parameters": int(model.count_params()),
            }
        )

        if validation_rmse_scaled < best_rmse:
            best_rmse = validation_rmse_scaled
            best_config = dict(config)
            best_epoch = best_epoch_candidate
            best_residuals_original = residuals_original.copy()

    if (
        best_config is None
        or best_residuals_original is None
        or best_epoch <= 0
    ):
        raise RuntimeError("LSTM configuration selection failed.")

    tuning_table = pd.DataFrame(rows).sort_values(
        "validation_rmse_scaled",
        ascending=True,
    )

    return (
        best_config,
        best_epoch,
        tuning_table,
        best_residuals_original,
    )


def load_scenarios(
    dataset: str,
    ready_dir: Path,
    anomaly_dir: Path,
    *,
    train_mean: float,
    train_std: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    test = (
        pd.read_csv(
            ready_dir / f"{dataset}_test_scaled.csv",
            parse_dates=["ds"],
        )
        .sort_values("ds")
        .reset_index(drop=True)
    )

    scenarios: dict[str, pd.DataFrame] = {
        "forecast_original": pd.DataFrame(
            {
                "ds": test["ds"],
                "y_clean": test["y"].to_numpy(dtype=float),
                "y_observed": test["y"].to_numpy(dtype=float),
                "y_observed_scaled": test["y_scaled"].to_numpy(dtype=float),
                "anomaly_truth": np.zeros(len(test), dtype=int),
            }
        )
    }

    for rate in (5, 10):
        frame = (
            pd.read_csv(
                anomaly_dir / f"{dataset}_test_injected_{rate}pct.csv",
                parse_dates=["ds"],
            )
            .sort_values("ds")
            .reset_index(drop=True)
        )

        if not np.array_equal(
            frame["ds"].to_numpy(dtype="datetime64[ns]"),
            test["ds"].to_numpy(dtype="datetime64[ns]"),
        ):
            raise ValueError(f"{dataset} injected_{rate}pct: timestamp mismatch")

        expected_scaled = (
            frame["y_anom"].to_numpy(dtype=float) - train_mean
        ) / train_std
        max_scaling_error = float(
            np.max(
                np.abs(
                    expected_scaled
                    - frame["y_anom_scaled"].to_numpy(dtype=float)
                )
            )
        )
        if max_scaling_error > 1e-5:
            raise ValueError(
                f"{dataset} injected_{rate}pct: y_anom_scaled mismatch "
                f"(max error={max_scaling_error:.6g})"
            )

        scenarios[f"injected_{rate}pct"] = pd.DataFrame(
            {
                "ds": frame["ds"],
                "y_clean": frame["y_clean"].to_numpy(dtype=float),
                "y_observed": frame["y_anom"].to_numpy(dtype=float),
                "y_observed_scaled": frame["y_anom_scaled"].to_numpy(dtype=float),
                "anomaly_truth": frame["anomaly_injected"].to_numpy(dtype=int),
            }
        )

    proxy = (
        pd.read_csv(
            anomaly_dir / f"{dataset}_test_proxy_robust_z.csv",
            parse_dates=["ds"],
        )
        .sort_values("ds")
        .reset_index(drop=True)
    )

    if not np.array_equal(
        proxy["ds"].to_numpy(dtype="datetime64[ns]"),
        test["ds"].to_numpy(dtype="datetime64[ns]"),
    ):
        raise ValueError(f"{dataset} proxy_robust_z: timestamp mismatch")

    scenarios["proxy_robust_z"] = pd.DataFrame(
        {
            "ds": proxy["ds"],
            "y_clean": proxy["y"].to_numpy(dtype=float),
            "y_observed": proxy["y"].to_numpy(dtype=float),
            "y_observed_scaled": proxy["y_scaled"].to_numpy(dtype=float),
            "anomaly_truth": proxy["anomaly_proxy_robust"].to_numpy(dtype=int),
        }
    )

    return test, scenarios


def run_scenario(
    model: tf.keras.Model,
    *,
    train_scaled: np.ndarray,
    scenario: pd.DataFrame,
    window_size: int,
    train_mean: float,
    train_std: float,
    threshold: float,
) -> tuple[pd.DataFrame, float]:
    history = list(np.asarray(train_scaled, dtype=float))
    predictions_scaled: list[float] = []
    latencies: list[float] = []

    for observed_scaled in scenario["y_observed_scaled"].to_numpy(dtype=float):
        features = np.asarray(
            history[-window_size:],
            dtype=np.float32,
        ).reshape(1, window_size, 1)

        started = time.perf_counter()
        prediction_scaled = float(
            model(features, training=False).numpy().reshape(-1)[0]
        )
        latencies.append(time.perf_counter() - started)

        predictions_scaled.append(prediction_scaled)
        history.append(float(observed_scaled))

    output = scenario.copy()
    output["y_pred_scaled"] = np.asarray(predictions_scaled, dtype=float)
    output["y_pred"] = output["y_pred_scaled"] * train_std + train_mean
    output["residual_abs"] = np.abs(
        output["y_observed"].to_numpy(dtype=float)
        - output["y_pred"].to_numpy(dtype=float)
    )
    output["anomaly_pred"] = (
        output["residual_abs"].to_numpy(dtype=float) > threshold
    ).astype(int)
    output["threshold"] = threshold

    return output, float(np.mean(latencies))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run corrected full-horizon LSTM experiments."
    )
    parser.add_argument("--ready-dir", required=True)
    parser.add_argument("--anomaly-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    ready_dir = Path(args.ready_dir)
    anomaly_dir = Path(args.anomaly_dir)
    output_dir = Path(args.output_dir)
    predictions_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    threshold_rows: list[dict] = []
    selected_configs: dict[str, dict] = {}

    for dataset in DATASETS:
        print(f"\n================ LSTM: {dataset.upper()} ================")

        npz = np.load(
            ready_dir / f"{dataset}_windows.npz",
            allow_pickle=True,
        )
        X_train = np.asarray(npz["X_train"], dtype=np.float32)
        y_train = np.asarray(npz["y_train"], dtype=np.float32)
        window_size = int(npz["window_size"])
        train_mean = float(npz["train_mean"])
        train_std = float(npz["train_std"])

        train = (
            pd.read_csv(
                ready_dir / f"{dataset}_train_scaled.csv",
                parse_dates=["ds"],
            )
            .sort_values("ds")
            .reset_index(drop=True)
        )
        train_scaled = train["y_scaled"].to_numpy(dtype=float)

        test, scenarios = load_scenarios(
            dataset,
            ready_dir,
            anomaly_dir,
            train_mean=train_mean,
            train_std=train_std,
        )

        (
            best_config,
            selected_epochs,
            tuning_table,
            calibration_residuals,
        ) = tune_configuration(
            X_train,
            y_train,
            window_size=window_size,
            train_std=train_std,
        )

        tuning_table.to_csv(
            output_dir / f"{dataset}_lstm_tuning.csv",
            index=False,
        )

        threshold, calibration_median, calibration_mad, threshold_method = (
            robust_residual_threshold(calibration_residuals)
        )

        selected_configs[dataset] = {
            **best_config,
            "selected_epochs": selected_epochs,
            "window_size": window_size,
            "validation_protocol": "last_20_percent_of_training_windows",
            "final_fit_protocol": "all_training_windows_for_selected_epochs",
            "threshold_calibration": "absolute_chronological_validation_residuals",
            "threshold": threshold,
            "threshold_method": threshold_method,
        }

        print(
            "Selected configuration:",
            {
                "architecture": best_config["architecture"],
                "batch_size": best_config["batch_size"],
                "learning_rate": best_config["learning_rate"],
                "selected_epochs": selected_epochs,
            },
        )
        print(f"Calibrated residual threshold: {threshold:.6g}")

        threshold_rows.append(
            {
                "dataset": dataset,
                "model": "LSTM",
                "threshold": threshold,
                "calibration_residual_median": calibration_median,
                "calibration_residual_mad": calibration_mad,
                "threshold_method": threshold_method,
                "n_calibration_residuals": len(calibration_residuals),
            }
        )

        X_train_lstm = X_train[..., np.newaxis]

        for seed in SEEDS:
            print(f"\n--- seed={seed} ---")
            tf.keras.backend.clear_session()
            set_global_seed(seed)

            model = build_model(
                window_size,
                architecture=best_config["architecture"],
                learning_rate=best_config["learning_rate"],
            )

            started = time.perf_counter()
            model.fit(
                X_train_lstm,
                y_train,
                epochs=selected_epochs,
                batch_size=best_config["batch_size"],
                shuffle=False,
                verbose=0,
            )
            training_time = time.perf_counter() - started
            size_bytes = model_size_bytes(model)

            for scenario_name, scenario in scenarios.items():
                predictions, mean_latency = run_scenario(
                    model,
                    train_scaled=train_scaled,
                    scenario=scenario,
                    window_size=window_size,
                    train_mean=train_mean,
                    train_std=train_std,
                    threshold=threshold,
                )

                if len(predictions) != len(test):
                    raise AssertionError(
                        f"{dataset}/{seed}/{scenario_name}: "
                        f"expected {len(test)} rows, got {len(predictions)}"
                    )

                predictions.to_csv(
                    predictions_dir
                    / f"{dataset}_lstm_seed_{seed}_{scenario_name}.csv",
                    index=False,
                )

                base_row = {
                    "dataset": dataset,
                    "model": "LSTM",
                    "seed": seed,
                    "scenario": scenario_name,
                    "n_eval": len(predictions),
                    "training_time_sec": training_time,
                    "mean_inference_latency_sec": mean_latency,
                    "model_size_bytes": size_bytes,
                    "n_parameters": int(model.count_params()),
                    "threshold": threshold,
                    "architecture": best_config["architecture"],
                    "batch_size": best_config["batch_size"],
                    "learning_rate": best_config["learning_rate"],
                    "selected_epochs": selected_epochs,
                }

                if scenario_name == "forecast_original":
                    metrics = forecast_metrics(
                        predictions["y_clean"].to_numpy(dtype=float),
                        predictions["y_pred"].to_numpy(dtype=float),
                    )
                    raw_rows.append(
                        {
                            **base_row,
                            **metrics,
                            "n_reference_anomalies": 0,
                            "n_detected": int(predictions["anomaly_pred"].sum()),
                            "Precision": np.nan,
                            "Recall": np.nan,
                            "F1": np.nan,
                            "metric_status": "forecast_only",
                        }
                    )

                    print(
                        f"forecast_original | RMSE={metrics['RMSE']:.6g}, "
                        f"MAE={metrics['MAE']:.6g}, "
                        f"sMAPE={metrics['sMAPE']:.3f}, "
                        f"R2={metrics['R2']:.6g}"
                    )
                else:
                    precision, recall, f1, status = anomaly_metrics(
                        predictions["anomaly_truth"].to_numpy(dtype=int),
                        predictions["anomaly_pred"].to_numpy(dtype=int),
                    )
                    raw_rows.append(
                        {
                            **base_row,
                            "RMSE": np.nan,
                            "MAE": np.nan,
                            "sMAPE": np.nan,
                            "R2": np.nan,
                            "R2_manual": np.nan,
                            "n_reference_anomalies": int(
                                predictions["anomaly_truth"].sum()
                            ),
                            "n_detected": int(predictions["anomaly_pred"].sum()),
                            "Precision": precision,
                            "Recall": recall,
                            "F1": f1,
                            "metric_status": status,
                        }
                    )

                    print(
                        f"{scenario_name} | "
                        f"reference={int(predictions['anomaly_truth'].sum())}, "
                        f"detected={int(predictions['anomaly_pred'].sum())}, "
                        f"F1={f1 if np.isfinite(f1) else 'N/A'}"
                    )

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(output_dir / "lstm_raw_results.csv", index=False)

    forecast_summary = (
        raw[raw["scenario"] == "forecast_original"]
        .groupby(["dataset", "model"], as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            sMAPE_mean=("sMAPE", "mean"),
            sMAPE_std=("sMAPE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            training_time_mean_sec=("training_time_sec", "mean"),
            training_time_std_sec=("training_time_sec", "std"),
            inference_latency_mean_sec=("mean_inference_latency_sec", "mean"),
            model_size_mean_bytes=("model_size_bytes", "mean"),
            n_parameters_mean=("n_parameters", "mean"),
        )
    )
    forecast_summary.to_csv(
        output_dir / "lstm_forecast_summary.csv",
        index=False,
    )

    anomaly_summary = (
        raw[raw["scenario"] != "forecast_original"]
        .groupby(["dataset", "model", "scenario"], as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            n_reference_anomalies=("n_reference_anomalies", "first"),
            n_detected_mean=("n_detected", "mean"),
            Precision_mean=("Precision", "mean"),
            Precision_std=("Precision", "std"),
            Recall_mean=("Recall", "mean"),
            Recall_std=("Recall", "std"),
            F1_mean=("F1", "mean"),
            F1_std=("F1", "std"),
            metric_status=("metric_status", "first"),
        )
    )
    anomaly_summary.to_csv(
        output_dir / "lstm_anomaly_summary.csv",
        index=False,
    )

    pd.DataFrame(threshold_rows).to_csv(
        output_dir / "lstm_residual_thresholds.csv",
        index=False,
    )

    (output_dir / "lstm_selected_configs.json").write_text(
        json.dumps(selected_configs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metadata = {
        "tensorflow_version": tf.__version__,
        "seeds": list(SEEDS),
        "tuning_seed": TUNING_SEED,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "protocol": "fixed_fit_full_test_one_step_ahead_with_observed_history",
        "hyperparameter_selection": "chronological holdout within training only",
        "final_training": "all training windows for selected epoch count",
        "anomaly_threshold_calibration": (
            "absolute residuals on chronological training holdout"
        ),
        "parameter_grid": PARAM_GRID,
    }
    (output_dir / "lstm_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== LSTM completed =====")
    print("\nForecast summary:")
    print(forecast_summary.to_string(index=False))
    print("\nAnomaly summary:")
    print(anomaly_summary.to_string(index=False))


if __name__ == "__main__":
    main()
