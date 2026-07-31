from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


DATASETS = ("cholera", "ilinet", "electricity")
SCENARIOS = (
    "forecast_original",
    "injected_5pct",
    "injected_10pct",
    "proxy_robust_z",
)

ROBUST_K = 3.5
EPS = 1e-12

CANDIDATE_CONFIGS = (
    {
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 1.0,
        "seasonality_mode": "additive",
    },
    {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 1.0,
        "seasonality_mode": "additive",
    },
    {
        "changepoint_prior_scale": 0.10,
        "seasonality_prior_scale": 1.0,
        "seasonality_mode": "additive",
    },
    {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "seasonality_mode": "additive",
    },
    {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 1.0,
        "seasonality_mode": "multiplicative",
    },
)


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


def forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    metrics = {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "sMAPE": smape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }

    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    numerator = float(np.sum((y_true - y_pred) ** 2))
    manual = np.nan if denominator <= EPS else 1.0 - numerator / denominator

    if np.isfinite(manual) and not np.isclose(
        metrics["R2"],
        manual,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise AssertionError(
            f"R² mismatch: sklearn={metrics['R2']}, manual={manual}"
        )

    metrics["R2_manual"] = float(manual)
    return metrics


def anomaly_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int | str]:
    truth = np.asarray(truth, dtype=int)
    predicted = np.asarray(predicted, dtype=int)

    tp = int(np.sum((truth == 1) & (predicted == 1)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))

    positives = tp + fn
    negatives = tn + fp
    fpr = float(fp / negatives) if negatives else np.nan
    specificity = float(tn / negatives) if negatives else np.nan

    if positives == 0:
        return {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "n_reference_anomalies": positives,
            "n_detected": int(predicted.sum()),
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "Specificity": specificity,
            "FalsePositiveRate": fpr,
            "metric_status": "not_applicable_no_positive_reference",
        }

    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        predicted,
        average="binary",
        zero_division=0,
    )

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "n_reference_anomalies": positives,
        "n_detected": int(predicted.sum()),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "Specificity": specificity,
        "FalsePositiveRate": fpr,
        "metric_status": "applicable",
    }


def robust_residual_threshold(
    absolute_residuals: np.ndarray,
) -> tuple[float, float, float, str]:
    residuals = np.asarray(absolute_residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    if residuals.size < 3:
        raise ValueError("At least three calibration residuals are required.")

    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    robust_scale = 1.4826 * mad

    if not np.isfinite(robust_scale) or robust_scale <= EPS:
        std = float(np.std(residuals, ddof=0))
        threshold = median + 3.0 * std
        method = "median_plus_3std_fallback"
    else:
        threshold = median + ROBUST_K * robust_scale
        method = "median_plus_k_scaled_mad"

    return float(threshold), median, mad, method


def build_prophet(config: dict) -> Prophet:
    return Prophet(
        growth="linear",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=float(
            config["changepoint_prior_scale"]
        ),
        seasonality_prior_scale=float(
            config["seasonality_prior_scale"]
        ),
        seasonality_mode=str(config["seasonality_mode"]),
        interval_width=0.80,
        uncertainty_samples=0,
    )


def fit_prophet(
    history: pd.DataFrame,
    config: dict,
) -> Prophet:
    model = build_prophet(config)
    model.fit(
        history[["ds", "y"]].copy()
    )
    return model


def predict_dates(
    model: Prophet,
    dates: pd.Series,
) -> np.ndarray:
    future = pd.DataFrame({"ds": pd.to_datetime(dates)})
    prediction = model.predict(future)["yhat"].to_numpy(dtype=float)

    # All three study series are nonnegative count/rate/consumption signals.
    return np.maximum(prediction, 0.0)


def chronological_train_validation(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(train)
    validation_size = max(20, int(round(0.20 * n)))
    validation_size = min(validation_size, n - 40)

    if validation_size <= 0:
        raise ValueError(
            f"Insufficient observations for validation: n={n}"
        )

    split = n - validation_size
    return (
        train.iloc[:split].copy(),
        train.iloc[split:].copy(),
    )


def select_configuration(
    train: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    fit_frame, validation_frame = chronological_train_validation(train)
    rows: list[dict] = []
    best_config: dict | None = None
    best_rmse = np.inf
    best_residuals: np.ndarray | None = None

    for candidate_index, config in enumerate(CANDIDATE_CONFIGS):
        started = time.perf_counter()

        try:
            model = fit_prophet(fit_frame, config)
            prediction = predict_dates(
                model,
                validation_frame["ds"],
            )
            score = rmse(
                validation_frame["y"].to_numpy(dtype=float),
                prediction,
            )
            status = "ok"
            error_message = ""
            residuals = np.abs(
                validation_frame["y"].to_numpy(dtype=float)
                - prediction
            )

            if score < best_rmse:
                best_rmse = score
                best_config = dict(config)
                best_residuals = residuals.copy()

        except Exception as exc:
            score = np.nan
            status = f"failed:{type(exc).__name__}"
            error_message = str(exc)
            residuals = np.asarray([], dtype=float)

        rows.append(
            {
                "candidate_index": candidate_index,
                **config,
                "validation_RMSE": score,
                "fit_time_sec": time.perf_counter() - started,
                "status": status,
                "error_message": error_message,
                "n_fit": len(fit_frame),
                "n_validation": len(validation_frame),
            }
        )

    if best_config is None or best_residuals is None:
        raise RuntimeError("All Prophet candidate configurations failed.")

    return (
        best_config,
        pd.DataFrame(rows).sort_values(
            "validation_RMSE",
            na_position="last",
        ),
        best_residuals,
    )


def load_scenarios(
    dataset: str,
    ready_dir: Path,
    anomaly_dir: Path,
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
        scenarios[f"injected_{rate}pct"] = pd.DataFrame(
            {
                "ds": frame["ds"],
                "y_clean": frame["y_clean"].to_numpy(dtype=float),
                "y_observed": frame["y_anom"].to_numpy(dtype=float),
                "anomaly_truth": frame[
                    "anomaly_injected"
                ].to_numpy(dtype=int),
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
    scenarios["proxy_robust_z"] = pd.DataFrame(
        {
            "ds": proxy["ds"],
            "y_clean": proxy["y"].to_numpy(dtype=float),
            "y_observed": proxy["y"].to_numpy(dtype=float),
            "anomaly_truth": proxy[
                "anomaly_proxy_robust"
            ].to_numpy(dtype=int),
        }
    )

    for name, frame in scenarios.items():
        if not np.array_equal(
            pd.to_datetime(frame["ds"]).to_numpy(dtype="datetime64[ns]"),
            test["ds"].to_numpy(dtype="datetime64[ns]"),
        ):
            raise ValueError(
                f"{dataset}/{name}: timestamp alignment failure"
            )

    return test, scenarios


def walk_forward_scenario(
    train: pd.DataFrame,
    scenario: pd.DataFrame,
    config: dict,
    threshold: float,
) -> tuple[pd.DataFrame, float, float, int]:
    history = train[["ds", "y"]].copy()
    predictions: list[float] = []
    fit_times: list[float] = []
    inference_times: list[float] = []
    final_model_size = 0

    for row in scenario.itertuples(index=False):
        started_fit = time.perf_counter()
        model = fit_prophet(history, config)
        fit_times.append(time.perf_counter() - started_fit)

        started_predict = time.perf_counter()
        prediction = float(
            predict_dates(
                model,
                pd.Series([pd.Timestamp(row.ds)]),
            )[0]
        )
        inference_times.append(time.perf_counter() - started_predict)
        predictions.append(prediction)

        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    {
                        "ds": [pd.Timestamp(row.ds)],
                        "y": [float(row.y_observed)],
                    }
                ),
            ],
            ignore_index=True,
        )

        final_model_size = len(
            pickle.dumps(
                model,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        )

    output = scenario.copy()
    output["y_pred"] = np.asarray(predictions, dtype=float)
    output["residual_abs"] = np.abs(
        output["y_observed"].to_numpy(dtype=float)
        - output["y_pred"].to_numpy(dtype=float)
    )
    output["anomaly_pred"] = (
        output["residual_abs"].to_numpy(dtype=float) > threshold
    ).astype(int)
    output["threshold"] = threshold

    return (
        output,
        float(np.mean(fit_times)),
        float(np.mean(inference_times)),
        int(final_model_size),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a corrected, training-selected Prophet baseline under the "
            "full one-step-ahead walk-forward protocol."
        )
    )
    parser.add_argument("--ready-dir", required=True)
    parser.add_argument("--anomaly-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    ready_dir = Path(args.ready_dir)
    anomaly_dir = Path(args.anomaly_dir)
    output_dir = Path(args.output_dir)
    prediction_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    forecast_rows: list[dict] = []
    anomaly_rows: list[dict] = []
    threshold_rows: list[dict] = []
    selected_configs: dict[str, dict] = {}

    for dataset in DATASETS:
        print(f"\n================ PROPHET: {dataset.upper()} ================")

        train = (
            pd.read_csv(
                ready_dir / f"{dataset}_train_scaled.csv",
                parse_dates=["ds"],
            )
            .sort_values("ds")
            .reset_index(drop=True)
        )

        test, scenarios = load_scenarios(
            dataset,
            ready_dir,
            anomaly_dir,
        )

        config, tuning, calibration_residuals = select_configuration(
            train[["ds", "y"]]
        )
        tuning.to_csv(
            output_dir / f"{dataset}_prophet_tuning.csv",
            index=False,
        )

        threshold, median, mad, threshold_method = (
            robust_residual_threshold(calibration_residuals)
        )

        selected_configs[dataset] = {
            **config,
            "validation_protocol": (
                "last 20 percent of training observations"
            ),
            "test_protocol": (
                "full one-step-ahead walk-forward refitting"
            ),
            "nonnegative_prediction_clipping": True,
            "threshold": threshold,
            "threshold_method": threshold_method,
        }

        threshold_rows.append(
            {
                "dataset": dataset,
                "model": "Prophet",
                "threshold": threshold,
                "calibration_residual_median": median,
                "calibration_residual_mad": mad,
                "threshold_method": threshold_method,
                "n_calibration_residuals": len(calibration_residuals),
            }
        )

        print("Selected configuration:", config)
        print(f"Residual threshold: {threshold:.6g}")

        for scenario_name in SCENARIOS:
            print(f"  Running {scenario_name} ...")
            predictions, mean_refit_time, mean_latency, model_size = (
                walk_forward_scenario(
                    train[["ds", "y"]],
                    scenarios[scenario_name],
                    config,
                    threshold,
                )
            )

            if len(predictions) != len(test):
                raise AssertionError(
                    f"{dataset}/{scenario_name}: expected {len(test)} rows, "
                    f"got {len(predictions)}"
                )

            predictions.to_csv(
                prediction_dir
                / f"{dataset}_prophet_{scenario_name}.csv",
                index=False,
            )

            common = {
                "dataset": dataset,
                "model": "Prophet",
                "scenario": scenario_name,
                "n_eval": len(predictions),
                "mean_refit_time_sec": mean_refit_time,
                "mean_inference_latency_sec": mean_latency,
                "model_size_bytes": model_size,
                "threshold": threshold,
                **config,
            }

            if scenario_name == "forecast_original":
                metrics = forecast_metrics(
                    predictions["y_clean"].to_numpy(dtype=float),
                    predictions["y_pred"].to_numpy(dtype=float),
                )
                forecast_rows.append({**common, **metrics})

                print(
                    f"    RMSE={metrics['RMSE']:.6g}, "
                    f"MAE={metrics['MAE']:.6g}, "
                    f"sMAPE={metrics['sMAPE']:.3f}, "
                    f"R2={metrics['R2']:.6g}"
                )
            else:
                metrics = anomaly_metrics(
                    predictions["anomaly_truth"].to_numpy(dtype=int),
                    predictions["anomaly_pred"].to_numpy(dtype=int),
                )
                anomaly_rows.append({**common, **metrics})

                print(
                    f"    reference={metrics['n_reference_anomalies']}, "
                    f"detected={metrics['n_detected']}, "
                    f"F1={metrics['F1']}"
                )

    forecast = pd.DataFrame(forecast_rows)
    anomaly = pd.DataFrame(anomaly_rows)
    thresholds = pd.DataFrame(threshold_rows)

    forecast.to_csv(
        output_dir / "prophet_forecast_results.csv",
        index=False,
    )
    anomaly.to_csv(
        output_dir / "prophet_anomaly_results.csv",
        index=False,
    )
    thresholds.to_csv(
        output_dir / "prophet_residual_thresholds.csv",
        index=False,
    )
    (
        output_dir / "prophet_selected_configs.json"
    ).write_text(
        json.dumps(
            selected_configs,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    metadata = {
        "prophet_role": "forecasting baseline",
        "candidate_configurations": list(CANDIDATE_CONFIGS),
        "selection": (
            "lowest RMSE on chronological validation within training only"
        ),
        "test_protocol": (
            "one-step-ahead walk-forward with refitting after each observed "
            "test value"
        ),
        "seasonality": {
            "yearly": True,
            "weekly": False,
            "daily": False,
        },
        "nonnegative_clipping": True,
        "warning": (
            "This implementation replaces the unreproducible/mislabeled "
            "Prophet pipeline from the original experimental code."
        ),
    }
    (
        output_dir / "prophet_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== Corrected Prophet baseline completed =====")
    print("\nForecast results:")
    print(forecast.to_string(index=False))
    print("\nAnomaly results:")
    print(anomaly.to_string(index=False))


if __name__ == "__main__":
    main()
