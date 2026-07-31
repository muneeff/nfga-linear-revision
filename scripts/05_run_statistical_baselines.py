from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


DATASETS = ("cholera", "ilinet", "electricity")
ARIMA_ORDERS = (
    (1, 0, 0),
    (1, 1, 0),
    (2, 1, 0),
    (1, 1, 1),
    (2, 1, 1),
    (3, 1, 0),
)
ROBUST_K = 3.5
SEASONAL_LAG = 52
EPS = 1e-12


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

    # Independent R² check to prevent mixing predictions or target arrays.
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
        raise ValueError("At least three finite training residuals are required.")

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


def select_arima_order(
    train_y: np.ndarray,
) -> tuple[tuple[int, int, int], float, pd.DataFrame]:
    rows: list[dict] = []
    best_order: tuple[int, int, int] | None = None
    best_aic = np.inf

    for order in ARIMA_ORDERS:
        started = time.perf_counter()
        try:
            fitted = ARIMA(
                train_y,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()

            aic = float(fitted.aic)
            status = "ok"

            if np.isfinite(aic) and aic < best_aic:
                best_aic = aic
                best_order = order

        except Exception as exc:
            aic = np.nan
            status = f"failed:{type(exc).__name__}"

        rows.append(
            {
                "order": str(order),
                "aic": aic,
                "status": status,
                "fit_time_sec": time.perf_counter() - started,
            }
        )

    if best_order is None:
        best_order = (1, 1, 0)
        best_aic = np.nan

    return best_order, float(best_aic), pd.DataFrame(rows)


def arima_training_residuals(
    train_y: np.ndarray,
    order: tuple[int, int, int],
) -> np.ndarray:
    fitted = ARIMA(
        train_y,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()

    residuals = np.abs(np.asarray(fitted.resid, dtype=float))
    burn = max(5, sum(order))
    residuals = residuals[burn:]
    residuals = residuals[np.isfinite(residuals)]

    if residuals.size < 3:
        raise ValueError(f"Too few ARIMA training residuals for order {order}")

    return residuals


def persistence_training_residuals(train_y: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(np.asarray(train_y, dtype=float)))


def seasonal_training_residuals(
    train_y: np.ndarray,
    lag: int,
) -> np.ndarray:
    values = np.asarray(train_y, dtype=float)
    if len(values) <= lag:
        raise ValueError(
            f"Training series length {len(values)} is not greater than seasonal lag {lag}"
        )
    return np.abs(values[lag:] - values[:-lag])


def one_step_persistence(history: list[float]) -> float:
    return float(history[-1])


def one_step_seasonal(history: list[float], lag: int) -> float:
    if len(history) >= lag:
        return float(history[-lag])
    return float(history[-1])


def one_step_arima(
    history: list[float],
    order: tuple[int, int, int],
) -> float:
    try:
        fitted = ARIMA(
            history,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
        return float(np.asarray(fitted.forecast(steps=1))[0])
    except Exception:
        return float(history[-1])


def load_scenarios(
    dataset: str,
    ready_dir: Path,
    anomaly_dir: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    test_path = ready_dir / f"{dataset}_test_scaled.csv"
    test = (
        pd.read_csv(test_path, parse_dates=["ds"])
        .sort_values("ds")
        .reset_index(drop=True)
    )

    required_test = {"ds", "y"}
    if not required_test.issubset(test.columns):
        raise ValueError(f"{dataset}: test file lacks {required_test - set(test.columns)}")

    scenarios: dict[str, pd.DataFrame] = {}

    original = pd.DataFrame(
        {
            "ds": test["ds"],
            "y_clean": test["y"].to_numpy(dtype=float),
            "y_observed": test["y"].to_numpy(dtype=float),
            "anomaly_truth": np.zeros(len(test), dtype=int),
        }
    )
    scenarios["forecast_original"] = original

    for rate in (5, 10):
        path = anomaly_dir / f"{dataset}_test_injected_{rate}pct.csv"
        frame = (
            pd.read_csv(path, parse_dates=["ds"])
            .sort_values("ds")
            .reset_index(drop=True)
        )

        if not np.array_equal(
            frame["ds"].to_numpy(dtype="datetime64[ns]"),
            test["ds"].to_numpy(dtype="datetime64[ns]"),
        ):
            raise ValueError(f"{dataset} injected_{rate}pct: timestamp mismatch")

        scenarios[f"injected_{rate}pct"] = pd.DataFrame(
            {
                "ds": frame["ds"],
                "y_clean": frame["y_clean"].to_numpy(dtype=float),
                "y_observed": frame["y_anom"].to_numpy(dtype=float),
                "anomaly_truth": frame["anomaly_injected"].to_numpy(dtype=int),
            }
        )

    proxy_path = anomaly_dir / f"{dataset}_test_proxy_robust_z.csv"
    proxy = (
        pd.read_csv(proxy_path, parse_dates=["ds"])
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
            "anomaly_truth": proxy["anomaly_proxy_robust"].to_numpy(dtype=int),
        }
    )

    return test, scenarios


def run_scenario(
    *,
    model_name: str,
    train_y: np.ndarray,
    scenario: pd.DataFrame,
    threshold: float,
    arima_order: tuple[int, int, int] | None = None,
    seasonal_lag: int = SEASONAL_LAG,
) -> tuple[pd.DataFrame, float]:
    history = list(np.asarray(train_y, dtype=float))
    predictions: list[float] = []
    latencies: list[float] = []

    for observed in scenario["y_observed"].to_numpy(dtype=float):
        started = time.perf_counter()

        if model_name == "Persistence":
            prediction = one_step_persistence(history)
        elif model_name == "SeasonalNaive":
            prediction = one_step_seasonal(history, seasonal_lag)
        elif model_name == "ARIMA":
            if arima_order is None:
                raise ValueError("ARIMA order is required.")
            prediction = one_step_arima(history, arima_order)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        latencies.append(time.perf_counter() - started)
        predictions.append(float(prediction))

        # At the next time step, the system has observed the scenario value.
        history.append(float(observed))

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

    return output, float(np.mean(latencies))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full-horizon Persistence, Seasonal Naive, and ARIMA baselines."
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

    forecast_rows: list[dict] = []
    anomaly_rows: list[dict] = []
    threshold_rows: list[dict] = []
    arima_order_rows: list[dict] = []

    for dataset in DATASETS:
        print(f"\n================ {dataset.upper()} ================")

        train_path = ready_dir / f"{dataset}_train_scaled.csv"
        train = (
            pd.read_csv(train_path, parse_dates=["ds"])
            .sort_values("ds")
            .reset_index(drop=True)
        )
        train_y = train["y"].to_numpy(dtype=float)

        test, scenarios = load_scenarios(dataset, ready_dir, anomaly_dir)

        best_order, best_aic, order_grid = select_arima_order(train_y)
        order_grid.insert(0, "dataset", dataset)
        order_grid.to_csv(
            output_dir / f"{dataset}_arima_order_search.csv",
            index=False,
        )

        arima_order_rows.append(
            {
                "dataset": dataset,
                "selected_order": str(best_order),
                "selected_aic": best_aic,
            }
        )
        print(f"Selected ARIMA order: {best_order}, AIC={best_aic:.3f}")

        residual_sources = {
            "Persistence": persistence_training_residuals(train_y),
            "SeasonalNaive": seasonal_training_residuals(
                train_y,
                SEASONAL_LAG,
            ),
            "ARIMA": arima_training_residuals(train_y, best_order),
        }

        for model_name, training_residuals in residual_sources.items():
            threshold, median, mad, threshold_method = robust_residual_threshold(
                training_residuals
            )

            threshold_rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "threshold": threshold,
                    "training_residual_median": median,
                    "training_residual_mad": mad,
                    "threshold_method": threshold_method,
                    "n_training_residuals": len(training_residuals),
                }
            )

            print(
                f"\n{model_name}: threshold={threshold:.6g} "
                f"({threshold_method})"
            )

            for scenario_name, scenario in scenarios.items():
                predictions, mean_latency = run_scenario(
                    model_name=model_name,
                    train_y=train_y,
                    scenario=scenario,
                    threshold=threshold,
                    arima_order=best_order if model_name == "ARIMA" else None,
                    seasonal_lag=SEASONAL_LAG,
                )

                if len(predictions) != len(test):
                    raise AssertionError(
                        f"{dataset}/{model_name}/{scenario_name}: "
                        f"expected {len(test)} rows, got {len(predictions)}"
                    )

                prediction_path = (
                    predictions_dir
                    / f"{dataset}_{slugify(model_name)}_{scenario_name}.csv"
                )
                predictions.to_csv(prediction_path, index=False)

                if scenario_name == "forecast_original":
                    metrics = forecast_metrics(
                        predictions["y_clean"].to_numpy(dtype=float),
                        predictions["y_pred"].to_numpy(dtype=float),
                    )
                    forecast_rows.append(
                        {
                            "dataset": dataset,
                            "model": model_name,
                            "scenario": scenario_name,
                            "n_eval": len(predictions),
                            **metrics,
                            "mean_inference_latency_sec": mean_latency,
                            "selected_order": (
                                str(best_order) if model_name == "ARIMA" else ""
                            ),
                        }
                    )

                    print(
                        f"  forecast_original | "
                        f"RMSE={metrics['RMSE']:.6g}, "
                        f"MAE={metrics['MAE']:.6g}, "
                        f"sMAPE={metrics['sMAPE']:.3f}, "
                        f"R2={metrics['R2']:.6g}"
                    )
                else:
                    precision, recall, f1, status = anomaly_metrics(
                        predictions["anomaly_truth"].to_numpy(dtype=int),
                        predictions["anomaly_pred"].to_numpy(dtype=int),
                    )
                    anomaly_rows.append(
                        {
                            "dataset": dataset,
                            "model": model_name,
                            "scenario": scenario_name,
                            "n_eval": len(predictions),
                            "n_reference_anomalies": int(
                                predictions["anomaly_truth"].sum()
                            ),
                            "n_detected": int(predictions["anomaly_pred"].sum()),
                            "Precision": precision,
                            "Recall": recall,
                            "F1": f1,
                            "metric_status": status,
                            "threshold": threshold,
                            "mean_inference_latency_sec": mean_latency,
                            "selected_order": (
                                str(best_order) if model_name == "ARIMA" else ""
                            ),
                        }
                    )

                    print(
                        f"  {scenario_name} | "
                        f"reference={int(predictions['anomaly_truth'].sum())}, "
                        f"detected={int(predictions['anomaly_pred'].sum())}, "
                        f"F1={f1 if np.isfinite(f1) else 'N/A'}"
                    )

    forecast_df = pd.DataFrame(forecast_rows)
    anomaly_df = pd.DataFrame(anomaly_rows)
    thresholds_df = pd.DataFrame(threshold_rows)
    arima_orders_df = pd.DataFrame(arima_order_rows)

    forecast_df.to_csv(
        output_dir / "statistical_forecast_results.csv",
        index=False,
    )
    anomaly_df.to_csv(
        output_dir / "statistical_anomaly_results.csv",
        index=False,
    )
    thresholds_df.to_csv(
        output_dir / "statistical_residual_thresholds.csv",
        index=False,
    )
    arima_orders_df.to_csv(
        output_dir / "selected_arima_orders.csv",
        index=False,
    )

    metadata = {
        "protocol": "full_test_one_step_ahead_walk_forward",
        "forecast_evaluation": "clean original test series only",
        "anomaly_evaluation": (
            "forecast each contaminated scenario before observing its current value; "
            "append the observed scenario value to history for the next step"
        ),
        "threshold": {
            "formula": "median(training_abs_residuals) + 3.5 * 1.4826 * MAD",
            "fallback": "median + 3 * standard_deviation",
        },
        "seasonal_lag": SEASONAL_LAG,
        "arima_candidates": [str(order) for order in ARIMA_ORDERS],
    }
    (output_dir / "statistical_baselines_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== Statistical baselines completed =====")
    print("Output directory:", output_dir)
    print("\nForecast results:")
    print(forecast_df.to_string(index=False))
    print("\nAnomaly results:")
    print(anomaly_df.to_string(index=False))


if __name__ == "__main__":
    main()
