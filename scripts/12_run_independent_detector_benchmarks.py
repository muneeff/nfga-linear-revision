from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


DATASETS = ("cholera", "ilinet", "electricity")
SCENARIOS = ("injected_5pct", "injected_10pct", "proxy_robust_z")
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)

VALIDATION_FRACTION = 0.20
TARGET_VALIDATION_FPR = 0.05
ROBUST_Z_THRESHOLD = 3.5
EPS = 1e-12


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("_").lower()


def model_size_bytes(model: object) -> int:
    return int(len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)))


def build_detection_windows(
    values: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Build windows ending at the current observed point.

    This differs from forecasting windows: anomaly detection occurs after the
    current observation is received, so the current value is included.
    """
    values = np.asarray(values, dtype=float)

    if len(values) < window_size:
        raise ValueError(
            f"Series length {len(values)} is shorter than window {window_size}"
        )

    return np.stack(
        [
            values[index - window_size + 1 : index + 1]
            for index in range(window_size - 1, len(values))
        ]
    )


def chronological_split(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(X)
    validation_size = max(20, int(round(VALIDATION_FRACTION * n)))
    validation_size = min(validation_size, n - 20)

    if validation_size <= 0:
        raise ValueError(
            f"Insufficient windows for chronological calibration: n={n}"
        )

    split = n - validation_size
    return X[:split], X[split:]


def scenario_frame(
    dataset: str,
    scenario: str,
    ready_dir: Path,
    anomaly_dir: Path,
) -> pd.DataFrame:
    test = (
        pd.read_csv(
            ready_dir / f"{dataset}_test_scaled.csv",
            parse_dates=["ds"],
        )
        .sort_values("ds")
        .reset_index(drop=True)
    )

    if scenario.startswith("injected_"):
        rate = scenario.split("_")[1]
        frame = (
            pd.read_csv(
                anomaly_dir / f"{dataset}_test_injected_{rate}.csv",
                parse_dates=["ds"],
            )
            .sort_values("ds")
            .reset_index(drop=True)
        )
        return pd.DataFrame(
            {
                "ds": frame["ds"],
                "y_observed": frame["y_anom"].to_numpy(dtype=float),
                "y_observed_scaled": frame["y_anom_scaled"].to_numpy(dtype=float),
                "anomaly_truth": frame["anomaly_injected"].to_numpy(dtype=int),
            }
        )

    if scenario == "proxy_robust_z":
        frame = (
            pd.read_csv(
                anomaly_dir / f"{dataset}_test_proxy_robust_z.csv",
                parse_dates=["ds"],
            )
            .sort_values("ds")
            .reset_index(drop=True)
        )
        return pd.DataFrame(
            {
                "ds": frame["ds"],
                "y_observed": frame["y"].to_numpy(dtype=float),
                "y_observed_scaled": frame["y_scaled"].to_numpy(dtype=float),
                "anomaly_truth": frame[
                    "anomaly_proxy_robust"
                ].to_numpy(dtype=int),
            }
        )

    raise ValueError(f"Unknown scenario: {scenario}")


def sequential_test_windows(
    train_scaled: np.ndarray,
    observed_test_scaled: np.ndarray,
    window_size: int,
) -> np.ndarray:
    history = list(np.asarray(train_scaled, dtype=float))
    windows: list[np.ndarray] = []

    for current_value in np.asarray(observed_test_scaled, dtype=float):
        history.append(float(current_value))
        windows.append(
            np.asarray(history[-window_size:], dtype=float)
        )

    return np.stack(windows)


def confusion_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int | str]:
    truth = np.asarray(truth, dtype=int)
    predicted = np.asarray(predicted, dtype=int)

    if truth.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: truth={truth.shape}, predicted={predicted.shape}"
        )

    tp = int(np.sum((truth == 1) & (predicted == 1)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))

    positives = tp + fn
    negatives = tn + fp

    specificity = float(tn / negatives) if negatives else np.nan
    false_positive_rate = float(fp / negatives) if negatives else np.nan

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
            "FalsePositiveRate": false_positive_rate,
            "BalancedAccuracy": np.nan,
            "metric_status": "not_applicable_no_positive_reference",
        }

    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        predicted,
        average="binary",
        zero_division=0,
    )
    balanced_accuracy = (
        float((recall + specificity) / 2.0)
        if np.isfinite(specificity)
        else np.nan
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
        "FalsePositiveRate": false_positive_rate,
        "BalancedAccuracy": balanced_accuracy,
        "metric_status": "applicable",
    }


def anomaly_score(model: object, X: np.ndarray) -> np.ndarray:
    """
    Larger scores always mean more anomalous.
    """
    if not hasattr(model, "decision_function"):
        raise TypeError(
            f"{type(model).__name__} does not expose decision_function"
        )

    return -np.asarray(model.decision_function(X), dtype=float).reshape(-1)


def fit_score_detector(
    detector_name: str,
    X_fit: np.ndarray,
    X_validation: np.ndarray,
    seed: int,
) -> tuple[object, StandardScaler, float, float, int]:
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)
    X_validation_scaled = scaler.transform(X_validation)

    if detector_name == "IsolationForest":
        model = IsolationForest(
            n_estimators=300,
            max_samples="auto",
            contamination="auto",
            random_state=seed,
            n_jobs=-1,
        )
    elif detector_name == "OneClassSVM":
        model = OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=0.05,
        )
    elif detector_name == "LocalOutlierFactor":
        n_neighbors = min(20, max(5, len(X_fit_scaled) - 1))
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
            contamination="auto",
        )
    else:
        raise ValueError(f"Unknown detector: {detector_name}")

    started = time.perf_counter()
    model.fit(X_fit_scaled)
    training_time = time.perf_counter() - started

    validation_scores = anomaly_score(model, X_validation_scaled)
    threshold = float(
        np.quantile(
            validation_scores,
            1.0 - TARGET_VALIDATION_FPR,
            method="higher",
        )
    )

    composite_size = model_size_bytes(
        {"scaler": scaler, "detector": model}
    )

    return model, scaler, threshold, training_time, composite_size


def detector_specs() -> tuple[dict, ...]:
    return (
        {
            "detector": "IsolationForest",
            "stochastic": True,
            "seeds": SEEDS,
        },
        {
            "detector": "OneClassSVM",
            "stochastic": False,
            "seeds": (0,),
        },
        {
            "detector": "LocalOutlierFactor",
            "stochastic": False,
            "seeds": (0,),
        },
    )


def run_independent_detectors(
    ready_dir: Path,
    anomaly_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for dataset in DATASETS:
        print(f"\n================ DETECTORS: {dataset.upper()} ================")

        npz = np.load(
            ready_dir / f"{dataset}_windows.npz",
            allow_pickle=True,
        )
        window_size = int(npz["window_size"])

        train = (
            pd.read_csv(
                ready_dir / f"{dataset}_train_scaled.csv",
                parse_dates=["ds"],
            )
            .sort_values("ds")
            .reset_index(drop=True)
        )
        train_scaled = train["y_scaled"].to_numpy(dtype=float)
        train_raw = train["y"].to_numpy(dtype=float)

        X_all = build_detection_windows(train_scaled, window_size)
        X_fit, X_validation = chronological_split(X_all)

        for spec in detector_specs():
            detector_name = spec["detector"]

            for seed in spec["seeds"]:
                print(f"\n{detector_name} | seed={seed}")

                (
                    model,
                    scaler,
                    threshold,
                    training_time,
                    size_bytes,
                ) = fit_score_detector(
                    detector_name,
                    X_fit,
                    X_validation,
                    seed,
                )

                for scenario in SCENARIOS:
                    frame = scenario_frame(
                        dataset,
                        scenario,
                        ready_dir,
                        anomaly_dir,
                    )
                    X_test = sequential_test_windows(
                        train_scaled,
                        frame["y_observed_scaled"].to_numpy(dtype=float),
                        window_size,
                    )
                    X_test_scaled = scaler.transform(X_test)

                    started = time.perf_counter()
                    scores = anomaly_score(model, X_test_scaled)
                    latency = (
                        time.perf_counter() - started
                    ) / len(X_test_scaled)

                    predicted = (scores > threshold).astype(int)
                    metrics = confusion_metrics(
                        frame["anomaly_truth"].to_numpy(dtype=int),
                        predicted,
                    )

                    point_output = frame.copy()
                    point_output["anomaly_score"] = scores
                    point_output["threshold"] = threshold
                    point_output["anomaly_pred"] = predicted
                    point_output.to_csv(
                        prediction_dir
                        / (
                            f"{dataset}_{slugify(detector_name)}_"
                            f"seed_{seed}_{scenario}.csv"
                        ),
                        index=False,
                    )

                    rows.append(
                        {
                            "dataset": dataset,
                            "detector": detector_name,
                            "detector_category": "independent_window_detector",
                            "stochastic": bool(spec["stochastic"]),
                            "seed": seed,
                            "scenario": scenario,
                            "window_size": window_size,
                            "n_fit_windows": len(X_fit),
                            "n_validation_windows": len(X_validation),
                            "validation_target_fpr": TARGET_VALIDATION_FPR,
                            "score_threshold": threshold,
                            "training_time_sec": training_time,
                            "inference_latency_sec": latency,
                            "model_size_bytes": size_bytes,
                            **metrics,
                        }
                    )

        # Robust-Z is a point detector and is intentionally separate from
        # the window-based detectors.
        median = float(np.median(train_raw))
        mad = float(np.median(np.abs(train_raw - median)))
        robust_scale = 1.4826 * mad

        if not np.isfinite(robust_scale) or robust_scale <= EPS:
            robust_scale = float(np.std(train_raw, ddof=0))
        if not np.isfinite(robust_scale) or robust_scale <= EPS:
            robust_scale = 1.0

        for scenario in SCENARIOS:
            frame = scenario_frame(
                dataset,
                scenario,
                ready_dir,
                anomaly_dir,
            )

            started = time.perf_counter()
            scores = np.abs(
                (
                    frame["y_observed"].to_numpy(dtype=float)
                    - median
                )
                / robust_scale
            )
            latency = (time.perf_counter() - started) / len(frame)
            predicted = (scores > ROBUST_Z_THRESHOLD).astype(int)

            metrics = confusion_metrics(
                frame["anomaly_truth"].to_numpy(dtype=int),
                predicted,
            )

            point_output = frame.copy()
            point_output["anomaly_score"] = scores
            point_output["threshold"] = ROBUST_Z_THRESHOLD
            point_output["anomaly_pred"] = predicted
            point_output.to_csv(
                prediction_dir
                / f"{dataset}_robust-z_seed_0_{scenario}.csv",
                index=False,
            )

            rows.append(
                {
                    "dataset": dataset,
                    "detector": "Robust-Z",
                    "detector_category": "independent_point_detector",
                    "stochastic": False,
                    "seed": 0,
                    "scenario": scenario,
                    "window_size": 1,
                    "n_fit_windows": len(train_raw),
                    "n_validation_windows": 0,
                    "validation_target_fpr": np.nan,
                    "score_threshold": ROBUST_Z_THRESHOLD,
                    "training_time_sec": 0.0,
                    "inference_latency_sec": latency,
                    "model_size_bytes": 16,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def summarize_independent(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(
            [
                "dataset",
                "detector",
                "detector_category",
                "scenario",
            ],
            as_index=False,
        )
        .agg(
            n_runs=("seed", "count"),
            n_reference_anomalies=("n_reference_anomalies", "first"),
            n_detected_mean=("n_detected", "mean"),
            n_detected_std=("n_detected", "std"),
            Precision_mean=("Precision", "mean"),
            Precision_std=("Precision", "std"),
            Recall_mean=("Recall", "mean"),
            Recall_std=("Recall", "std"),
            F1_mean=("F1", "mean"),
            F1_std=("F1", "std"),
            Specificity_mean=("Specificity", "mean"),
            FalsePositiveRate_mean=("FalsePositiveRate", "mean"),
            BalancedAccuracy_mean=("BalancedAccuracy", "mean"),
            training_time_mean_sec=("training_time_sec", "mean"),
            inference_latency_mean_sec=("inference_latency_sec", "mean"),
            model_size_mean_bytes=("model_size_bytes", "mean"),
            metric_status=("metric_status", "first"),
        )
    )


def load_nfga_linear_baseline(stage6_raw: Path) -> pd.DataFrame:
    frame = pd.read_csv(stage6_raw)
    frame = frame[
        (frame["model"] == "NFGA-LINEAR")
        & frame["scenario"].isin(SCENARIOS)
    ].copy()

    return (
        frame.groupby(
            ["dataset", "scenario"],
            as_index=False,
        )
        .agg(
            n_runs=("seed", "count"),
            n_reference_anomalies=("n_reference_anomalies", "first"),
            n_detected_mean=("n_detected", "mean"),
            n_detected_std=("n_detected", "std"),
            Precision_mean=("Precision", "mean"),
            Precision_std=("Precision", "std"),
            Recall_mean=("Recall", "mean"),
            Recall_std=("Recall", "std"),
            F1_mean=("F1", "mean"),
            F1_std=("F1", "std"),
            training_time_mean_sec=("training_time_sec", "mean"),
            inference_latency_mean_sec=("mean_inference_latency_sec", "mean"),
            model_size_mean_bytes=("model_size_bytes", "mean"),
            metric_status=("metric_status", "first"),
        )
        .assign(
            detector="NFGA-LINEAR residual detector",
            detector_category="forecast_residual_detector",
            Specificity_mean=np.nan,
            FalsePositiveRate_mean=np.nan,
            BalancedAccuracy_mean=np.nan,
        )
    )


def build_comparison(
    independent_summary: pd.DataFrame,
    nfga_summary: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "dataset",
        "detector",
        "detector_category",
        "scenario",
        "n_runs",
        "n_reference_anomalies",
        "n_detected_mean",
        "n_detected_std",
        "Precision_mean",
        "Precision_std",
        "Recall_mean",
        "Recall_std",
        "F1_mean",
        "F1_std",
        "Specificity_mean",
        "FalsePositiveRate_mean",
        "BalancedAccuracy_mean",
        "training_time_mean_sec",
        "inference_latency_mean_sec",
        "model_size_mean_bytes",
        "metric_status",
    ]

    combined = pd.concat(
        [
            independent_summary[columns],
            nfga_summary[columns],
        ],
        ignore_index=True,
    )

    applicable = combined["metric_status"] == "applicable"
    combined["F1_rank_within_dataset_scenario"] = np.nan
    combined.loc[applicable, "F1_rank_within_dataset_scenario"] = (
        combined.loc[applicable]
        .groupby(["dataset", "scenario"])["F1_mean"]
        .rank(method="min", ascending=False)
    )

    return combined.sort_values(
        [
            "dataset",
            "scenario",
            "F1_rank_within_dataset_scenario",
            "detector",
        ],
        na_position="last",
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark independent anomaly detectors separately from "
            "forecasting models."
        )
    )
    parser.add_argument("--ready-dir", required=True)
    parser.add_argument("--anomaly-dir", required=True)
    parser.add_argument("--stage6-raw", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    ready_dir = Path(args.ready_dir)
    anomaly_dir = Path(args.anomaly_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = run_independent_detectors(
        ready_dir,
        anomaly_dir,
        output_dir,
    )
    independent_summary = summarize_independent(raw)
    nfga_summary = load_nfga_linear_baseline(Path(args.stage6_raw))
    comparison = build_comparison(
        independent_summary,
        nfga_summary,
    )

    raw.to_csv(
        output_dir / "independent_detector_raw_results.csv",
        index=False,
    )
    independent_summary.to_csv(
        output_dir / "independent_detector_summary.csv",
        index=False,
    )
    comparison.to_csv(
        output_dir / "detector_comparison_with_nfga_linear.csv",
        index=False,
    )

    metadata = {
        "separation_principle": (
            "Isolation Forest, One-Class SVM, Local Outlier Factor, and "
            "Robust-Z are evaluated only as anomaly detectors, never as "
            "forecasting models."
        ),
        "window_detectors": {
            "training": "first chronological 80% of clean training windows",
            "calibration": "last chronological 20% of clean training windows",
            "threshold": (
                "95th percentile of calibration anomaly scores, corresponding "
                "to a prespecified 5% calibration false-positive target"
            ),
            "test_input": (
                "window ending at the current observed value; prior injected "
                "values remain in subsequent windows"
            ),
        },
        "robust_z": {
            "threshold": ROBUST_Z_THRESHOLD,
            "statistics": "median and MAD from clean training observations",
        },
        "important_limitations": [
            "Injected scenarios are synthetic and contain few positive points.",
            "Proxy robust-z labels are exploratory rather than verified ground truth.",
            "Detectors use different inductive biases; ranking is scenario-dependent.",
        ],
    }
    (
        output_dir / "independent_detector_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== Independent detector benchmark completed =====")
    print("\nComparison:")
    print(comparison.to_string(index=False))
    print("\nOutputs:", output_dir)


if __name__ == "__main__":
    main()
