from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


DATASETS = ("cholera", "ilinet", "electricity")
SCENARIOS = ("injected_5pct", "injected_10pct", "proxy_robust_z")
K_VALUES = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
BASELINE_K = 3.5
SCALED_MAD = 1.4826
EPS = 1e-12

MODEL_SPECS = (
    {
        "model": "Persistence",
        "stage": "stage3",
        "slug": "persistence",
        "stochastic": False,
    },
    {
        "model": "SeasonalNaive",
        "stage": "stage3",
        "slug": "seasonalnaive",
        "stochastic": False,
    },
    {
        "model": "ARIMA",
        "stage": "stage3",
        "slug": "arima",
        "stochastic": False,
    },
    {
        "model": "XGBoost",
        "stage": "stage4",
        "slug": "xgboost",
        "stochastic": True,
    },
    {
        "model": "LSTM",
        "stage": "stage5",
        "slug": "lstm",
        "stochastic": True,
    },
    {
        "model": "NFGA-Core",
        "stage": "stage6",
        "slug": "nfga-core",
        "stochastic": True,
    },
    {
        "model": "NFGA-LINEAR",
        "stage": "stage6",
        "slug": "nfga-linear",
        "stochastic": True,
    },
)


def load_threshold_statistics(
    stage3_dir: Path,
    stage4_dir: Path,
    stage5_dir: Path,
    stage6_dir: Path,
) -> pd.DataFrame:
    sources = [
        (
            stage3_dir / "statistical_residual_thresholds.csv",
            "training_residual_median",
            "training_residual_mad",
        ),
        (
            stage4_dir / "xgboost_residual_thresholds.csv",
            "calibration_residual_median",
            "calibration_residual_mad",
        ),
        (
            stage5_dir / "lstm_residual_thresholds.csv",
            "calibration_residual_median",
            "calibration_residual_mad",
        ),
        (
            stage6_dir / "nfga_residual_thresholds.csv",
            "calibration_residual_median",
            "calibration_residual_mad",
        ),
    ]

    frames: list[pd.DataFrame] = []

    for path, median_col, mad_col in sources:
        if not path.exists():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)

        required = {"dataset", "model", median_col, mad_col}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")

        normalized = frame[
            ["dataset", "model", median_col, mad_col]
        ].rename(
            columns={
                median_col: "residual_median",
                mad_col: "residual_mad",
            }
        )
        frames.append(normalized)

    combined = pd.concat(frames, ignore_index=True)

    duplicates = combined.duplicated(["dataset", "model"], keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate threshold statistics:\n"
            + combined.loc[duplicates].to_string(index=False)
        )

    if (combined["residual_mad"] < 0).any():
        raise ValueError("Residual MAD must not be negative.")

    return combined


def prediction_files(
    predictions_dir: Path,
    *,
    dataset: str,
    slug: str,
    scenario: str,
    stochastic: bool,
) -> list[Path]:
    if stochastic:
        pattern = f"{dataset}_{slug}_seed_*_{scenario}.csv"
    else:
        pattern = f"{dataset}_{slug}_{scenario}.csv"

    files = sorted(predictions_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No prediction files matched {predictions_dir / pattern}"
        )
    return files


def parse_seed(path: Path, stochastic: bool) -> int:
    if not stochastic:
        return 0

    match = re.search(r"_seed_(\d+)_", path.name)
    if match is None:
        raise ValueError(f"Could not parse seed from {path.name}")

    return int(match.group(1))


def confusion_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int | str]:
    truth = np.asarray(truth, dtype=int)
    predicted = np.asarray(predicted, dtype=int)

    if truth.shape != predicted.shape:
        raise ValueError(
            f"Truth/prediction shape mismatch: {truth.shape} vs {predicted.shape}"
        )

    tp = int(np.sum((truth == 1) & (predicted == 1)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))

    positives = tp + fn
    negatives = tn + fp

    false_positive_rate = (
        float(fp / negatives) if negatives > 0 else np.nan
    )
    specificity = (
        float(tn / negatives) if negatives > 0 else np.nan
    )

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


def build_run_level_results(
    stage_dirs: dict[str, Path],
    threshold_stats: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    threshold_lookup = threshold_stats.set_index(
        ["dataset", "model"]
    )

    for dataset in DATASETS:
        for spec in MODEL_SPECS:
            model = spec["model"]
            predictions_dir = (
                stage_dirs[spec["stage"]] / "predictions"
            )

            if (dataset, model) not in threshold_lookup.index:
                raise KeyError(
                    f"Missing threshold statistics for {dataset}/{model}"
                )

            threshold_row = threshold_lookup.loc[(dataset, model)]
            residual_median = float(threshold_row["residual_median"])
            residual_mad = float(threshold_row["residual_mad"])

            for scenario in SCENARIOS:
                files = prediction_files(
                    predictions_dir,
                    dataset=dataset,
                    slug=spec["slug"],
                    scenario=scenario,
                    stochastic=spec["stochastic"],
                )

                for file_path in files:
                    frame = pd.read_csv(file_path)

                    required = {
                        "anomaly_truth",
                        "residual_abs",
                    }
                    missing = required.difference(frame.columns)
                    if missing:
                        raise ValueError(
                            f"{file_path}: missing columns {sorted(missing)}"
                        )

                    truth = frame["anomaly_truth"].to_numpy(dtype=int)
                    residuals = frame["residual_abs"].to_numpy(dtype=float)

                    if not np.isfinite(residuals).all():
                        raise ValueError(
                            f"{file_path}: non-finite residuals"
                        )

                    seed = parse_seed(
                        file_path,
                        bool(spec["stochastic"]),
                    )

                    for k in K_VALUES:
                        threshold = (
                            residual_median
                            + k * SCALED_MAD * residual_mad
                        )
                        predicted = (residuals > threshold).astype(int)
                        metrics = confusion_metrics(truth, predicted)

                        rows.append(
                            {
                                "dataset": dataset,
                                "model": model,
                                "stochastic": bool(spec["stochastic"]),
                                "seed": seed,
                                "scenario": scenario,
                                "k_mad": k,
                                "threshold": threshold,
                                "residual_median": residual_median,
                                "residual_mad": residual_mad,
                                "n_eval": len(frame),
                                "prediction_file": str(file_path),
                                **metrics,
                            }
                        )

    return pd.DataFrame(rows)


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby(
            ["dataset", "model", "scenario", "k_mad"],
            as_index=False,
        )
        .agg(
            n_runs=("seed", "count"),
            n_reference_anomalies=("n_reference_anomalies", "first"),
            threshold=("threshold", "first"),
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
            metric_status=("metric_status", "first"),
        )
    )

    baseline = summary[
        summary["k_mad"] == BASELINE_K
    ][
        [
            "dataset",
            "model",
            "scenario",
            "F1_mean",
            "FalsePositiveRate_mean",
        ]
    ].rename(
        columns={
            "F1_mean": "baseline_F1_mean",
            "FalsePositiveRate_mean": "baseline_FalsePositiveRate_mean",
        }
    )

    summary = summary.merge(
        baseline,
        on=["dataset", "model", "scenario"],
        how="left",
        validate="many_to_one",
    )
    summary["F1_change_vs_baseline"] = (
        summary["F1_mean"] - summary["baseline_F1_mean"]
    )
    summary["FalsePositiveRate_change_vs_baseline"] = (
        summary["FalsePositiveRate_mean"]
        - summary["baseline_FalsePositiveRate_mean"]
    )

    return summary


def build_macro_injected_summary(raw: pd.DataFrame) -> pd.DataFrame:
    injected = raw[
        raw["scenario"].isin(("injected_5pct", "injected_10pct"))
    ].copy()

    per_run = (
        injected.groupby(
            ["dataset", "model", "seed", "k_mad"],
            as_index=False,
        )
        .agg(
            macro_F1=("F1", "mean"),
            macro_Precision=("Precision", "mean"),
            macro_Recall=("Recall", "mean"),
            macro_FalsePositiveRate=("FalsePositiveRate", "mean"),
        )
    )

    macro = (
        per_run.groupby(
            ["dataset", "model", "k_mad"],
            as_index=False,
        )
        .agg(
            n_runs=("seed", "count"),
            macro_F1_mean=("macro_F1", "mean"),
            macro_F1_std=("macro_F1", "std"),
            macro_Precision_mean=("macro_Precision", "mean"),
            macro_Recall_mean=("macro_Recall", "mean"),
            macro_FalsePositiveRate_mean=("macro_FalsePositiveRate", "mean"),
        )
    )

    baseline = macro[
        macro["k_mad"] == BASELINE_K
    ][
        [
            "dataset",
            "model",
            "macro_F1_mean",
            "macro_FalsePositiveRate_mean",
        ]
    ].rename(
        columns={
            "macro_F1_mean": "baseline_macro_F1_mean",
            "macro_FalsePositiveRate_mean": (
                "baseline_macro_FalsePositiveRate_mean"
            ),
        }
    )

    macro = macro.merge(
        baseline,
        on=["dataset", "model"],
        how="left",
        validate="many_to_one",
    )
    macro["macro_F1_change_vs_baseline"] = (
        macro["macro_F1_mean"]
        - macro["baseline_macro_F1_mean"]
    )
    macro["macro_FalsePositiveRate_change_vs_baseline"] = (
        macro["macro_FalsePositiveRate_mean"]
        - macro["baseline_macro_FalsePositiveRate_mean"]
    )

    return macro


def build_descriptive_best_k(macro: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for (dataset, model), group in macro.groupby(
        ["dataset", "model"],
        sort=True,
    ):
        ordered = group.sort_values(
            [
                "macro_F1_mean",
                "macro_FalsePositiveRate_mean",
                "k_mad",
            ],
            ascending=[False, True, True],
        )
        best = ordered.iloc[0]
        baseline = group[group["k_mad"] == BASELINE_K].iloc[0]

        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "baseline_k": BASELINE_K,
                "baseline_macro_F1": baseline["macro_F1_mean"],
                "best_k_descriptive_only": best["k_mad"],
                "best_macro_F1_descriptive_only": best["macro_F1_mean"],
                "absolute_gain_over_baseline": (
                    best["macro_F1_mean"]
                    - baseline["macro_F1_mean"]
                ),
                "baseline_macro_FalsePositiveRate": (
                    baseline["macro_FalsePositiveRate_mean"]
                ),
                "best_macro_FalsePositiveRate": (
                    best["macro_FalsePositiveRate_mean"]
                ),
                "interpretation": (
                    "descriptive test-scenario sensitivity only; "
                    "not a tuning recommendation"
                ),
            }
        )

    return pd.DataFrame(rows)


def write_markdown_summary(
    path: Path,
    best_k: pd.DataFrame,
) -> None:
    lines = [
        "# Anomaly-threshold sensitivity",
        "",
        f"Prespecified baseline: k = {BASELINE_K}.",
        "",
        "The best-k entries below are descriptive only because the injected "
        "test scenarios were used to compute them. They must not replace the "
        "prespecified baseline in the main comparison.",
        "",
    ]

    for dataset in DATASETS:
        lines.append(f"## {dataset}")
        subset = best_k[best_k["dataset"] == dataset]
        for _, row in subset.iterrows():
            lines.append(
                f"- {row['model']}: baseline macro-F1="
                f"{row['baseline_macro_F1']:.4f}; descriptive best "
                f"k={row['best_k_descriptive_only']:.1f}, macro-F1="
                f"{row['best_macro_F1_descriptive_only']:.4f}, gain="
                f"{row['absolute_gain_over_baseline']:+.4f}."
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation rule",
            "",
            "- Main-paper model comparisons retain k = 3.5.",
            "- Sensitivity results quantify threshold dependence but do not "
            "constitute post-hoc tuning.",
            "- Proxy robust-z labels remain exploratory rather than verified "
            "ground truth.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MAD-threshold sensitivity audit for anomaly detection."
    )
    parser.add_argument("--stage3-dir", required=True)
    parser.add_argument("--stage4-dir", required=True)
    parser.add_argument("--stage5-dir", required=True)
    parser.add_argument("--stage6-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    stage_dirs = {
        "stage3": Path(args.stage3_dir),
        "stage4": Path(args.stage4_dir),
        "stage5": Path(args.stage5_dir),
        "stage6": Path(args.stage6_dir),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_stats = load_threshold_statistics(
        stage_dirs["stage3"],
        stage_dirs["stage4"],
        stage_dirs["stage5"],
        stage_dirs["stage6"],
    )
    raw = build_run_level_results(stage_dirs, threshold_stats)
    summary = build_summary(raw)
    macro = build_macro_injected_summary(raw)
    best_k = build_descriptive_best_k(macro)

    raw.to_csv(
        output_dir / "anomaly_threshold_sensitivity_raw.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "anomaly_threshold_sensitivity_summary.csv",
        index=False,
    )
    macro.to_csv(
        output_dir / "anomaly_threshold_macro_injected.csv",
        index=False,
    )
    best_k.to_csv(
        output_dir / "anomaly_threshold_descriptive_best_k.csv",
        index=False,
    )

    write_markdown_summary(
        output_dir / "anomaly_threshold_sensitivity_summary.md",
        best_k,
    )

    metadata = {
        "k_values": list(K_VALUES),
        "baseline_k": BASELINE_K,
        "threshold_formula": (
            "median(calibration absolute residuals) "
            "+ k * 1.4826 * MAD(calibration absolute residuals)"
        ),
        "main_analysis_rule": (
            "Retain the prespecified baseline k=3.5. "
            "Do not select k using injected test scenarios."
        ),
        "proxy_label_warning": (
            "Robust-z proxy labels are exploratory and are not verified "
            "ground-truth anomalies."
        ),
    }
    (
        output_dir / "anomaly_threshold_sensitivity_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== Anomaly-threshold sensitivity completed =====")
    print("\nDescriptive best k values:")
    print(best_k.to_string(index=False))
    print("\nMacro injected summary:")
    print(
        macro[
            [
                "dataset",
                "model",
                "k_mad",
                "macro_F1_mean",
                "macro_F1_std",
                "macro_FalsePositiveRate_mean",
                "macro_F1_change_vs_baseline",
            ]
        ].to_string(index=False)
    )
    print("\nOutputs:", output_dir)


if __name__ == "__main__":
    main()
