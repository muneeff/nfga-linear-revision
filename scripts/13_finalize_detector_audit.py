from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


INJECTED_SCENARIOS = ("injected_5pct", "injected_10pct")
PROXY_SCENARIO = "proxy_robust_z"
EPS = 1e-12


def confusion_metrics(
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


def load_nfga_point_results(
    predictions_dir: Path,
) -> pd.DataFrame:
    rows: list[dict] = []

    pattern = "*_nfga-linear_seed_*_*.csv"
    files = sorted(predictions_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No NFGA-LINEAR prediction files found in {predictions_dir}"
        )

    for path in files:
        name = path.stem

        dataset = name.split("_", 1)[0]
        seed_token = name.split("_seed_", 1)[1]
        seed = int(seed_token.split("_", 1)[0])
        scenario = seed_token.split("_", 1)[1]

        if scenario not in (*INJECTED_SCENARIOS, PROXY_SCENARIO):
            continue

        frame = pd.read_csv(path)
        required = {"anomaly_truth", "anomaly_pred"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")

        metrics = confusion_metrics(
            frame["anomaly_truth"].to_numpy(dtype=int),
            frame["anomaly_pred"].to_numpy(dtype=int),
        )

        rows.append(
            {
                "dataset": dataset,
                "detector": "NFGA-LINEAR residual detector",
                "detector_category": "forecast_residual_detector",
                "seed": seed,
                "scenario": scenario,
                "n_eval": len(frame),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def summarize_nfga(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(
            ["dataset", "detector", "detector_category", "scenario"],
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
            metric_status=("metric_status", "first"),
        )
    )


def combine_detector_results(
    independent_summary_path: Path,
    nfga_summary: pd.DataFrame,
) -> pd.DataFrame:
    independent = pd.read_csv(independent_summary_path)

    common_columns = [
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
        "metric_status",
    ]

    combined = pd.concat(
        [
            independent[common_columns],
            nfga_summary[common_columns],
        ],
        ignore_index=True,
    )

    return combined


def build_injected_table(combined: pd.DataFrame) -> pd.DataFrame:
    injected = combined[
        combined["scenario"].isin(INJECTED_SCENARIOS)
    ].copy()

    injected["F1_rank"] = (
        injected.groupby(["dataset", "scenario"])["F1_mean"]
        .rank(method="min", ascending=False)
    )
    injected["FPR_rank"] = (
        injected.groupby(["dataset", "scenario"])["FalsePositiveRate_mean"]
        .rank(method="min", ascending=True)
    )

    injected["best_F1_flag"] = injected["F1_rank"] == 1
    return injected.sort_values(
        ["dataset", "scenario", "F1_rank", "FalsePositiveRate_mean"]
    ).reset_index(drop=True)


def build_macro_table(injected: pd.DataFrame) -> pd.DataFrame:
    macro = (
        injected.groupby(
            ["dataset", "detector", "detector_category"],
            as_index=False,
        )
        .agg(
            scenarios=("scenario", "count"),
            macro_Precision=("Precision_mean", "mean"),
            macro_Recall=("Recall_mean", "mean"),
            macro_F1=("F1_mean", "mean"),
            macro_FalsePositiveRate=("FalsePositiveRate_mean", "mean"),
            macro_BalancedAccuracy=("BalancedAccuracy_mean", "mean"),
        )
    )

    macro["macro_F1_rank"] = (
        macro.groupby("dataset")["macro_F1"]
        .rank(method="min", ascending=False)
    )

    return macro.sort_values(
        ["dataset", "macro_F1_rank", "macro_FalsePositiveRate"]
    ).reset_index(drop=True)


def build_proxy_sanity_table(combined: pd.DataFrame) -> pd.DataFrame:
    proxy = combined[
        combined["scenario"] == PROXY_SCENARIO
    ].copy()

    proxy["interpretation"] = np.where(
        proxy["detector"] == "Robust-Z",
        (
            "Circular sanity check: labels were generated by the same "
            "Robust-Z rule, so F1 is not an independent performance estimate."
        ),
        (
            "Exploratory agreement with proxy labels only; not verified "
            "ground-truth anomaly performance."
        ),
    )

    return proxy.sort_values(["dataset", "detector"]).reset_index(drop=True)


def build_claim_audit(
    injected: pd.DataFrame,
    macro: pd.DataFrame,
) -> dict:
    nfga_rows = injected[
        injected["detector"] == "NFGA-LINEAR residual detector"
    ]
    best_count = int((nfga_rows["F1_rank"] == 1).sum())
    scenario_count = int(len(nfga_rows))

    best_macro = macro[
        macro["macro_F1_rank"] == 1
    ][["dataset", "detector", "macro_F1", "macro_FalsePositiveRate"]]

    return {
        "injected_scenarios_evaluated_for_nfga": scenario_count,
        "nfga_best_f1_scenario_count": best_count,
        "nfga_best_f1_fraction": (
            best_count / scenario_count if scenario_count else np.nan
        ),
        "best_macro_detector_by_dataset": best_macro.to_dict(orient="records"),
        "allowed_claim": (
            "NFGA-LINEAR's residual detector achieved the highest F1 in most "
            "synthetic injected scenarios, while performance remained "
            "scenario-dependent."
        ),
        "forbidden_claims": [
            "NFGA-LINEAR was universally the best anomaly detector.",
            "Robust-Z achieved perfect real-world anomaly detection on ILINet.",
            "Proxy-label results constitute verified ground-truth validation.",
        ],
    }


def write_markdown(
    path: Path,
    injected: pd.DataFrame,
    macro: pd.DataFrame,
    claim_audit: dict,
) -> None:
    lines = [
        "# Final detector audit",
        "",
        "## Injected-scenario ranking",
        "",
    ]

    for dataset in sorted(injected["dataset"].unique()):
        lines.append(f"### {dataset}")
        subset = injected[injected["dataset"] == dataset]
        for scenario in INJECTED_SCENARIOS:
            scenario_rows = subset[subset["scenario"] == scenario]
            lines.append(f"- {scenario}:")
            for _, row in scenario_rows.iterrows():
                lines.append(
                    f"  - {int(row['F1_rank'])}. {row['detector']}: "
                    f"F1={row['F1_mean']:.4f}, "
                    f"precision={row['Precision_mean']:.4f}, "
                    f"recall={row['Recall_mean']:.4f}, "
                    f"FPR={row['FalsePositiveRate_mean']:.4f}"
                )
        lines.append("")

    lines.extend(
        [
            "## Macro injected results",
            "",
        ]
    )

    for _, row in macro.iterrows():
        lines.append(
            f"- {row['dataset']} / {row['detector']}: "
            f"macro-F1={row['macro_F1']:.4f}, "
            f"macro-FPR={row['macro_FalsePositiveRate']:.4f}, "
            f"rank={int(row['macro_F1_rank'])}."
        )

    lines.extend(
        [
            "",
            "## Claim control",
            "",
            f"- NFGA-LINEAR ranked first in "
            f"{claim_audit['nfga_best_f1_scenario_count']}/"
            f"{claim_audit['injected_scenarios_evaluated_for_nfga']} "
            "synthetic injected scenarios.",
            f"- Allowed: {claim_audit['allowed_claim']}",
            "- Proxy robust-z labels are not verified ground truth.",
            "- Robust-Z performance on the robust-z proxy scenario is circular "
            "and must not be reported as independent validation.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize anomaly-detector comparison and repair NFGA FPR."
    )
    parser.add_argument("--stage6-predictions", required=True)
    parser.add_argument("--stage10-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    stage6_predictions = Path(args.stage6_predictions)
    stage10_dir = Path(args.stage10_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nfga_raw = load_nfga_point_results(stage6_predictions)
    nfga_summary = summarize_nfga(nfga_raw)
    combined = combine_detector_results(
        stage10_dir / "independent_detector_summary.csv",
        nfga_summary,
    )

    injected = build_injected_table(combined)
    macro = build_macro_table(injected)
    proxy = build_proxy_sanity_table(combined)
    claims = build_claim_audit(injected, macro)

    nfga_raw.to_csv(
        output_dir / "nfga_linear_detector_confusion_raw.csv",
        index=False,
    )
    nfga_summary.to_csv(
        output_dir / "nfga_linear_detector_summary_with_fpr.csv",
        index=False,
    )
    injected.to_csv(
        output_dir / "detector_injected_scenarios_final.csv",
        index=False,
    )
    macro.to_csv(
        output_dir / "detector_macro_injected_final.csv",
        index=False,
    )
    proxy.to_csv(
        output_dir / "detector_proxy_sanity_checks.csv",
        index=False,
    )
    (
        output_dir / "detector_claim_audit.json"
    ).write_text(
        json.dumps(claims, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_markdown(
        output_dir / "final_detector_audit.md",
        injected,
        macro,
        claims,
    )

    print("\n===== Final detector audit completed =====")
    print("\nInjected-scenario final table:")
    print(
        injected[
            [
                "dataset",
                "scenario",
                "detector",
                "F1_mean",
                "Precision_mean",
                "Recall_mean",
                "FalsePositiveRate_mean",
                "F1_rank",
            ]
        ].to_string(index=False)
    )
    print("\nMacro table:")
    print(macro.to_string(index=False))
    print("\nClaim audit:")
    print(json.dumps(claims, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
