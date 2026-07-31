from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


STOCHASTIC_MODELS = ("XGBoost", "LSTM", "NFGA-Core", "NFGA-LINEAR")
PRIMARY_MODEL = "NFGA-LINEAR"
FORECAST_METRICS = ("RMSE", "MAE", "sMAPE")
BOOTSTRAP_REPETITIONS = 20000
BOOTSTRAP_SEED = 20260801
EPS = 1e-12


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down family-wise error correction."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = (m - rank) * p[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(1.0, running_max)

    return adjusted.tolist()


def paired_rank_biserial(
    primary: np.ndarray,
    comparator: np.ndarray,
    *,
    lower_is_better: bool,
) -> float:
    """
    Positive values mean the primary model is better.

    For error metrics, benefit = comparator - primary.
    For score metrics, benefit = primary - comparator.
    """
    if lower_is_better:
        benefit = np.asarray(comparator) - np.asarray(primary)
    else:
        benefit = np.asarray(primary) - np.asarray(comparator)

    benefit = benefit[np.abs(benefit) > EPS]
    if benefit.size == 0:
        return 0.0

    absolute = np.abs(benefit)
    ranks = pd.Series(absolute).rank(method="average").to_numpy(dtype=float)
    positive = float(ranks[benefit > 0].sum())
    negative = float(ranks[benefit < 0].sum())
    denominator = positive + negative

    if denominator <= EPS:
        return 0.0

    return (positive - negative) / denominator


def paired_bootstrap_ci(
    primary: np.ndarray,
    comparator: np.ndarray,
    *,
    lower_is_better: bool,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """
    Returns mean paired benefit and percentile 95% CI.
    Positive benefit means the primary model is better.
    """
    primary = np.asarray(primary, dtype=float)
    comparator = np.asarray(comparator, dtype=float)

    if lower_is_better:
        benefit = comparator - primary
    else:
        benefit = primary - comparator

    rng = np.random.default_rng(seed)
    n = len(benefit)
    indices = rng.integers(0, n, size=(repetitions, n))
    means = benefit[indices].mean(axis=1)

    return (
        float(np.mean(benefit)),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def exact_or_auto_wilcoxon(
    primary: np.ndarray,
    comparator: np.ndarray,
) -> tuple[float, float, int, str]:
    primary = np.asarray(primary, dtype=float)
    comparator = np.asarray(comparator, dtype=float)
    differences = primary - comparator
    nonzero = np.abs(differences) > EPS
    n_nonzero = int(nonzero.sum())

    if n_nonzero == 0:
        return 0.0, 1.0, 0, "all_differences_zero"

    method = "exact" if n_nonzero <= 25 and n_nonzero == len(differences) else "auto"
    result = wilcoxon(
        primary,
        comparator,
        alternative="two-sided",
        zero_method="wilcox",
        method=method,
    )
    return float(result.statistic), float(result.pvalue), n_nonzero, method


def load_stochastic_results(
    xgboost_path: Path,
    lstm_path: Path,
    nfga_path: Path,
) -> pd.DataFrame:
    xgb = pd.read_csv(xgboost_path)
    lstm = pd.read_csv(lstm_path)
    nfga = pd.read_csv(nfga_path)

    combined = pd.concat([xgb, lstm, nfga], ignore_index=True, sort=False)
    combined = combined[
        combined["model"].isin(STOCHASTIC_MODELS)
    ].copy()

    required = {"dataset", "model", "seed", "scenario"}
    missing = required.difference(combined.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    combined["seed"] = combined["seed"].astype(int)
    return combined


def validate_seed_alignment(frame: pd.DataFrame) -> None:
    for dataset in sorted(frame["dataset"].unique()):
        subset = frame[
            (frame["dataset"] == dataset)
            & (frame["scenario"] == "forecast_original")
        ]

        reference_seeds: set[int] | None = None
        for model in STOCHASTIC_MODELS:
            seeds = set(
                subset.loc[subset["model"] == model, "seed"].astype(int)
            )
            if reference_seeds is None:
                reference_seeds = seeds
            elif seeds != reference_seeds:
                raise ValueError(
                    f"{dataset}: seed mismatch for {model}. "
                    f"Expected {sorted(reference_seeds)}, got {sorted(seeds)}"
                )

        if reference_seeds is None or len(reference_seeds) < 5:
            raise ValueError(f"{dataset}: insufficient aligned seeds")


def build_overall_forecast_table(
    stochastic: pd.DataFrame,
    deterministic_path: Path,
) -> pd.DataFrame:
    deterministic = pd.read_csv(deterministic_path).copy()
    deterministic = deterministic[
        deterministic["scenario"] == "forecast_original"
    ].copy()

    deterministic_summary = deterministic.rename(
        columns={
            "RMSE": "RMSE_mean",
            "MAE": "MAE_mean",
            "sMAPE": "sMAPE_mean",
            "R2": "R2_mean",
        }
    )
    deterministic_summary["RMSE_std"] = np.nan
    deterministic_summary["MAE_std"] = np.nan
    deterministic_summary["sMAPE_std"] = np.nan
    deterministic_summary["R2_std"] = np.nan
    deterministic_summary["n_runs"] = 1
    deterministic_summary["result_type"] = "single_deterministic_run"

    stochastic_forecast = stochastic[
        stochastic["scenario"] == "forecast_original"
    ].copy()

    stochastic_summary = (
        stochastic_forecast
        .groupby(["dataset", "model"], as_index=False)
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            sMAPE_mean=("sMAPE", "mean"),
            sMAPE_std=("sMAPE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            n_runs=("seed", "count"),
        )
    )
    stochastic_summary["result_type"] = "multiseed_mean"

    columns = [
        "dataset",
        "model",
        "RMSE_mean",
        "RMSE_std",
        "MAE_mean",
        "MAE_std",
        "sMAPE_mean",
        "sMAPE_std",
        "R2_mean",
        "R2_std",
        "n_runs",
        "result_type",
    ]

    overall = pd.concat(
        [
            deterministic_summary[columns],
            stochastic_summary[columns],
        ],
        ignore_index=True,
    )
    overall["RMSE_rank"] = (
        overall.groupby("dataset")["RMSE_mean"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    return overall.sort_values(
        ["dataset", "RMSE_rank", "model"]
    ).reset_index(drop=True)


def forecast_inference(
    stochastic: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast = stochastic[
        stochastic["scenario"] == "forecast_original"
    ].copy()

    friedman_rows: list[dict] = []
    pairwise_rows: list[dict] = []

    for dataset in sorted(forecast["dataset"].unique()):
        dataset_frame = forecast[forecast["dataset"] == dataset]

        for metric in FORECAST_METRICS:
            pivot = dataset_frame.pivot(
                index="seed",
                columns="model",
                values=metric,
            ).sort_index()

            missing_models = set(STOCHASTIC_MODELS).difference(pivot.columns)
            if missing_models:
                raise ValueError(
                    f"{dataset}/{metric}: missing models {sorted(missing_models)}"
                )

            arrays = [pivot[model].to_numpy(dtype=float) for model in STOCHASTIC_MODELS]
            friedman = friedmanchisquare(*arrays)

            friedman_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "n_seeds": len(pivot),
                    "models": "|".join(STOCHASTIC_MODELS),
                    "friedman_statistic": float(friedman.statistic),
                    "friedman_p_value": float(friedman.pvalue),
                }
            )

            local_rows: list[dict] = []
            local_p_values: list[float] = []

            primary = pivot[PRIMARY_MODEL].to_numpy(dtype=float)

            for comparator_name in (
                "NFGA-Core",
                "XGBoost",
                "LSTM",
            ):
                comparator = pivot[comparator_name].to_numpy(dtype=float)

                statistic, p_value, n_nonzero, method = exact_or_auto_wilcoxon(
                    primary,
                    comparator,
                )
                mean_benefit, ci_low, ci_high = paired_bootstrap_ci(
                    primary,
                    comparator,
                    lower_is_better=True,
                    seed=BOOTSTRAP_SEED + len(local_rows),
                )
                rank_biserial = paired_rank_biserial(
                    primary,
                    comparator,
                    lower_is_better=True,
                )

                local_rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "primary_model": PRIMARY_MODEL,
                        "comparator_model": comparator_name,
                        "n_pairs": len(primary),
                        "n_nonzero_pairs": n_nonzero,
                        "primary_mean": float(np.mean(primary)),
                        "comparator_mean": float(np.mean(comparator)),
                        "mean_paired_benefit": mean_benefit,
                        "benefit_ci95_low": ci_low,
                        "benefit_ci95_high": ci_high,
                        "wilcoxon_statistic": statistic,
                        "wilcoxon_p_raw": p_value,
                        "wilcoxon_method": method,
                        "rank_biserial_positive_favors_primary": rank_biserial,
                    }
                )
                local_p_values.append(p_value)

            adjusted = holm_adjust(local_p_values)
            for row, p_adjusted in zip(local_rows, adjusted):
                row["wilcoxon_p_holm"] = p_adjusted
                row["significant_holm_0_05"] = bool(p_adjusted < 0.05)
                pairwise_rows.append(row)

    return pd.DataFrame(friedman_rows), pd.DataFrame(pairwise_rows)


def ablation_table(stochastic: pd.DataFrame) -> pd.DataFrame:
    forecast = stochastic[
        (stochastic["scenario"] == "forecast_original")
        & stochastic["model"].isin(("NFGA-Core", "NFGA-LINEAR"))
    ].copy()

    pivot = forecast.pivot_table(
        index=["dataset", "seed"],
        columns="model",
        values=["RMSE", "MAE", "sMAPE", "K"],
        aggfunc="first",
    )

    rows: list[dict] = []
    for dataset in sorted(forecast["dataset"].unique()):
        dataset_pivot = pivot.loc[dataset]

        for metric in ("RMSE", "MAE", "sMAPE"):
            core = dataset_pivot[(metric, "NFGA-Core")].to_numpy(dtype=float)
            linear = dataset_pivot[(metric, "NFGA-LINEAR")].to_numpy(dtype=float)

            statistic, p_value, n_nonzero, method = exact_or_auto_wilcoxon(
                linear,
                core,
            )
            mean_benefit, ci_low, ci_high = paired_bootstrap_ci(
                linear,
                core,
                lower_is_better=True,
            )
            improvement_percent = 100.0 * (core - linear) / np.maximum(
                np.abs(core),
                EPS,
            )

            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "n_pairs": len(core),
                    "core_mean": float(np.mean(core)),
                    "linear_mean": float(np.mean(linear)),
                    "mean_paired_benefit": mean_benefit,
                    "benefit_ci95_low": ci_low,
                    "benefit_ci95_high": ci_high,
                    "mean_relative_improvement_percent": float(
                        np.mean(improvement_percent)
                    ),
                    "median_relative_improvement_percent": float(
                        np.median(improvement_percent)
                    ),
                    "linear_better_seed_count": int(np.sum(linear < core)),
                    "core_better_seed_count": int(np.sum(core < linear)),
                    "ties": int(np.sum(np.isclose(linear, core))),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_value": p_value,
                    "wilcoxon_method": method,
                    "n_nonzero_pairs": n_nonzero,
                    "rank_biserial_positive_favors_linear": (
                        paired_rank_biserial(
                            linear,
                            core,
                            lower_is_better=True,
                        )
                    ),
                }
            )

        core_k = dataset_pivot[("K", "NFGA-Core")].to_numpy(dtype=float)
        linear_k = dataset_pivot[("K", "NFGA-LINEAR")].to_numpy(dtype=float)
        rows.append(
            {
                "dataset": dataset,
                "metric": "K",
                "n_pairs": len(core_k),
                "core_mean": float(np.mean(core_k)),
                "linear_mean": float(np.mean(linear_k)),
                "mean_paired_benefit": float(np.mean(core_k - linear_k)),
                "benefit_ci95_low": np.nan,
                "benefit_ci95_high": np.nan,
                "mean_relative_improvement_percent": float(
                    np.mean(
                        100.0 * (core_k - linear_k)
                        / np.maximum(np.abs(core_k), EPS)
                    )
                ),
                "median_relative_improvement_percent": float(
                    np.median(
                        100.0 * (core_k - linear_k)
                        / np.maximum(np.abs(core_k), EPS)
                    )
                ),
                "linear_better_seed_count": int(np.sum(linear_k < core_k)),
                "core_better_seed_count": int(np.sum(core_k < linear_k)),
                "ties": int(np.sum(np.isclose(linear_k, core_k))),
                "wilcoxon_statistic": np.nan,
                "wilcoxon_p_value": np.nan,
                "wilcoxon_method": "descriptive_only",
                "n_nonzero_pairs": int(np.sum(~np.isclose(linear_k, core_k))),
                "rank_biserial_positive_favors_linear": np.nan,
            }
        )

    table = pd.DataFrame(rows)
    metric_mask = table["metric"].isin(("RMSE", "MAE", "sMAPE"))
    table.loc[metric_mask, "wilcoxon_p_holm_within_dataset"] = np.nan

    for dataset in sorted(table["dataset"].unique()):
        indices = table.index[
            (table["dataset"] == dataset) & metric_mask
        ].tolist()
        adjusted = holm_adjust(
            table.loc[indices, "wilcoxon_p_value"].astype(float).tolist()
        )
        table.loc[indices, "wilcoxon_p_holm_within_dataset"] = adjusted

    return table


def anomaly_inference(stochastic: pd.DataFrame) -> pd.DataFrame:
    anomaly = stochastic[
        (stochastic["scenario"] != "forecast_original")
        & (stochastic["metric_status"] == "applicable")
    ].copy()

    rows: list[dict] = []

    for dataset in sorted(anomaly["dataset"].unique()):
        for scenario in sorted(
            anomaly.loc[anomaly["dataset"] == dataset, "scenario"].unique()
        ):
            subset = anomaly[
                (anomaly["dataset"] == dataset)
                & (anomaly["scenario"] == scenario)
            ]
            pivot = subset.pivot(
                index="seed",
                columns="model",
                values="F1",
            ).sort_index()

            if PRIMARY_MODEL not in pivot.columns:
                continue

            local_rows: list[dict] = []
            local_p_values: list[float] = []

            primary = pivot[PRIMARY_MODEL].to_numpy(dtype=float)

            for comparator_name in (
                "NFGA-Core",
                "XGBoost",
                "LSTM",
            ):
                if comparator_name not in pivot.columns:
                    continue

                comparator = pivot[comparator_name].to_numpy(dtype=float)
                valid = np.isfinite(primary) & np.isfinite(comparator)

                if int(valid.sum()) < 3:
                    continue

                p_primary = primary[valid]
                p_comparator = comparator[valid]

                statistic, p_value, n_nonzero, method = exact_or_auto_wilcoxon(
                    p_primary,
                    p_comparator,
                )
                mean_benefit, ci_low, ci_high = paired_bootstrap_ci(
                    p_primary,
                    p_comparator,
                    lower_is_better=False,
                )

                local_rows.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "metric": "F1",
                        "primary_model": PRIMARY_MODEL,
                        "comparator_model": comparator_name,
                        "n_pairs": len(p_primary),
                        "primary_mean": float(np.mean(p_primary)),
                        "comparator_mean": float(np.mean(p_comparator)),
                        "mean_paired_benefit": mean_benefit,
                        "benefit_ci95_low": ci_low,
                        "benefit_ci95_high": ci_high,
                        "wilcoxon_statistic": statistic,
                        "wilcoxon_p_raw": p_value,
                        "wilcoxon_method": method,
                        "n_nonzero_pairs": n_nonzero,
                        "rank_biserial_positive_favors_primary": (
                            paired_rank_biserial(
                                p_primary,
                                p_comparator,
                                lower_is_better=False,
                            )
                        ),
                    }
                )
                local_p_values.append(p_value)

            if local_rows:
                adjusted = holm_adjust(local_p_values)
                for row, p_adjusted in zip(local_rows, adjusted):
                    row["wilcoxon_p_holm"] = p_adjusted
                    row["significant_holm_0_05"] = bool(p_adjusted < 0.05)
                    rows.append(row)

    return pd.DataFrame(rows)


def write_markdown_summary(
    path: Path,
    overall: pd.DataFrame,
    ablation: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> None:
    lines: list[str] = [
        "# Stage 7 Statistical Audit",
        "",
        "## Overall RMSE ranking",
        "",
    ]

    for dataset in sorted(overall["dataset"].unique()):
        lines.append(f"### {dataset}")
        subset = overall[overall["dataset"] == dataset].sort_values("RMSE_rank")
        for _, row in subset.iterrows():
            std_text = (
                ""
                if pd.isna(row["RMSE_std"])
                else f" ± {row['RMSE_std']:.6g}"
            )
            lines.append(
                f"- {int(row['RMSE_rank'])}. {row['model']}: "
                f"{row['RMSE_mean']:.6g}{std_text}"
            )
        lines.append("")

    lines.extend(
        [
            "## Matched NFGA ablation",
            "",
            "Positive benefit means NFGA-LINEAR has lower error than NFGA-Core.",
            "",
        ]
    )

    rmse_ablation = ablation[ablation["metric"] == "RMSE"]
    for _, row in rmse_ablation.iterrows():
        p_holm = row["wilcoxon_p_holm_within_dataset"]
        lines.append(
            f"- {row['dataset']}: Core={row['core_mean']:.6g}, "
            f"LINEAR={row['linear_mean']:.6g}, "
            f"LINEAR better in {int(row['linear_better_seed_count'])}/"
            f"{int(row['n_pairs'])} seeds, "
            f"Holm-adjusted p={p_holm:.6g}."
        )

    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "- Statistical superiority is claimed only when the Holm-adjusted p-value is below 0.05 and the effect direction favors NFGA-LINEAR.",
            "- Deterministic single-run models are ranked descriptively; they are not included in seed-level Wilcoxon tests.",
            "- Proxy robust-z anomaly labels remain exploratory and are not treated as verified ground truth.",
            "",
        ]
    )

    significant = pairwise[
        (pairwise["metric"] == "RMSE")
        & (pairwise["significant_holm_0_05"])
        & (
            pairwise["rank_biserial_positive_favors_primary"] > 0
        )
    ]
    if significant.empty:
        lines.append(
            "No Holm-adjusted RMSE comparison supports a blanket superiority claim."
        )
    else:
        lines.append("Supported RMSE comparisons:")
        for _, row in significant.iterrows():
            lines.append(
                f"- {row['dataset']}: NFGA-LINEAR vs "
                f"{row['comparator_model']}, "
                f"p={row['wilcoxon_p_holm']:.6g}, "
                f"rank-biserial={row['rank_biserial_positive_favors_primary']:.3f}."
            )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical audit for corrected NFGA-LINEAR experiments."
    )
    parser.add_argument("--stage3-dir", required=True)
    parser.add_argument("--stage4-dir", required=True)
    parser.add_argument("--stage5-dir", required=True)
    parser.add_argument("--stage6-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    stage3 = Path(args.stage3_dir)
    stage4 = Path(args.stage4_dir)
    stage5 = Path(args.stage5_dir)
    stage6 = Path(args.stage6_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    stochastic = load_stochastic_results(
        stage4 / "xgboost_raw_results.csv",
        stage5 / "lstm_raw_results.csv",
        stage6 / "nfga_raw_results.csv",
    )
    validate_seed_alignment(stochastic)

    overall = build_overall_forecast_table(
        stochastic,
        stage3 / "statistical_forecast_results.csv",
    )
    friedman, forecast_pairwise = forecast_inference(stochastic)
    ablation = ablation_table(stochastic)
    anomaly_pairwise = anomaly_inference(stochastic)

    overall.to_csv(output / "overall_forecast_ranking.csv", index=False)
    friedman.to_csv(output / "forecast_friedman_tests.csv", index=False)
    forecast_pairwise.to_csv(
        output / "forecast_pairwise_wilcoxon.csv",
        index=False,
    )
    ablation.to_csv(output / "matched_nfga_ablation.csv", index=False)
    anomaly_pairwise.to_csv(
        output / "anomaly_pairwise_wilcoxon.csv",
        index=False,
    )

    write_markdown_summary(
        output / "stage7_statistical_summary.md",
        overall,
        ablation,
        forecast_pairwise,
    )

    metadata = {
        "primary_model": PRIMARY_MODEL,
        "stochastic_models": list(STOCHASTIC_MODELS),
        "forecast_metrics": list(FORECAST_METRICS),
        "multiple_comparison_correction": "Holm step-down within each dataset and metric",
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "interpretation": (
            "Positive paired benefit and positive rank-biserial effect favor "
            "NFGA-LINEAR. Deterministic single-run baselines are descriptive only."
        ),
    }
    (output / "statistical_audit_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== Stage 7 statistical audit completed =====")
    print("\nOverall ranking:")
    print(overall.to_string(index=False))

    print("\nMatched NFGA ablation:")
    print(
        ablation[
            ablation["metric"].isin(("RMSE", "MAE", "sMAPE"))
        ].to_string(index=False)
    )

    print("\nForecast pairwise Wilcoxon:")
    print(forecast_pairwise.to_string(index=False))

    print("\nFriedman tests:")
    print(friedman.to_string(index=False))

    print("\nOutputs:", output)


if __name__ == "__main__":
    main()
