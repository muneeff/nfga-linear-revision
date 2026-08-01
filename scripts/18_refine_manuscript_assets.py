from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET_ORDER = ("cholera", "ilinet", "electricity")


def holm_adjust(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p) - rank) * p[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )


def corrected_ablation(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[frame["metric"] == "RMSE"].copy()
    out = out.set_index("dataset").loc[list(DATASET_ORDER)].reset_index()
    out["primary_rmse_holm_across_datasets"] = holm_adjust(
        out["wilcoxon_p_value"].astype(float).tolist()
    )
    out["significant_primary_holm_0_05"] = (
        out["primary_rmse_holm_across_datasets"] < 0.05
    )
    return out


def forecast_figure(forecast: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 9.2),
        constrained_layout=True,
    )

    for i, dataset in enumerate(DATASET_ORDER):
        ax = axes[i]
        local = (
            forecast[forecast["dataset"] == dataset]
            .sort_values("RMSE_mean")
            .reset_index(drop=True)
        )
        pos = np.arange(len(local))
        vals = local["RMSE_mean"].to_numpy(float)
        errs = local["RMSE_std"].fillna(0.0).to_numpy(float)

        ax.barh(pos, vals, xerr=errs, capsize=2.5)
        ax.set_yticks(pos)
        ax.set_yticklabels(local["model"])
        ax.invert_yaxis()
        ax.set_xlabel("RMSE")
        ax.set_title(dataset.capitalize())
        ax.grid(axis="x", alpha=0.25)

        maximum = float(np.max(vals + errs))
        ax.set_xlim(0.0, maximum * 1.22)

        for j, row in enumerate(local.itertuples(index=False)):
            value = float(row.RMSE_mean)
            error = 0.0 if pd.isna(row.RMSE_std) else float(row.RMSE_std)
            label = f"{value:.1f}" if dataset == "cholera" else f"{value:.4f}"
            ax.text(
                value + error + maximum * 0.015,
                j,
                label,
                va="center",
                fontsize=7.5,
            )

        if dataset == "electricity":
            ax.text(
                0.99,
                0.02,
                "Prophet and XGBoost are tied at four decimal places.",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
            )

        panel(ax, f"({chr(97 + i)})")

    fig.suptitle(
        "Corrected forecasting RMSE under the full walk-forward protocol",
        fontsize=12,
    )
    save(fig, outdir / "fig1_forecast_rmse_overview")


def ablation_figure(ablation: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    values = ablation["mean_relative_improvement_percent"].to_numpy(float)
    positions = np.arange(len(ablation))

    ax.barh(positions, values)
    ax.set_yticks(positions)
    ax.set_yticklabels(ablation["dataset"].str.capitalize())
    ax.invert_yaxis()
    ax.set_xlabel("Mean RMSE improvement of NFGA-LINEAR over NFGA-Core (%)")
    ax.set_title("Matched ten-seed primary RMSE ablation")
    ax.grid(axis="x", alpha=0.25)

    maximum = float(np.max(values))
    ax.set_xlim(0.0, maximum * 1.34)

    for i, row in enumerate(ablation.itertuples(index=False)):
        ax.text(
            float(row.mean_relative_improvement_percent) + maximum * 0.02,
            i,
            f"{row.mean_relative_improvement_percent:.1f}%  "
            f"(Holm p={row.primary_rmse_holm_across_datasets:.4f})",
            va="center",
            fontsize=8,
        )

    save(fig, outdir / "fig2_nfga_primary_rmse_ablation")


def detector_figure(macro: pd.DataFrame, outdir: Path) -> None:
    names = {
        "IsolationForest": "Isolation Forest",
        "LocalOutlierFactor": "LOF",
        "NFGA-LINEAR residual detector": "NFGA residual",
        "OneClassSVM": "OCSVM",
        "Prophet residual detector": "Prophet residual",
        "Robust-Z": "Robust-Z",
    }
    frame = macro.copy()
    frame["detector_short"] = frame["detector"].map(names).fillna(frame["detector"])

    order = [
        "Isolation Forest",
        "LOF",
        "NFGA residual",
        "OCSVM",
        "Prophet residual",
        "Robust-Z",
    ]
    pivot = frame.pivot(
        index="detector_short",
        columns="dataset",
        values="macro_F1",
    ).reindex(order)
    pivot = pivot[[dataset for dataset in DATASET_ORDER if dataset in pivot.columns]]

    ax = pivot.plot(kind="bar", figsize=(7.2, 4.4), width=0.82)
    ax.set_ylabel("Macro-F1 across 5% and 10% injected scenarios")
    ax.set_xlabel("")
    ax.set_title("Synthetic anomaly-detection performance")
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        title="Dataset",
        labels=[name.capitalize() for name in pivot.columns],
        ncol=3,
        loc="upper left",
    )
    ax.tick_params(axis="x", labelrotation=25)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    fig = ax.get_figure()
    fig.tight_layout()
    save(fig, outdir / "fig3_detector_macro_f1")


def latency_figure(resources: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 9.0),
        constrained_layout=True,
    )

    for i, dataset in enumerate(DATASET_ORDER):
        ax = axes[i]
        local = (
            resources[resources["dataset"] == dataset]
            .sort_values("total_online_step_time_mean_sec")
            .reset_index(drop=True)
        )
        positions = np.arange(len(local))
        ax.barh(positions, local["total_online_step_time_mean_sec"])
        ax.set_yticks(positions)
        ax.set_yticklabels(local["model"])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("Mean online time per step (seconds, log scale)")
        ax.set_title(dataset.capitalize())
        ax.grid(axis="x", alpha=0.25)
        panel(ax, f"({chr(97 + i)})")

    fig.suptitle("Measured online computational cost", fontsize=12)
    save(fig, outdir / "fig4_online_latency_overview")


def sensitivity_figure(sensitivity: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 10.0),
        constrained_layout=True,
    )

    for i, dataset in enumerate(DATASET_ORDER):
        ax = axes[i]
        local = sensitivity[
            (sensitivity["dataset"] == dataset)
            & (sensitivity["config_id"] != "baseline")
        ].copy()
        local = local.sort_values("RMSE_change_percent_vs_baseline")
        labels = local["factor"].astype(str) + ": " + local["level"].astype(str)
        positions = np.arange(len(local))

        ax.barh(positions, local["RMSE_change_percent_vs_baseline"])
        ax.axvline(0.0, linewidth=1.0)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("RMSE change relative to baseline (%)")
        ax.set_title(dataset.capitalize())
        ax.grid(axis="x", alpha=0.25)
        panel(ax, f"({chr(97 + i)})")

    fig.suptitle(
        "Supplementary one-factor-at-a-time NFGA-LINEAR sensitivity",
        fontsize=12,
    )
    save(fig, outdir / "figS1_nfga_ofat_sensitivity")


def threshold_figure(threshold: pd.DataFrame, outdir: Path) -> None:
    frame = threshold[threshold["model"] == "NFGA-LINEAR"].copy()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.2, 3.4),
        constrained_layout=True,
    )

    for i, dataset in enumerate(DATASET_ORDER):
        ax = axes[i]
        local = frame[frame["dataset"] == dataset].sort_values("k_mad")
        ax.plot(local["k_mad"], local["macro_F1_mean"], marker="o")
        ax.axvline(3.5, linestyle="--", linewidth=1.0)
        ax.set_xlabel("MAD multiplier k")
        ax.set_ylabel("Macro-F1")
        ax.set_title(dataset.capitalize())
        ax.grid(alpha=0.25)
        panel(ax, f"({chr(97 + i)})")

    fig.suptitle(
        "Supplementary anomaly-threshold sensitivity (baseline k=3.5)",
        fontsize=11,
    )
    save(fig, outdir / "figS2_threshold_sensitivity")


def write_ablation_table(ablation: pd.DataFrame, outdir: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Matched ten-seed primary RMSE ablation of local linear consequents.}",
        r"\label{tab:nfga_primary_rmse_ablation}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Dataset & Core RMSE & Linear RMSE & Improvement & Holm-adjusted $p$ \\",
        r"\midrule",
    ]

    for row in ablation.itertuples(index=False):
        if row.dataset == "cholera":
            core = f"{row.core_mean:.3f}"
            linear = f"{row.linear_mean:.3f}"
        else:
            core = f"{row.core_mean:.4f}"
            linear = f"{row.linear_mean:.4f}"

        line = (
            f"{row.dataset.capitalize()} & {core} & {linear} & "
            f"{row.mean_relative_improvement_percent:.1f}\\% & "
            f"{row.primary_rmse_holm_across_datasets:.4f} "
            + r"\\"
        )
        lines.append(line)

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.96\linewidth}",
            r"\footnotesize Holm correction was applied across the three prespecified "
            r"NFGA-LINEAR versus NFGA-Core RMSE comparisons (one per dataset).",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )

    (outdir / "table_nfga_primary_rmse_ablation_corrected.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_plan(outdir: Path) -> None:
    text = """# Manuscript asset placement plan

## Main manuscript

1. `fig1_forecast_rmse_overview`
2. `fig2_nfga_primary_rmse_ablation`
3. `fig3_detector_macro_f1`
4. `fig4_online_latency_overview`

## Supplementary material

1. `figS1_nfga_ofat_sensitivity`
2. `figS2_threshold_sensitivity`

## Statistical reporting correction

The Stage 14 ablation asset used Holm correction across RMSE, MAE, and sMAPE
within each dataset. The main manuscript claim is defined on the prespecified
primary endpoint RMSE. The corrected multiplicity family therefore contains
the three matched NFGA-LINEAR versus NFGA-Core RMSE tests, one per dataset.

Corrected Holm-adjusted p-values:

- Cholera: 0.0059
- ILINet: 0.0059
- Electricity: 0.0371
"""
    (outdir / "asset_placement_plan.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage7-dir", required=True)
    parser.add_argument("--stage8-dir", required=True)
    parser.add_argument("--stage9-dir", required=True)
    parser.add_argument("--stage13-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    stage7 = Path(args.stage7_dir)
    stage8 = Path(args.stage8_dir)
    stage9 = Path(args.stage9_dir)
    stage13 = Path(args.stage13_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    forecast = pd.read_csv(stage13 / "final_forecast_ranking.csv")
    macro = pd.read_csv(stage13 / "final_detector_macro_results.csv")
    resources = pd.read_csv(stage13 / "final_resource_costs.csv")
    matched = pd.read_csv(stage7 / "matched_nfga_ablation.csv")
    sensitivity = pd.read_csv(stage8 / "nfga_linear_sensitivity_summary.csv")
    threshold = pd.read_csv(stage9 / "anomaly_threshold_macro_injected.csv")

    ablation = corrected_ablation(matched)
    ablation.to_csv(
        outdir / "primary_rmse_ablation_corrected.csv",
        index=False,
    )

    forecast_figure(forecast, outdir)
    ablation_figure(ablation, outdir)
    detector_figure(macro, outdir)
    latency_figure(resources, outdir)
    sensitivity_figure(sensitivity, outdir)
    threshold_figure(threshold, outdir)
    write_ablation_table(ablation, outdir)
    write_plan(outdir)

    metadata = {
        "main_figures": [
            "fig1_forecast_rmse_overview",
            "fig2_nfga_primary_rmse_ablation",
            "fig3_detector_macro_f1",
            "fig4_online_latency_overview",
        ],
        "supplementary_figures": [
            "figS1_nfga_ofat_sensitivity",
            "figS2_threshold_sensitivity",
        ],
        "primary_rmse_holm_adjusted_p": ablation.set_index("dataset")[
            "primary_rmse_holm_across_datasets"
        ].to_dict(),
    }
    (outdir / "refined_asset_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("===== Refined manuscript assets completed =====")
    print(
        ablation[
            [
                "dataset",
                "core_mean",
                "linear_mean",
                "mean_relative_improvement_percent",
                "wilcoxon_p_value",
                "primary_rmse_holm_across_datasets",
                "significant_primary_holm_0_05",
            ]
        ].to_string(index=False)
    )
    print("Outputs:", outdir)


if __name__ == "__main__":
    main()
