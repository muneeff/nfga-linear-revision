from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STOCHASTIC_MODELS = {
    "XGBoost",
    "LSTM",
    "NFGA-Core",
    "NFGA-LINEAR",
}


def ensure_output_dirs(root: Path) -> tuple[Path, Path]:
    figures = root / "figures"
    tables = root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return figures, tables


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        base_path.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def format_value(value: float, dataset: str) -> str:
    if pd.isna(value):
        return "N/A"
    if dataset == "cholera":
        return f"{value:.3f}"
    return f"{value:.4f}"


def latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    output = str(value)
    for old, new in replacements.items():
        output = output.replace(old, new)
    return output


def plot_forecast_rankings(
    forecast: pd.DataFrame,
    figures_dir: Path,
) -> list[dict]:
    manifest: list[dict] = []

    for dataset in sorted(forecast["dataset"].unique()):
        subset = (
            forecast[forecast["dataset"] == dataset]
            .sort_values("RMSE_mean", ascending=True)
            .reset_index(drop=True)
        )

        y = np.arange(len(subset))
        values = subset["RMSE_mean"].to_numpy(dtype=float)
        errors = subset["RMSE_std"].fillna(0.0).to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        ax.barh(y, values, xerr=errors, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(subset["model"])
        ax.invert_yaxis()
        ax.set_xlabel("RMSE")
        ax.set_title(f"Forecasting RMSE — {dataset.capitalize()}")
        ax.grid(axis="x", alpha=0.25)

        for index, (value, error, model) in enumerate(
            zip(values, errors, subset["model"])
        ):
            label = format_value(value, dataset)
            if model in STOCHASTIC_MODELS and error > 0:
                label += f" ± {format_value(error, dataset)}"
            ax.text(
                value + max(values) * 0.012,
                index,
                label,
                va="center",
                fontsize=8,
            )

        if dataset == "electricity":
            prophet = subset.loc[
                subset["model"] == "Prophet",
                "RMSE_mean",
            ]
            xgb = subset.loc[
                subset["model"] == "XGBoost",
                "RMSE_mean",
            ]
            if len(prophet) and len(xgb):
                difference = abs(float(prophet.iloc[0] - xgb.iloc[0]))
                if difference < 1e-4:
                    ax.text(
                        0.99,
                        0.02,
                        "Prophet and XGBoost are tied at four decimal places.",
                        ha="right",
                        va="bottom",
                        transform=ax.transAxes,
                        fontsize=8,
                    )

        base = figures_dir / f"forecast_rmse_{dataset}"
        save_figure(fig, base)
        manifest.append(
            {
                "figure": base.name,
                "purpose": (
                    f"Final RMSE ranking for {dataset}; error bars denote "
                    "standard deviation across ten seeds where applicable."
                ),
            }
        )

    return manifest


def plot_nfga_ablation(
    matched_ablation: pd.DataFrame,
    figures_dir: Path,
) -> dict:
    subset = matched_ablation[
        matched_ablation["metric"] == "RMSE"
    ].copy()
    subset = subset.sort_values(
        "mean_relative_improvement_percent",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.barh(
        subset["dataset"],
        subset["mean_relative_improvement_percent"],
    )
    ax.set_xlabel("Mean RMSE improvement of NFGA-LINEAR over NFGA-Core (%)")
    ax.set_title("Matched ten-seed NFGA ablation")
    ax.grid(axis="x", alpha=0.25)

    for index, row in enumerate(subset.itertuples(index=False)):
        ax.text(
            row.mean_relative_improvement_percent + 0.8,
            index,
            (
                f"{row.mean_relative_improvement_percent:.1f}% "
                f"(Holm p={row.wilcoxon_p_holm_within_dataset:.4f})"
            ),
            va="center",
            fontsize=8,
        )

    base = figures_dir / "nfga_matched_rmse_ablation"
    save_figure(fig, base)
    return {
        "figure": base.name,
        "purpose": (
            "Matched-seed RMSE benefit of local linear consequents over "
            "zero-order NFGA-Core."
        ),
    }


def plot_detector_macro(
    detector_macro: pd.DataFrame,
    figures_dir: Path,
) -> dict:
    pivot = detector_macro.pivot(
        index="detector",
        columns="dataset",
        values="macro_F1",
    ).fillna(0.0)

    ax = pivot.plot(
        kind="bar",
        figsize=(10.0, 5.4),
        width=0.8,
    )
    ax.set_ylabel("Macro-F1 across 5% and 10% injected scenarios")
    ax.set_xlabel("Detector")
    ax.set_title("Synthetic anomaly-detection performance")
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Dataset")
    plt.xticks(rotation=30, ha="right")

    fig = ax.get_figure()
    base = figures_dir / "detector_macro_f1"
    save_figure(fig, base)
    return {
        "figure": base.name,
        "purpose": (
            "Macro-F1 comparison across synthetic injected scenarios. "
            "Proxy robust-z labels are excluded."
        ),
    }


def plot_sensitivity(
    sensitivity: pd.DataFrame,
    figures_dir: Path,
) -> list[dict]:
    manifest: list[dict] = []

    for dataset in sorted(sensitivity["dataset"].unique()):
        subset = sensitivity[
            (sensitivity["dataset"] == dataset)
            & (sensitivity["config_id"] != "baseline")
        ].copy()
        subset = subset.sort_values(
            "RMSE_change_percent_vs_baseline",
            ascending=True,
        )

        labels = (
            subset["factor"].astype(str)
            + ": "
            + subset["level"].astype(str)
        )

        fig, ax = plt.subplots(figsize=(9.0, 6.2))
        ax.barh(
            np.arange(len(subset)),
            subset["RMSE_change_percent_vs_baseline"],
        )
        ax.axvline(0.0, linewidth=1.0)
        ax.set_yticks(np.arange(len(subset)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("RMSE change relative to baseline (%)")
        ax.set_title(
            f"NFGA-LINEAR one-factor sensitivity — {dataset.capitalize()}"
        )
        ax.grid(axis="x", alpha=0.25)

        base = figures_dir / f"nfga_sensitivity_{dataset}"
        save_figure(fig, base)
        manifest.append(
            {
                "figure": base.name,
                "purpose": (
                    f"One-factor-at-a-time RMSE sensitivity for {dataset}. "
                    "Negative values indicate improvement."
                ),
            }
        )

    return manifest


def plot_threshold_sensitivity(
    threshold_macro: pd.DataFrame,
    figures_dir: Path,
) -> list[dict]:
    subset = threshold_macro[
        threshold_macro["model"] == "NFGA-LINEAR"
    ].copy()

    manifest: list[dict] = []

    for dataset in sorted(subset["dataset"].unique()):
        local = subset[
            subset["dataset"] == dataset
        ].sort_values("k_mad")

        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        ax.plot(
            local["k_mad"],
            local["macro_F1_mean"],
            marker="o",
        )
        ax.axvline(
            3.5,
            linestyle="--",
            linewidth=1.0,
            label="Prespecified baseline k=3.5",
        )
        ax.set_xlabel("MAD multiplier k")
        ax.set_ylabel("Macro-F1")
        ax.set_title(
            f"NFGA-LINEAR threshold sensitivity — {dataset.capitalize()}"
        )
        ax.grid(alpha=0.25)
        ax.legend()

        base = figures_dir / f"threshold_sensitivity_{dataset}"
        save_figure(fig, base)
        manifest.append(
            {
                "figure": base.name,
                "purpose": (
                    f"Descriptive anomaly-threshold sensitivity for {dataset}; "
                    "the main analysis retains k=3.5."
                ),
            }
        )

    return manifest


def plot_resource_latency(
    resources: pd.DataFrame,
    figures_dir: Path,
) -> list[dict]:
    manifest: list[dict] = []

    for dataset in sorted(resources["dataset"].unique()):
        subset = (
            resources[resources["dataset"] == dataset]
            .sort_values("total_online_step_time_mean_sec")
            .copy()
        )

        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        ax.barh(
            subset["model"],
            subset["total_online_step_time_mean_sec"],
        )
        ax.set_xscale("log")
        ax.set_xlabel("Mean online time per step (seconds, log scale)")
        ax.set_title(
            f"Online computational cost — {dataset.capitalize()}"
        )
        ax.grid(axis="x", alpha=0.25)

        base = figures_dir / f"online_latency_{dataset}"
        save_figure(fig, base)
        manifest.append(
            {
                "figure": base.name,
                "purpose": (
                    f"Online per-step timing for {dataset}. Prophet and ARIMA "
                    "include walk-forward refitting; fixed models do not."
                ),
            }
        )

    return manifest


def forecast_latex_table(forecast: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Corrected forecasting performance under the full test walk-forward protocol.}",
        r"\label{tab:final_forecast}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Model & RMSE & MAE & sMAPE (\%) & $R^2$ \\",
        r"\midrule",
    ]

    for dataset in ("cholera", "ilinet", "electricity"):
        subset = forecast[
            forecast["dataset"] == dataset
        ].sort_values("RMSE_mean")

        for row in subset.itertuples(index=False):
            rmse_text = format_value(row.RMSE_mean, dataset)
            mae_text = format_value(row.MAE_mean, dataset)
            smape_text = f"{row.sMAPE_mean:.3f}"
            r2_text = f"{row.R2_mean:.4f}"

            if row.model in STOCHASTIC_MODELS:
                rmse_text += (
                    r" $\pm$ "
                    + format_value(row.RMSE_std, dataset)
                )
                mae_text += (
                    r" $\pm$ "
                    + format_value(row.MAE_std, dataset)
                )

            lines.append(
                f"{latex_escape(dataset.capitalize())} & "
                f"{latex_escape(row.model)} & "
                f"{rmse_text} & {mae_text} & "
                f"{smape_text} & {r2_text} \\\\"
            )
        lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\linewidth}",
            r"\footnotesize Deterministic methods are reported as single runs; "
            r"XGBoost, LSTM, NFGA-Core, and NFGA-LINEAR are mean $\pm$ standard "
            r"deviation across ten matched seeds. Prophet and XGBoost are "
            r"numerically tied on Electricity at four decimal places.",
            r"\end{minipage}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def ablation_latex_table(ablation: pd.DataFrame) -> str:
    subset = ablation[ablation["metric"] == "RMSE"].copy()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Matched ten-seed RMSE ablation of local linear consequents.}",
        r"\label{tab:nfga_ablation}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Dataset & Core RMSE & Linear RMSE & Improvement & Holm $p$ \\",
        r"\midrule",
    ]

    for row in subset.itertuples(index=False):
        dataset = row.dataset
        lines.append(
            f"{latex_escape(dataset.capitalize())} & "
            f"{format_value(row.core_mean, dataset)} & "
            f"{format_value(row.linear_mean, dataset)} & "
            f"{row.mean_relative_improvement_percent:.1f}\\% & "
            f"{row.wilcoxon_p_holm_within_dataset:.4f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def detector_latex_table(macro: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Macro performance across the 5\% and 10\% synthetic anomaly-injection scenarios.}",
        r"\label{tab:detector_macro}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Detector & Precision & Recall & F1 & FPR \\",
        r"\midrule",
    ]

    for dataset in ("cholera", "ilinet", "electricity"):
        subset = macro[
            macro["dataset"] == dataset
        ].sort_values("macro_F1", ascending=False)

        for row in subset.itertuples(index=False):
            lines.append(
                f"{latex_escape(dataset.capitalize())} & "
                f"{latex_escape(row.detector)} & "
                f"{row.macro_Precision:.3f} & "
                f"{row.macro_Recall:.3f} & "
                f"{row.macro_F1:.3f} & "
                f"{row.macro_FalsePositiveRate:.3f} \\\\"
            )
        lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\linewidth}",
            r"\footnotesize Proxy robust-z labels are excluded because they are "
            r"not verified ground truth; Robust-Z evaluation against labels "
            r"generated by the same rule is circular.",
            r"\end{minipage}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def resource_latex_table(resources: pd.DataFrame) -> str:
    selected_models = {
        "ARIMA",
        "Prophet",
        "XGBoost",
        "LSTM",
        "NFGA-Core",
        "NFGA-LINEAR",
    }
    subset = resources[
        resources["model"].isin(selected_models)
    ].copy()

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Measured computational costs in the experimental environment.}",
        r"\label{tab:resource_cost}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Model & Training (s) & Online step (ms) & Stored size (KB) \\",
        r"\midrule",
    ]

    for dataset in ("cholera", "ilinet", "electricity"):
        local = subset[
            subset["dataset"] == dataset
        ].sort_values("model")
        for row in local.itertuples(index=False):
            training = (
                "N/A"
                if pd.isna(row.training_time_mean_sec)
                else f"{row.training_time_mean_sec:.3f}"
            )
            online_ms = 1000.0 * row.total_online_step_time_mean_sec
            size_kb = (
                "N/A"
                if pd.isna(row.model_size_mean_bytes)
                else f"{row.model_size_mean_bytes / 1024.0:.2f}"
            )
            lines.append(
                f"{latex_escape(dataset.capitalize())} & "
                f"{latex_escape(row.model)} & "
                f"{training} & {online_ms:.3f} & {size_kb} \\\\"
            )
        lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\linewidth}",
            r"\footnotesize Timing is hardware- and implementation-dependent. "
            r"ARIMA and Prophet online costs include per-step refitting. Stored "
            r"size definitions differ by model: NFGA reports parameter arrays, "
            r"LSTM reports weight arrays, XGBoost reports the serialized booster, "
            r"and Prophet reports a serialized fitted object; sizes are therefore "
            r"approximate rather than strictly equivalent.",
            r"\end{minipage}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def write_tables(
    tables_dir: Path,
    forecast: pd.DataFrame,
    ablation: pd.DataFrame,
    detector_macro: pd.DataFrame,
    resources: pd.DataFrame,
) -> None:
    table_text = "\n\n".join(
        [
            forecast_latex_table(forecast),
            ablation_latex_table(ablation),
            detector_latex_table(detector_macro),
            resource_latex_table(resources),
        ]
    )
    (tables_dir / "manuscript_tables.tex").write_text(
        table_text,
        encoding="utf-8",
    )

    notes = """# Table and figure notes

1. RMSE is the prespecified primary forecasting endpoint.
2. Prophet and XGBoost on Electricity differ by approximately 1.06e-6 RMSE and
   must be described as numerically tied at the reported precision.
3. Statistical comparisons among stochastic models use matched seeds. The
   deterministic models are ranked descriptively only.
4. Synthetic anomaly scenarios contain few positives; detector rankings must
   not be generalized to verified real-world anomalies.
5. Proxy robust-z labels are exploratory and excluded from the main detector
   ranking.
6. Model-size measurements use different serialization/parameter definitions
   and are approximate rather than strictly commensurate.
7. The sensitivity study is one-factor-at-a-time and does not estimate
   hyperparameter interactions.
"""
    (tables_dir / "manuscript_table_notes.md").write_text(
        notes,
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate manuscript-ready figures and LaTeX tables."
    )
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
    output = Path(args.output_dir)
    figures_dir, tables_dir = ensure_output_dirs(output)

    forecast = pd.read_csv(
        stage13 / "final_forecast_ranking.csv"
    )
    detector_macro = pd.read_csv(
        stage13 / "final_detector_macro_results.csv"
    )
    resources = pd.read_csv(
        stage13 / "final_resource_costs.csv"
    )
    matched_ablation = pd.read_csv(
        stage7 / "matched_nfga_ablation.csv"
    )
    sensitivity = pd.read_csv(
        stage8 / "nfga_linear_sensitivity_summary.csv"
    )
    threshold_macro = pd.read_csv(
        stage9 / "anomaly_threshold_macro_injected.csv"
    )

    manifest: list[dict] = []
    manifest.extend(
        plot_forecast_rankings(forecast, figures_dir)
    )
    manifest.append(
        plot_nfga_ablation(matched_ablation, figures_dir)
    )
    manifest.append(
        plot_detector_macro(detector_macro, figures_dir)
    )
    manifest.extend(
        plot_sensitivity(sensitivity, figures_dir)
    )
    manifest.extend(
        plot_threshold_sensitivity(
            threshold_macro,
            figures_dir,
        )
    )
    manifest.extend(
        plot_resource_latency(resources, figures_dir)
    )

    pd.DataFrame(manifest).to_csv(
        output / "figure_manifest.csv",
        index=False,
    )

    write_tables(
        tables_dir,
        forecast,
        matched_ablation,
        detector_macro,
        resources,
    )

    metadata = {
        "figure_count": len(manifest),
        "formats": ["PDF", "PNG at 300 dpi"],
        "tables": [
            "final forecasting performance",
            "matched NFGA ablation",
            "macro detector performance",
            "computational resource costs",
        ],
        "plotting_library": "matplotlib",
        "important_notes": [
            "No result values are recomputed in this script.",
            "Prophet and XGBoost are treated as numerically tied on Electricity.",
            "Error bars are shown only where multiseed standard deviations exist.",
        ],
    }
    (output / "manuscript_asset_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== Manuscript assets completed =====")
    print(f"Figures: {figures_dir}")
    print(f"Tables:  {tables_dir}")
    print(pd.DataFrame(manifest).to_string(index=False))


if __name__ == "__main__":
    main()
