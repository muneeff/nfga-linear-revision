from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_ORDER = ("cholera", "ilinet", "electricity")


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )


def copy_existing_assets(stage17_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in stage17_dir.iterdir():
        destination = output_dir / source.name
        if source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def plot_forecast_overview(
    forecast: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.3))
    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        top=0.93,
        bottom=0.075,
        hspace=0.34,
    )

    for panel_index, dataset in enumerate(DATASET_ORDER):
        ax = axes[panel_index]
        subset = (
            forecast[forecast["dataset"] == dataset]
            .sort_values("RMSE_mean", ascending=True)
            .reset_index(drop=True)
        )

        positions = np.arange(len(subset))
        values = subset["RMSE_mean"].to_numpy(dtype=float)
        errors = subset["RMSE_std"].fillna(0.0).to_numpy(dtype=float)

        ax.barh(positions, values, xerr=errors, capsize=2.5)
        ax.set_yticks(positions)
        ax.set_yticklabels(subset["model"])
        ax.invert_yaxis()
        ax.set_xlabel("RMSE", labelpad=4)
        title = "ILINet" if dataset == "ilinet" else dataset.capitalize()
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)

        max_value = float(np.max(values + errors))
        ax.set_xlim(0.0, max_value * 1.22)

        for index, row in subset.iterrows():
            label = (
                f"{row['RMSE_mean']:.1f}"
                if dataset == "cholera"
                else f"{row['RMSE_mean']:.4f}"
            )
            error = 0.0 if pd.isna(row["RMSE_std"]) else float(row["RMSE_std"])
            ax.text(
                float(row["RMSE_mean"]) + error + max_value * 0.015,
                index,
                label,
                va="center",
                fontsize=7.5,
            )

        add_panel_label(ax, f"({chr(97 + panel_index)})")

    fig.suptitle(
        "Corrected forecasting RMSE under the full walk-forward protocol",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Note: Prophet and XGBoost are numerically tied on Electricity "
            "at four decimal places."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )

    save_figure(fig, output_dir / "fig1_forecast_rmse_overview")


def plot_latency_overview(
    resources: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.3))
    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        top=0.93,
        bottom=0.085,
        hspace=0.34,
    )

    for panel_index, dataset in enumerate(DATASET_ORDER):
        ax = axes[panel_index]
        subset = (
            resources[resources["dataset"] == dataset]
            .sort_values("total_online_step_time_mean_sec")
            .reset_index(drop=True)
        )

        ax.barh(
            np.arange(len(subset)),
            subset["total_online_step_time_mean_sec"],
        )
        ax.set_yticks(np.arange(len(subset)))
        ax.set_yticklabels(subset["model"])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel(
            "Mean online time per step (seconds, log scale)",
            labelpad=4,
        )
        title = "ILINet" if dataset == "ilinet" else dataset.capitalize()
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        add_panel_label(ax, f"({chr(97 + panel_index)})")

    fig.suptitle("Measured online computational cost", fontsize=12)
    fig.text(
        0.5,
        0.012,
        (
            "Note: ARIMA and Prophet include per-step refitting; "
            "the remaining models use fixed fitted predictors."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )

    save_figure(fig, output_dir / "fig4_online_latency_overview")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the final layout patch to manuscript Figures 1 and 4 "
            "without changing any numerical result."
        )
    )
    parser.add_argument("--stage13-dir", required=True)
    parser.add_argument("--stage17-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    stage13_dir = Path(args.stage13_dir)
    stage17_dir = Path(args.stage17_dir)
    output_dir = Path(args.output_dir)

    copy_existing_assets(stage17_dir, output_dir)

    forecast = pd.read_csv(
        stage13_dir / "final_forecast_ranking.csv"
    )
    resources = pd.read_csv(
        stage13_dir / "final_resource_costs.csv"
    )

    plot_forecast_overview(forecast, output_dir)
    plot_latency_overview(resources, output_dir)

    metadata = {
        "source_stage17": str(stage17_dir),
        "patched_figures": [
            "fig1_forecast_rmse_overview",
            "fig4_online_latency_overview",
        ],
        "changes": [
            "Reserved additional bottom margin for figure notes.",
            "Moved the Electricity numerical-tie note below Figure 1.",
            "Moved the ARIMA/Prophet refitting note below Figure 4.",
            "No numerical values or rankings were changed.",
        ],
    }
    (output_dir / "final_layout_patch_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== Final layout patch completed =====")
    print("Patched Figure 1 and Figure 4.")
    print("Outputs:", output_dir)


if __name__ == "__main__":
    main()
