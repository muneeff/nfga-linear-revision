from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12
INJECTED_SCENARIOS = ("injected_5pct", "injected_10pct")


def load_forecast_results(
    stage3: Path,
    stage4: Path,
    stage5: Path,
    stage6: Path,
    stage12: Path,
) -> pd.DataFrame:
    # Deterministic statistical baselines
    deterministic = pd.read_csv(
        stage3 / "statistical_forecast_results.csv"
    )
    deterministic = deterministic[
        deterministic["scenario"] == "forecast_original"
    ].copy()
    deterministic = deterministic.rename(
        columns={
            "RMSE": "RMSE_mean",
            "MAE": "MAE_mean",
            "sMAPE": "sMAPE_mean",
            "R2": "R2_mean",
        }
    )
    deterministic["RMSE_std"] = np.nan
    deterministic["MAE_std"] = np.nan
    deterministic["sMAPE_std"] = np.nan
    deterministic["R2_std"] = np.nan
    deterministic["n_runs"] = 1
    deterministic["result_type"] = "single_deterministic_run"

    # Stochastic multiseed baselines
    xgb = pd.read_csv(stage4 / "xgboost_forecast_summary.csv").rename(
        columns={"n_seeds": "n_runs"}
    )
    lstm = pd.read_csv(stage5 / "lstm_forecast_summary.csv").rename(
        columns={"n_seeds": "n_runs"}
    )
    nfga = pd.read_csv(stage6 / "nfga_forecast_summary.csv").rename(
        columns={"n_seeds": "n_runs"}
    )

    stochastic = pd.concat([xgb, lstm, nfga], ignore_index=True, sort=False)
    stochastic["result_type"] = "multiseed_mean"

    # Corrected deterministic Prophet baseline
    prophet = pd.read_csv(stage12 / "prophet_forecast_results.csv")
    prophet = prophet[
        prophet["scenario"] == "forecast_original"
    ].copy()
    prophet = prophet.rename(
        columns={
            "RMSE": "RMSE_mean",
            "MAE": "MAE_mean",
            "sMAPE": "sMAPE_mean",
            "R2": "R2_mean",
        }
    )
    prophet["RMSE_std"] = np.nan
    prophet["MAE_std"] = np.nan
    prophet["sMAPE_std"] = np.nan
    prophet["R2_std"] = np.nan
    prophet["n_runs"] = 1
    prophet["result_type"] = "single_deterministic_run"

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

    combined = pd.concat(
        [
            deterministic[columns],
            stochastic[columns],
            prophet[columns],
        ],
        ignore_index=True,
    )

    for metric in ("RMSE", "MAE", "sMAPE"):
        combined[f"{metric}_rank"] = (
            combined.groupby("dataset")[f"{metric}_mean"]
            .rank(method="min", ascending=True)
            .astype(int)
        )

    combined["R2_rank"] = (
        combined.groupby("dataset")["R2_mean"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return combined.sort_values(
        ["dataset", "RMSE_rank", "model"]
    ).reset_index(drop=True)


def prophet_detector_rows(stage12: Path) -> pd.DataFrame:
    prophet = pd.read_csv(stage12 / "prophet_anomaly_results.csv")
    prophet = prophet[prophet["scenario"].isin(INJECTED_SCENARIOS)].copy()

    prophet["detector"] = "Prophet residual detector"
    prophet["detector_category"] = "forecast_residual_detector"
    prophet["n_runs"] = 1
    prophet["n_detected_mean"] = prophet["n_detected"].astype(float)
    prophet["n_detected_std"] = np.nan
    prophet["Precision_mean"] = prophet["Precision"]
    prophet["Precision_std"] = np.nan
    prophet["Recall_mean"] = prophet["Recall"]
    prophet["Recall_std"] = np.nan
    prophet["F1_mean"] = prophet["F1"]
    prophet["F1_std"] = np.nan
    prophet["Specificity_mean"] = prophet["Specificity"]
    prophet["FalsePositiveRate_mean"] = prophet["FalsePositiveRate"]

    balanced = (
        prophet["Recall_mean"] + prophet["Specificity_mean"]
    ) / 2.0
    prophet["BalancedAccuracy_mean"] = balanced

    return prophet[
        [
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
    ]


def load_detector_results(
    stage11: Path,
    stage12: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing = pd.read_csv(
        stage11 / "detector_injected_scenarios_final.csv"
    )

    keep = [
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

    prophet = prophet_detector_rows(stage12)
    combined = pd.concat(
        [existing[keep], prophet[keep]],
        ignore_index=True,
    )

    combined["F1_rank"] = (
        combined.groupby(["dataset", "scenario"])["F1_mean"]
        .rank(method="min", ascending=False)
    )
    combined["FPR_rank"] = (
        combined.groupby(["dataset", "scenario"])[
            "FalsePositiveRate_mean"
        ]
        .rank(method="min", ascending=True)
    )

    combined = combined.sort_values(
        ["dataset", "scenario", "F1_rank", "detector"]
    ).reset_index(drop=True)

    macro = (
        combined.groupby(
            ["dataset", "detector", "detector_category"],
            as_index=False,
        )
        .agg(
            scenarios=("scenario", "count"),
            macro_Precision=("Precision_mean", "mean"),
            macro_Recall=("Recall_mean", "mean"),
            macro_F1=("F1_mean", "mean"),
            macro_FalsePositiveRate=(
                "FalsePositiveRate_mean",
                "mean",
            ),
            macro_BalancedAccuracy=(
                "BalancedAccuracy_mean",
                "mean",
            ),
        )
    )
    macro["macro_F1_rank"] = (
        macro.groupby("dataset")["macro_F1"]
        .rank(method="min", ascending=False)
    )
    macro = macro.sort_values(
        ["dataset", "macro_F1_rank", "detector"]
    ).reset_index(drop=True)

    return combined, macro


def build_resource_table(
    stage3: Path,
    stage4: Path,
    stage5: Path,
    stage6: Path,
    stage12: Path,
) -> pd.DataFrame:
    rows: list[dict] = []

    statistical = pd.read_csv(
        stage3 / "statistical_forecast_results.csv"
    )
    statistical = statistical[
        statistical["scenario"] == "forecast_original"
    ]

    for row in statistical.itertuples(index=False):
        rows.append(
            {
                "dataset": row.dataset,
                "model": row.model,
                "training_time_mean_sec": np.nan,
                "per_step_refit_time_mean_sec": (
                    row.mean_inference_latency_sec
                    if row.model == "ARIMA"
                    else 0.0
                ),
                "inference_latency_mean_sec": (
                    0.0
                    if row.model == "ARIMA"
                    else row.mean_inference_latency_sec
                ),
                "total_online_step_time_mean_sec": (
                    row.mean_inference_latency_sec
                ),
                "model_size_mean_bytes": np.nan,
                "cost_interpretation": (
                    "ARIMA timing includes per-step refitting and forecasting."
                    if row.model == "ARIMA"
                    else "Closed-form deterministic baseline."
                ),
            }
        )

    xgb = pd.read_csv(stage4 / "xgboost_forecast_summary.csv")
    for row in xgb.itertuples(index=False):
        rows.append(
            {
                "dataset": row.dataset,
                "model": row.model,
                "training_time_mean_sec": row.training_time_mean_sec,
                "per_step_refit_time_mean_sec": 0.0,
                "inference_latency_mean_sec": (
                    row.inference_latency_mean_sec
                ),
                "total_online_step_time_mean_sec": (
                    row.inference_latency_mean_sec
                ),
                "model_size_mean_bytes": row.model_size_mean_bytes,
                "cost_interpretation": (
                    "Fixed fitted model; serialized booster size."
                ),
            }
        )

    lstm = pd.read_csv(stage5 / "lstm_forecast_summary.csv")
    for row in lstm.itertuples(index=False):
        rows.append(
            {
                "dataset": row.dataset,
                "model": row.model,
                "training_time_mean_sec": row.training_time_mean_sec,
                "per_step_refit_time_mean_sec": 0.0,
                "inference_latency_mean_sec": (
                    row.inference_latency_mean_sec
                ),
                "total_online_step_time_mean_sec": (
                    row.inference_latency_mean_sec
                ),
                "model_size_mean_bytes": row.model_size_mean_bytes,
                "cost_interpretation": (
                    "Fixed fitted model; stored weight-array size."
                ),
            }
        )

    nfga = pd.read_csv(stage6 / "nfga_forecast_summary.csv")
    for row in nfga.itertuples(index=False):
        rows.append(
            {
                "dataset": row.dataset,
                "model": row.model,
                "training_time_mean_sec": row.training_time_mean_sec,
                "per_step_refit_time_mean_sec": 0.0,
                "inference_latency_mean_sec": (
                    row.inference_latency_mean_sec
                ),
                "total_online_step_time_mean_sec": (
                    row.inference_latency_mean_sec
                ),
                "model_size_mean_bytes": row.model_size_mean_bytes,
                "cost_interpretation": (
                    "Fixed fitted model; antecedent/consequent array size only."
                ),
            }
        )

    prophet = pd.read_csv(stage12 / "prophet_forecast_results.csv")
    for row in prophet.itertuples(index=False):
        rows.append(
            {
                "dataset": row.dataset,
                "model": row.model,
                "training_time_mean_sec": np.nan,
                "per_step_refit_time_mean_sec": row.mean_refit_time_sec,
                "inference_latency_mean_sec": (
                    row.mean_inference_latency_sec
                ),
                "total_online_step_time_mean_sec": (
                    row.mean_refit_time_sec
                    + row.mean_inference_latency_sec
                ),
                "model_size_mean_bytes": row.model_size_bytes,
                "cost_interpretation": (
                    "Walk-forward Prophet refits at every test step; "
                    "serialized final fitted object size."
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["dataset", "model"]
    ).reset_index(drop=True)


def build_claim_audit(
    forecast: pd.DataFrame,
    detectors: pd.DataFrame,
    detector_macro: pd.DataFrame,
) -> dict:
    best_forecast = (
        forecast.sort_values(["dataset", "RMSE_rank", "model"])
        .groupby("dataset", as_index=False)
        .first()
    )

    scenario_winners = (
        detectors.sort_values(
            ["dataset", "scenario", "F1_rank", "detector"]
        )
        .groupby(["dataset", "scenario"], as_index=False)
        .first()
    )

    winner_counts = (
        scenario_winners.groupby("detector")
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )

    macro_winners = detector_macro[
        detector_macro["macro_F1_rank"] == 1
    ]

    electricity = forecast[forecast["dataset"] == "electricity"]
    prophet_rmse = float(
        electricity.loc[
            electricity["model"] == "Prophet",
            "RMSE_mean",
        ].iloc[0]
    )
    xgb_rmse = float(
        electricity.loc[
            electricity["model"] == "XGBoost",
            "RMSE_mean",
        ].iloc[0]
    )

    return {
        "best_forecast_by_dataset_descriptive": best_forecast[
            ["dataset", "model", "RMSE_mean", "result_type"]
        ].to_dict(orient="records"),
        "electricity_prophet_minus_xgboost_rmse": (
            prophet_rmse - xgb_rmse
        ),
        "electricity_interpretation": (
            "Prophet and XGBoost are numerically indistinguishable at the "
            "reported precision; Prophet is a single deterministic run, while "
            "XGBoost is summarized over ten seeds."
        ),
        "detector_scenario_winner_counts": winner_counts,
        "detector_macro_winners": macro_winners[
            [
                "dataset",
                "detector",
                "macro_F1",
                "macro_FalsePositiveRate",
            ]
        ].to_dict(orient="records"),
        "allowed_claims": [
            (
                "No forecasting model was best on all datasets."
            ),
            (
                "NFGA-LINEAR offered a compact accuracy-complexity trade-off "
                "and significantly improved over NFGA-Core in the primary "
                "matched RMSE analysis."
            ),
            (
                "After adding corrected Prophet results, NFGA-LINEAR was the "
                "best synthetic-anomaly detector in two of six scenarios and "
                "the best macro-F1 detector on Cholera and ILINet."
            ),
            (
                "Prophet achieved perfect detection on the small synthetic "
                "Electricity scenarios, but this does not establish broad "
                "real-world superiority."
            ),
        ],
        "forbidden_claims": [
            "NFGA-LINEAR was universally best in forecasting.",
            "NFGA-LINEAR was best in five of six anomaly scenarios.",
            "Prophet was significantly better than XGBoost on Electricity.",
            "Synthetic anomaly results prove real-world anomaly detection.",
            "Robust-Z proxy-label performance is independent validation.",
        ],
    }


def write_markdown_summary(
    path: Path,
    forecast: pd.DataFrame,
    detectors: pd.DataFrame,
    macro: pd.DataFrame,
    claims: dict,
) -> None:
    lines = [
        "# Final corrected result audit",
        "",
        "## Forecasting winners by RMSE",
        "",
    ]

    for dataset in sorted(forecast["dataset"].unique()):
        subset = forecast[forecast["dataset"] == dataset].sort_values(
            ["RMSE_rank", "model"]
        )
        lines.append(f"### {dataset}")
        for _, row in subset.iterrows():
            std = (
                ""
                if pd.isna(row["RMSE_std"])
                else f" ± {row['RMSE_std']:.6g}"
            )
            lines.append(
                f"- {int(row['RMSE_rank'])}. {row['model']}: "
                f"RMSE={row['RMSE_mean']:.6g}{std} "
                f"({row['result_type']})."
            )
        lines.append("")

    lines.extend(
        [
            "## Synthetic injected anomaly winners",
            "",
        ]
    )
    winners = detectors[detectors["F1_rank"] == 1]
    for _, row in winners.iterrows():
        lines.append(
            f"- {row['dataset']} / {row['scenario']}: "
            f"{row['detector']} (F1={row['F1_mean']:.4f}, "
            f"FPR={row['FalsePositiveRate_mean']:.4f})."
        )

    lines.extend(
        [
            "",
            "## Macro anomaly winners",
            "",
        ]
    )
    for _, row in macro[macro["macro_F1_rank"] == 1].iterrows():
        lines.append(
            f"- {row['dataset']}: {row['detector']} "
            f"(macro-F1={row['macro_F1']:.4f}, "
            f"macro-FPR={row['macro_FalsePositiveRate']:.4f})."
        )

    lines.extend(
        [
            "",
            "## Claim control",
            "",
        ]
    )
    for claim in claims["allowed_claims"]:
        lines.append(f"- Allowed: {claim}")
    for claim in claims["forbidden_claims"]:
        lines.append(f"- Do not claim: {claim}")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build final corrected forecast, detector, and resource tables."
    )
    parser.add_argument("--stage3-dir", required=True)
    parser.add_argument("--stage4-dir", required=True)
    parser.add_argument("--stage5-dir", required=True)
    parser.add_argument("--stage6-dir", required=True)
    parser.add_argument("--stage11-dir", required=True)
    parser.add_argument("--stage12-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    stage3 = Path(args.stage3_dir)
    stage4 = Path(args.stage4_dir)
    stage5 = Path(args.stage5_dir)
    stage6 = Path(args.stage6_dir)
    stage11 = Path(args.stage11_dir)
    stage12 = Path(args.stage12_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    forecast = load_forecast_results(
        stage3,
        stage4,
        stage5,
        stage6,
        stage12,
    )
    detectors, macro = load_detector_results(stage11, stage12)
    resources = build_resource_table(
        stage3,
        stage4,
        stage5,
        stage6,
        stage12,
    )
    claims = build_claim_audit(forecast, detectors, macro)

    forecast.to_csv(
        output / "final_forecast_ranking.csv",
        index=False,
    )
    detectors.to_csv(
        output / "final_detector_injected_scenarios.csv",
        index=False,
    )
    macro.to_csv(
        output / "final_detector_macro_results.csv",
        index=False,
    )
    resources.to_csv(
        output / "final_resource_costs.csv",
        index=False,
    )
    (
        output / "final_claim_audit.json"
    ).write_text(
        json.dumps(claims, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_markdown_summary(
        output / "final_result_audit.md",
        forecast,
        detectors,
        macro,
        claims,
    )

    print("\n===== Final corrected tables completed =====")
    print("\nForecast ranking:")
    print(forecast.to_string(index=False))

    print("\nInjected detector winners:")
    print(
        detectors[detectors["F1_rank"] == 1][
            [
                "dataset",
                "scenario",
                "detector",
                "F1_mean",
                "FalsePositiveRate_mean",
            ]
        ].to_string(index=False)
    )

    print("\nMacro detector winners:")
    print(
        macro[macro["macro_F1_rank"] == 1][
            [
                "dataset",
                "detector",
                "macro_F1",
                "macro_FalsePositiveRate",
            ]
        ].to_string(index=False)
    )

    print("\nClaim audit:")
    print(json.dumps(claims, indent=2, ensure_ascii=False))
    print("\nOutputs:", output)


if __name__ == "__main__":
    main()
