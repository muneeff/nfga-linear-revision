import os
import pandas as pd
import numpy as np

BASE_DIR = "/content/drive/MyDrive/Colab_Projects/Cholera_Forecasting/data/processed"

RAW_FILES = {
    "ARIMA_IF": "baseline_fair_results/baseline_fair_raw_results.csv",
    "NFGA-Core": "nfga_core_results/nfga_core_raw_results.csv",
    "NFGA-LINEAR": "nfga_linear_results/nfga_linear_raw_results.csv",
    "Prophet": "prophet_results/prophet_raw_results.csv",
    "XGBoost": "xgboost_results/xgboost_raw_results.csv",
    "LSTM": "lstm_results/lstm_raw_results.csv"
}

OUT_DIR = os.path.join(BASE_DIR, "final_tables")
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# Load all raw results
# --------------------------------------------------

all_frames = []

for source_name, rel_path in RAW_FILES.items():

    path = os.path.join(BASE_DIR, rel_path)

    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue

    df = pd.read_csv(path)

    df["source_file"] = source_name

    all_frames.append(df)

all_results = pd.concat(all_frames, ignore_index=True)

print("Loaded rows:", len(all_results))

# --------------------------------------------------
# Forecasting table
# --------------------------------------------------

forecast_df = all_results[
    all_results["scenario"] == "forecast_original"
].copy()

forecast_table = (
    forecast_df
    .groupby(
        ["dataset", "model"],
        as_index=False
    )
    .agg({
        "RMSE": ["mean", "std"],
        "MAE": ["mean", "std"],
        "sMAPE": ["mean", "std"],
        "R2": ["mean", "std"]
    })
)

forecast_table.columns = [
    "_".join(col).strip("_")
    for col in forecast_table.columns
]

forecast_table.to_csv(
    os.path.join(
        OUT_DIR,
        "forecast_table.csv"
    ),
    index=False
)

# --------------------------------------------------
# Anomaly table
# --------------------------------------------------

anomaly_df = all_results[
    all_results["scenario"] != "forecast_original"
].copy()

if len(anomaly_df) > 0:

    anomaly_table = (
        anomaly_df
        .groupby(
            ["dataset", "model", "scenario"],
            as_index=False
        )
        .agg({
            "Precision": ["mean", "std"],
            "Recall": ["mean", "std"],
            "F1": ["mean", "std"]
        })
    )

    anomaly_table.columns = [
        "_".join(col).strip("_")
        for col in anomaly_table.columns
    ]

    anomaly_table.to_csv(
        os.path.join(
            OUT_DIR,
            "anomaly_table.csv"
        ),
        index=False
    )

# --------------------------------------------------
# Ablation Study
# --------------------------------------------------

ablation_rows = []

for dataset in forecast_df["dataset"].unique():

    core = forecast_df[
        (forecast_df["dataset"] == dataset) &
        (forecast_df["model"].str.contains("Core", case=False, na=False))
    ]

    linear = forecast_df[
        (forecast_df["dataset"] == dataset) &
        (forecast_df["model"].str.contains("LINEAR", case=False, na=False))
    ]

    if len(core) == 0 or len(linear) == 0:
        continue

    core_rmse = core["RMSE"].mean()
    linear_rmse = linear["RMSE"].mean()

    improvement_pct = (
        (core_rmse - linear_rmse)
        / core_rmse
    ) * 100

    ablation_rows.append({
        "dataset": dataset,
        "NFGA_Core_RMSE": core_rmse,
        "NFGA_LINEAR_RMSE": linear_rmse,
        "Improvement_%": improvement_pct
    })

ablation_table = pd.DataFrame(ablation_rows)

ablation_table.to_csv(
    os.path.join(
        OUT_DIR,
        "ablation_table.csv"
    ),
    index=False
)

# --------------------------------------------------
# Ranking table
# --------------------------------------------------

ranking_rows = []

for dataset in forecast_df["dataset"].unique():

    tmp = (
        forecast_df[
            forecast_df["dataset"] == dataset
        ]
        .groupby("model")["RMSE"]
        .mean()
        .reset_index()
        .sort_values("RMSE")
    )

    if len(tmp) < 2:
        continue

    ranking_rows.append({
        "dataset": dataset,
        "best_model": tmp.iloc[0]["model"],
        "best_rmse": tmp.iloc[0]["RMSE"],
        "second_model": tmp.iloc[1]["model"],
        "second_rmse": tmp.iloc[1]["RMSE"]
    })

ranking_table = pd.DataFrame(ranking_rows)

ranking_table.to_csv(
    os.path.join(
        OUT_DIR,
        "ranking_table.csv"
    ),
    index=False
)

# --------------------------------------------------
# Global summary
# --------------------------------------------------

summary_rows = []

for model in forecast_df["model"].unique():

    sub = forecast_df[
        forecast_df["model"] == model
    ]

    summary_rows.append({
        "model": model,
        "mean_RMSE": sub["RMSE"].mean(),
        "mean_MAE": sub["MAE"].mean(),
        "mean_sMAPE": sub["sMAPE"].mean()
    })

summary_table = pd.DataFrame(summary_rows)

summary_table.to_csv(
    os.path.join(
        OUT_DIR,
        "global_summary_table.csv"
    ),
    index=False
)

print("\n===== FINAL TABLES CREATED =====")
print("forecast_table.csv")
print("anomaly_table.csv")
print("ablation_table.csv")
print("ranking_table.csv")
print("global_summary_table.csv")