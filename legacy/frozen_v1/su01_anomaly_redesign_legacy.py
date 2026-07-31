import os
import numpy as np
import pandas as pd

BASE_DIR = "/content/drive/MyDrive/Colab_Projects/Cholera_Forecasting/data/processed"
INPUT_DIR = os.path.join(BASE_DIR, "ready_for_models")
OUTPUT_DIR = os.path.join(BASE_DIR, "anomaly_redesign")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = ["cholera", "ilinet", "electricity"]
RANDOM_SEED = 42

ROBUST_Z_THRESHOLD = 3.5
INJECTION_RATES = [0.05, 0.10]


def robust_zscore_flags(y, threshold=3.5):
    """
    Robust anomaly detection using median and MAD.
    """
    y = np.asarray(y, dtype=float)
    median = np.median(y)
    mad = np.median(np.abs(y - median))

    if mad == 0:
        return np.zeros(len(y), dtype=int), median, mad

    robust_z = 0.6745 * (y - median) / mad
    flags = (np.abs(robust_z) > threshold).astype(int)
    return flags, median, mad


def inject_anomalies(df, rate=0.05, seed=42):
    """
    Inject synthetic anomalies into test data only.
    Types:
    - spike
    - drop
    - level shift
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    y = df["y"].values.astype(float)

    n = len(y)
    k = max(1, int(n * rate))

    anomaly_idx = rng.choice(np.arange(n), size=k, replace=False)

    df["y_clean"] = y.copy()
    df["y_anom"] = y.copy()
    df["anomaly_injected"] = 0
    df["anomaly_type"] = "normal"

    std = np.std(y)
    mean = np.mean(y)

    for idx in anomaly_idx:
        anomaly_type = rng.choice(["spike", "drop", "level_shift"])

        if anomaly_type == "spike":
            df.loc[df.index[idx], "y_anom"] = y[idx] + rng.uniform(2.5, 4.0) * std

        elif anomaly_type == "drop":
            df.loc[df.index[idx], "y_anom"] = max(0, y[idx] - rng.uniform(2.0, 3.5) * std)

        elif anomaly_type == "level_shift":
            shift_len = min(3, n - idx)
            df.loc[df.index[idx:idx + shift_len], "y_anom"] = y[idx:idx + shift_len] + rng.uniform(1.5, 2.5) * std
            df.loc[df.index[idx:idx + shift_len], "anomaly_injected"] = 1
            df.loc[df.index[idx:idx + shift_len], "anomaly_type"] = "level_shift"
            continue

        df.loc[df.index[idx], "anomaly_injected"] = 1
        df.loc[df.index[idx], "anomaly_type"] = anomaly_type

    return df


summary = []

for name in DATASETS:
    print(f"\n===== Redesign anomalies for {name.upper()} =====")

    train_path = os.path.join(INPUT_DIR, f"{name}_train_scaled.csv")
    test_path = os.path.join(INPUT_DIR, f"{name}_test_scaled.csv")

    train = pd.read_csv(train_path, parse_dates=["ds"])
    test = pd.read_csv(test_path, parse_dates=["ds"])

    train_flags, train_median, train_mad = robust_zscore_flags(
        train["y"].values,
        threshold=ROBUST_Z_THRESHOLD
    )

    # نحسب العتبة من التدريب، ثم نطبقها على الاختبار
    if train_mad == 0:
        test_robust_z = np.zeros(len(test))
    else:
        test_robust_z = 0.6745 * (test["y"].values - train_median) / train_mad

    test["anomaly_real_robust"] = (np.abs(test_robust_z) > ROBUST_Z_THRESHOLD).astype(int)
    test["robust_z"] = test_robust_z

    real_count = int(test["anomaly_real_robust"].sum())

    print(f"Real robust anomalies in test: {real_count} / {len(test)}")

    test.to_csv(
        os.path.join(OUTPUT_DIR, f"{name}_test_real_anomalies.csv"),
        index=False
    )

    for rate in INJECTION_RATES:
        injected = inject_anomalies(
            test[["ds", "year", "week", "y", "y_scaled"]].copy(),
            rate=rate,
            seed=RANDOM_SEED
        )

        out_path = os.path.join(
            OUTPUT_DIR,
            f"{name}_test_injected_{int(rate * 100)}pct.csv"
        )
        injected.to_csv(out_path, index=False)

        injected_count = int(injected["anomaly_injected"].sum())

        print(f"Injected anomalies {int(rate * 100)}%: {injected_count} / {len(injected)}")

        summary.append({
            "dataset": name,
            "method": f"injected_{int(rate * 100)}pct",
            "test_size": len(test),
            "n_anomalies": injected_count,
            "anomaly_rate": injected_count / len(test)
        })

    summary.append({
        "dataset": name,
        "method": "real_robust_z",
        "test_size": len(test),
        "n_anomalies": real_count,
        "anomaly_rate": real_count / len(test)
    })


summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "anomaly_redesign_summary.csv"), index=False)

print("\n===== Stage 2 completed =====")
print(summary_df)
print("Saved to:", OUTPUT_DIR)