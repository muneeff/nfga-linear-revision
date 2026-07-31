from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


BASE_DIR = "/content/drive/MyDrive/Colab_Projects/Cholera_Forecasting/data/processed"
INPUT_DIR = os.path.join(BASE_DIR, "ready_for_models")
OUTPUT_DIR = os.path.join(BASE_DIR, "anomaly_redesign")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = ["cholera", "ilinet", "electricity"]
MASTER_SEED = 42

ROBUST_Z_THRESHOLD = 3.5
INJECTION_RATES = [0.05, 0.10]

REQUIRED_COLUMNS = {"ds", "year", "week", "y", "y_scaled"}


@dataclass(frozen=True)
class InjectionConfig:
    dataset: str
    nominal_rate: float
    seed: int
    target_count: int
    actual_count: int
    actual_rate: float
    perturbation_scale_from_training: float
    scaler_slope: float
    scaler_intercept: float


def validate_frame(df: pd.DataFrame, name: str) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError(f"{name} is empty.")

    if df["ds"].duplicated().any():
        raise ValueError(f"{name} contains duplicated timestamps.")

    numeric_columns = ["year", "week", "y", "y_scaled"]
    if df[numeric_columns].isna().any().any():
        raise ValueError(f"{name} contains missing numeric values.")

    if not np.isfinite(df[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains non-finite numeric values.")


def robust_location_scale(y: np.ndarray) -> tuple[float, float, float]:
    """
    Return median, raw MAD, and a robust scale estimate.

    robust_scale = 1.4826 * MAD approximates the standard deviation
    for normally distributed data. Standard deviation is used only
    as a fallback when MAD is zero.
    """
    values = np.asarray(y, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    robust_scale = 1.4826 * mad
    if not np.isfinite(robust_scale) or robust_scale <= 0:
        robust_scale = float(np.std(values, ddof=0))

    if not np.isfinite(robust_scale) or robust_scale <= 0:
        robust_scale = 1.0

    return median, mad, robust_scale


def modified_z_scores(
    y: np.ndarray,
    training_median: float,
    training_mad: float,
) -> np.ndarray:
    """
    Apply training-derived modified z-score parameters to another split.
    """
    values = np.asarray(y, dtype=float)

    if training_mad <= 0 or not np.isfinite(training_mad):
        return np.zeros(values.size, dtype=float)

    return 0.6745 * (values - training_median) / training_mad


def recover_affine_scaler(train: pd.DataFrame) -> tuple[float, float]:
    """
    Recover the affine mapping used by the prepared files:

        y_scaled ~= slope * y + intercept

    This avoids silently assuming a particular sklearn scaler while ensuring
    synthetic anomalies are transformed with the same training-derived mapping.
    """
    y = train["y"].to_numpy(dtype=float)
    z = train["y_scaled"].to_numpy(dtype=float)

    design = np.column_stack([y, np.ones_like(y)])
    slope, intercept = np.linalg.lstsq(design, z, rcond=None)[0]

    reconstructed = slope * y + intercept
    max_error = float(np.max(np.abs(reconstructed - z)))

    if not np.isfinite(slope) or abs(slope) < 1e-12:
        raise ValueError("Could not recover a valid y -> y_scaled mapping.")

    if max_error > 1e-5:
        raise ValueError(
            "y_scaled is not an affine transformation of y within tolerance; "
            f"maximum reconstruction error={max_error:.6g}. "
            "Inspect the upstream preprocessing script before continuing."
        )

    return float(slope), float(intercept)


def exact_target_count(n: int, rate: float) -> int:
    """
    Convert a nominal rate into an exact number of anomalous time points.
    Uses round-half-up behavior and always injects at least one point.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not 0 < rate < 1:
        raise ValueError("rate must be between 0 and 1.")

    return max(1, min(n, int(np.floor(n * rate + 0.5))))


def apply_single_point_anomaly(
    y_anom: np.ndarray,
    y_clean: np.ndarray,
    index: int,
    anomaly_type: str,
    scale: float,
    rng: np.random.Generator,
) -> None:
    if anomaly_type == "spike":
        delta = float(rng.uniform(3.0, 5.0) * scale)
        y_anom[index] = y_clean[index] + delta
        return

    if anomaly_type == "drop":
        delta = float(rng.uniform(2.5, 4.0) * scale)
        candidate = y_clean[index] - delta

        # Avoid a nominal anomaly whose clipped value is unchanged.
        if candidate < 0 and y_clean[index] <= 1e-12:
            y_anom[index] = y_clean[index] + delta
        else:
            y_anom[index] = max(0.0, candidate)
        return

    raise ValueError(f"Unsupported single-point anomaly type: {anomaly_type}")


def inject_anomalies_exact(
    df: pd.DataFrame,
    *,
    rate: float,
    seed: int,
    perturbation_scale: float,
    scaler_slope: float,
    scaler_intercept: float,
) -> tuple[pd.DataFrame, int]:
    """
    Inject exactly round(n * rate) anomalous time points.

    A level shift consumes two or three points from the target count; it does
    not add extra labels beyond the requested anomaly budget.
    """
    rng = np.random.default_rng(seed)
    result = df.copy()

    y_clean = result["y"].to_numpy(dtype=float)
    y_anom = y_clean.copy()
    n = y_clean.size
    target_count = exact_target_count(n, rate)

    labels = np.zeros(n, dtype=int)
    types = np.full(n, "normal", dtype=object)
    event_ids = np.full(n, -1, dtype=int)

    remaining = target_count
    event_id = 0

    # Include one short level-shift event only when the anomaly budget is large
    # enough, while preserving the exact requested number of anomalous points.
    if remaining >= 5 and n >= 3:
        shift_length = int(rng.integers(2, min(3, remaining) + 1))
        possible_starts = np.arange(0, n - shift_length + 1)
        start = int(rng.choice(possible_starts))
        selected = np.arange(start, start + shift_length)

        direction = float(rng.choice([-1.0, 1.0]))
        delta = direction * float(rng.uniform(2.0, 3.0) * perturbation_scale)
        shifted = y_clean[selected] + delta
        y_anom[selected] = np.maximum(0.0, shifted)

        # If clipping made a value unchanged, force a positive shift.
        unchanged = np.isclose(y_anom[selected], y_clean[selected])
        if unchanged.any():
            y_anom[selected[unchanged]] = (
                y_clean[selected[unchanged]]
                + float(rng.uniform(2.0, 3.0) * perturbation_scale)
            )

        labels[selected] = 1
        types[selected] = "level_shift"
        event_ids[selected] = event_id
        remaining -= shift_length
        event_id += 1

    available = np.flatnonzero(labels == 0)
    if remaining > available.size:
        raise RuntimeError("Not enough available points to complete injection.")

    selected_points = rng.choice(available, size=remaining, replace=False)

    for index in selected_points:
        anomaly_type = str(rng.choice(["spike", "drop"]))
        apply_single_point_anomaly(
            y_anom=y_anom,
            y_clean=y_clean,
            index=int(index),
            anomaly_type=anomaly_type,
            scale=perturbation_scale,
            rng=rng,
        )
        labels[index] = 1
        types[index] = anomaly_type
        event_ids[index] = event_id
        event_id += 1

    actual_count = int(labels.sum())
    if actual_count != target_count:
        raise AssertionError(
            f"Injection-count mismatch: target={target_count}, actual={actual_count}"
        )

    changed = ~np.isclose(y_anom, y_clean)
    if not np.array_equal(changed.astype(int), labels):
        bad = np.flatnonzero(changed.astype(int) != labels)
        raise AssertionError(
            f"Label/value mismatch at indices: {bad.tolist()}"
        )

    result["y_clean"] = y_clean
    result["y_anom"] = y_anom
    result["y_scaled_clean"] = result["y_scaled"].to_numpy(dtype=float)
    result["y_anom_scaled"] = scaler_slope * y_anom + scaler_intercept
    result["anomaly_injected"] = labels
    result["anomaly_type"] = types
    result["anomaly_event_id"] = event_ids
    result["anomaly_magnitude"] = y_anom - y_clean

    return result, target_count


def derived_seed(dataset_index: int, rate: float) -> int:
    return MASTER_SEED + dataset_index * 1000 + int(round(rate * 100))


def main() -> None:
    summary_rows: list[dict] = []
    metadata_rows: list[InjectionConfig] = []

    for dataset_index, name in enumerate(DATASETS):
        print(f"\n===== Redesign anomalies for {name.upper()} =====")

        train_path = os.path.join(INPUT_DIR, f"{name}_train_scaled.csv")
        test_path = os.path.join(INPUT_DIR, f"{name}_test_scaled.csv")

        if not os.path.exists(train_path):
            raise FileNotFoundError(train_path)
        if not os.path.exists(test_path):
            raise FileNotFoundError(test_path)

        train = pd.read_csv(train_path, parse_dates=["ds"])
        test = pd.read_csv(test_path, parse_dates=["ds"])

        validate_frame(train, f"{name} train")
        validate_frame(test, f"{name} test")

        train = train.sort_values("ds").reset_index(drop=True)
        test = test.sort_values("ds").reset_index(drop=True)

        train_median, train_mad, perturbation_scale = robust_location_scale(
            train["y"].to_numpy(dtype=float)
        )
        scaler_slope, scaler_intercept = recover_affine_scaler(train)

        test_robust_z = modified_z_scores(
            test["y"].to_numpy(dtype=float),
            training_median=train_median,
            training_mad=train_mad,
        )

        # These are proxy/reference labels, not externally verified ground truth.
        proxy_flags = (
            np.abs(test_robust_z) > ROBUST_Z_THRESHOLD
        ).astype(int)

        test_real = test.copy()
        test_real["anomaly_proxy_robust"] = proxy_flags

        # Compatibility alias for older downstream scripts.
        # Do not describe this column as verified "real anomalies" in the paper.
        test_real["anomaly_real_robust"] = proxy_flags
        test_real["robust_z"] = test_robust_z

        proxy_count = int(proxy_flags.sum())
        print(
            "Robust proxy flags in test "
            f"(not verified ground truth): {proxy_count} / {len(test_real)}"
        )

        test_real.to_csv(
            os.path.join(OUTPUT_DIR, f"{name}_test_real_anomalies.csv"),
            index=False,
        )

        for rate in INJECTION_RATES:
            seed = derived_seed(dataset_index, rate)

            injected, target_count = inject_anomalies_exact(
                test[["ds", "year", "week", "y", "y_scaled"]].copy(),
                rate=rate,
                seed=seed,
                perturbation_scale=perturbation_scale,
                scaler_slope=scaler_slope,
                scaler_intercept=scaler_intercept,
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                f"{name}_test_injected_{int(round(rate * 100))}pct.csv",
            )
            injected.to_csv(output_path, index=False)

            actual_count = int(injected["anomaly_injected"].sum())
            actual_rate = actual_count / len(injected)

            print(
                f"Injected anomalies {rate:.0%}: "
                f"{actual_count} / {len(injected)} "
                f"(actual={actual_rate:.6f}, seed={seed})"
            )

            summary_rows.append(
                {
                    "dataset": name,
                    "method": f"injected_{int(round(rate * 100))}pct",
                    "test_size": len(injected),
                    "nominal_rate": rate,
                    "target_count": target_count,
                    "n_anomalies": actual_count,
                    "actual_rate": actual_rate,
                    "seed": seed,
                }
            )

            metadata_rows.append(
                InjectionConfig(
                    dataset=name,
                    nominal_rate=rate,
                    seed=seed,
                    target_count=target_count,
                    actual_count=actual_count,
                    actual_rate=actual_rate,
                    perturbation_scale_from_training=perturbation_scale,
                    scaler_slope=scaler_slope,
                    scaler_intercept=scaler_intercept,
                )
            )

        summary_rows.append(
            {
                "dataset": name,
                "method": "proxy_robust_z",
                "test_size": len(test_real),
                "nominal_rate": np.nan,
                "target_count": np.nan,
                "n_anomalies": proxy_count,
                "actual_rate": proxy_count / len(test_real),
                "seed": np.nan,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "anomaly_redesign_summary.csv")
    summary.to_csv(summary_path, index=False)

    metadata = {
        "master_seed": MASTER_SEED,
        "robust_z_threshold": ROBUST_Z_THRESHOLD,
        "injection_rates": INJECTION_RATES,
        "notes": [
            "Synthetic anomaly counts exactly match the rounded target rate.",
            "Perturbation magnitudes use training-derived robust scale.",
            "y_anom_scaled uses the same affine scaler recovered from training data.",
            "Robust test flags are proxy/reference labels, not verified real ground truth.",
        ],
        "injections": [asdict(item) for item in metadata_rows],
    }
    metadata_path = os.path.join(OUTPUT_DIR, "anomaly_redesign_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print("\n===== Corrected anomaly redesign completed =====")
    print(summary.to_string(index=False))
    print("Saved summary to:", summary_path)
    print("Saved metadata to:", metadata_path)


if __name__ == "__main__":
    main()
