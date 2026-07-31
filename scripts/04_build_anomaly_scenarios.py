from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("cholera", "ilinet", "electricity")
RATES = (0.05, 0.10)
MASTER_SEED = 42
ROBUST_Z_THRESHOLD = 3.5
REQUIRED_COLUMNS = {"ds", "year", "week", "y", "y_scaled"}


@dataclass(frozen=True)
class ScenarioMetadata:
    dataset: str
    scenario: str
    nominal_rate: float
    seed: int
    test_size: int
    target_count: int
    actual_count: int
    actual_rate: float
    perturbation_scale: float
    scaler_slope: float
    scaler_intercept: float


def validate_frame(df: pd.DataFrame, label: str) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")

    if df.empty:
        raise ValueError(f"{label}: empty dataframe")

    if df["ds"].duplicated().any():
        raise ValueError(f"{label}: duplicated timestamps")

    numeric = df[["year", "week", "y", "y_scaled"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label}: non-finite numeric values")


def robust_training_statistics(y: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(y, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    robust_scale = 1.4826 * mad
    if not np.isfinite(robust_scale) or robust_scale <= 0:
        robust_scale = float(np.std(values, ddof=0))

    if not np.isfinite(robust_scale) or robust_scale <= 0:
        robust_scale = 1.0

    return median, mad, robust_scale


def recover_affine_scaler(train: pd.DataFrame) -> tuple[float, float]:
    """
    Recover y_scaled = slope * y + intercept from training data.
    """
    y = train["y"].to_numpy(dtype=float)
    z = train["y_scaled"].to_numpy(dtype=float)

    design = np.column_stack([y, np.ones_like(y)])
    slope, intercept = np.linalg.lstsq(design, z, rcond=None)[0]

    reconstructed = slope * y + intercept
    max_error = float(np.max(np.abs(reconstructed - z)))

    if not np.isfinite(slope) or abs(slope) < 1e-12:
        raise ValueError("Invalid scaler slope")

    if max_error > 1e-5:
        raise ValueError(
            "y_scaled is not an affine transform of y within tolerance; "
            f"max reconstruction error={max_error:.6g}"
        )

    return float(slope), float(intercept)


def round_half_up(value: float) -> int:
    return int(np.floor(value + 0.5))


def target_count(n: int, rate: float) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 < rate < 1:
        raise ValueError("rate must be between 0 and 1")

    return max(1, min(n, round_half_up(n * rate)))


def derived_seed(dataset_index: int, rate: float) -> int:
    return MASTER_SEED + dataset_index * 1000 + int(round(rate * 100))


def apply_point_anomaly(
    y_clean: np.ndarray,
    y_anom: np.ndarray,
    idx: int,
    anomaly_type: str,
    scale: float,
    rng: np.random.Generator,
) -> None:
    if anomaly_type == "spike":
        y_anom[idx] = y_clean[idx] + float(rng.uniform(3.0, 5.0) * scale)
        return

    if anomaly_type == "drop":
        delta = float(rng.uniform(2.5, 4.0) * scale)
        candidate = y_clean[idx] - delta

        if candidate < 0 and y_clean[idx] <= 1e-12:
            y_anom[idx] = y_clean[idx] + delta
        else:
            y_anom[idx] = max(0.0, candidate)
        return

    raise ValueError(f"Unsupported anomaly type: {anomaly_type}")


def inject_exact_scenario(
    base: pd.DataFrame,
    *,
    rate: float,
    seed: int,
    perturbation_scale: float,
    scaler_slope: float,
    scaler_intercept: float,
) -> tuple[pd.DataFrame, int]:
    """
    Inject exactly round(test_size * rate) anomalous time points.

    A short level-shift event may be used when the anomaly budget is large
    enough, but it consumes points from the exact anomaly budget instead of
    creating extra unlabeled points.
    """
    rng = np.random.default_rng(seed)
    result = base.copy()

    y_clean = result["y"].to_numpy(dtype=float)
    y_anom = y_clean.copy()
    n = len(result)
    k = target_count(n, rate)

    labels = np.zeros(n, dtype=int)
    types = np.full(n, "normal", dtype=object)
    event_ids = np.full(n, -1, dtype=int)

    remaining = k
    event_id = 0

    # Include one short level shift when the budget is sufficiently large.
    if remaining >= 5 and n >= 3:
        shift_len = int(rng.integers(2, min(3, remaining) + 1))
        starts = np.arange(0, n - shift_len + 1)
        start = int(rng.choice(starts))
        idxs = np.arange(start, start + shift_len)

        direction = float(rng.choice([-1.0, 1.0]))
        delta = direction * float(rng.uniform(2.0, 3.0) * perturbation_scale)
        shifted = np.maximum(0.0, y_clean[idxs] + delta)

        unchanged = np.isclose(shifted, y_clean[idxs])
        if unchanged.any():
            shifted[unchanged] = (
                y_clean[idxs][unchanged]
                + float(rng.uniform(2.0, 3.0) * perturbation_scale)
            )

        y_anom[idxs] = shifted
        labels[idxs] = 1
        types[idxs] = "level_shift"
        event_ids[idxs] = event_id

        remaining -= shift_len
        event_id += 1

    available = np.flatnonzero(labels == 0)
    chosen = rng.choice(available, size=remaining, replace=False)

    for idx in chosen:
        anomaly_type = str(rng.choice(["spike", "drop"]))
        apply_point_anomaly(
            y_clean=y_clean,
            y_anom=y_anom,
            idx=int(idx),
            anomaly_type=anomaly_type,
            scale=perturbation_scale,
            rng=rng,
        )
        labels[idx] = 1
        types[idx] = anomaly_type
        event_ids[idx] = event_id
        event_id += 1

    actual = int(labels.sum())
    if actual != k:
        raise AssertionError(f"Expected {k} anomalous points, got {actual}")

    changed = (~np.isclose(y_anom, y_clean)).astype(int)
    if not np.array_equal(changed, labels):
        mismatches = np.flatnonzero(changed != labels)
        raise AssertionError(
            f"Value/label mismatch at indices {mismatches.tolist()}"
        )

    result["y_clean"] = y_clean
    result["y_anom"] = y_anom
    result["y_scaled_clean"] = result["y_scaled"].to_numpy(dtype=float)
    result["y_anom_scaled"] = scaler_slope * y_anom + scaler_intercept
    result["anomaly_injected"] = labels
    result["anomaly_type"] = types
    result["anomaly_event_id"] = event_ids
    result["anomaly_magnitude"] = y_anom - y_clean

    return result, k


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build corrected synthetic anomaly scenarios."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing corrected train/test CSV files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for corrected anomaly scenarios",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    metadata_rows: list[ScenarioMetadata] = []

    for dataset_index, dataset in enumerate(DATASETS):
        print(f"\n===== Building anomaly scenarios: {dataset.upper()} =====")

        train_path = input_dir / f"{dataset}_train_scaled.csv"
        test_path = input_dir / f"{dataset}_test_scaled.csv"

        if not train_path.exists():
            raise FileNotFoundError(train_path)
        if not test_path.exists():
            raise FileNotFoundError(test_path)

        train = (
            pd.read_csv(train_path, parse_dates=["ds"])
            .sort_values("ds")
            .reset_index(drop=True)
        )
        test = (
            pd.read_csv(test_path, parse_dates=["ds"])
            .sort_values("ds")
            .reset_index(drop=True)
        )

        validate_frame(train, f"{dataset} train")
        validate_frame(test, f"{dataset} test")

        train_median, train_mad, perturbation_scale = (
            robust_training_statistics(train["y"].to_numpy(dtype=float))
        )
        scaler_slope, scaler_intercept = recover_affine_scaler(train)

        if train_mad > 0:
            proxy_z = (
                0.6745
                * (test["y"].to_numpy(dtype=float) - train_median)
                / train_mad
            )
        else:
            proxy_z = np.zeros(len(test), dtype=float)

        proxy_labels = (np.abs(proxy_z) > ROBUST_Z_THRESHOLD).astype(int)

        proxy = test.copy()
        proxy["anomaly_proxy_robust"] = proxy_labels
        proxy["robust_z"] = proxy_z
        proxy_path = output_dir / f"{dataset}_test_proxy_robust_z.csv"
        proxy.to_csv(proxy_path, index=False)

        proxy_count = int(proxy_labels.sum())
        print(
            "Robust-z proxy labels "
            f"(not verified ground truth): {proxy_count}/{len(test)}"
        )

        summary_rows.append(
            {
                "dataset": dataset,
                "scenario": "proxy_robust_z",
                "test_size": len(test),
                "nominal_rate": np.nan,
                "target_count": np.nan,
                "n_anomalies": proxy_count,
                "actual_rate": proxy_count / len(test),
                "seed": np.nan,
            }
        )

        for rate in RATES:
            seed = derived_seed(dataset_index, rate)
            scenario = f"injected_{int(round(rate * 100))}pct"

            injected, expected_count = inject_exact_scenario(
                test[["ds", "year", "week", "y", "y_scaled"]].copy(),
                rate=rate,
                seed=seed,
                perturbation_scale=perturbation_scale,
                scaler_slope=scaler_slope,
                scaler_intercept=scaler_intercept,
            )

            actual_count = int(injected["anomaly_injected"].sum())
            actual_rate = actual_count / len(injected)

            out_path = output_dir / f"{dataset}_test_{scenario}.csv"
            injected.to_csv(out_path, index=False)

            print(
                f"{scenario}: {actual_count}/{len(injected)} "
                f"(actual rate={actual_rate:.6f}, seed={seed})"
            )

            summary_rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "test_size": len(injected),
                    "nominal_rate": rate,
                    "target_count": expected_count,
                    "n_anomalies": actual_count,
                    "actual_rate": actual_rate,
                    "seed": seed,
                }
            )

            metadata_rows.append(
                ScenarioMetadata(
                    dataset=dataset,
                    scenario=scenario,
                    nominal_rate=rate,
                    seed=seed,
                    test_size=len(injected),
                    target_count=expected_count,
                    actual_count=actual_count,
                    actual_rate=actual_rate,
                    perturbation_scale=perturbation_scale,
                    scaler_slope=scaler_slope,
                    scaler_intercept=scaler_intercept,
                )
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "anomaly_scenarios_summary.csv"
    summary.to_csv(summary_path, index=False)

    metadata = {
        "master_seed": MASTER_SEED,
        "rates": list(RATES),
        "robust_z_threshold": ROBUST_Z_THRESHOLD,
        "principles": [
            "Synthetic scenario counts exactly match round-half-up(test_size * nominal_rate).",
            "Perturbation scale is derived only from training data.",
            "Both raw and scaled anomalous targets are saved.",
            "Robust-z labels are exploratory proxy labels, not verified anomaly ground truth.",
        ],
        "scenarios": [asdict(item) for item in metadata_rows],
    }

    metadata_path = output_dir / "anomaly_scenarios_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== Corrected anomaly scenarios completed =====")
    print(summary.to_string(index=False))
    print("Summary:", summary_path)
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
