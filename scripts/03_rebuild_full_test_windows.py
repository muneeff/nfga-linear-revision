from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("cholera", "ilinet", "electricity")
REQUIRED_COLUMNS = {"ds", "y", "y_scaled"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_frame(df: pd.DataFrame, label: str) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")

    if df.empty:
        raise ValueError(f"{label}: empty file")

    if df["ds"].duplicated().any():
        raise ValueError(f"{label}: duplicated timestamps")

    numeric = df[["y", "y_scaled"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label}: non-finite y/y_scaled values")


def build_train_windows(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= window:
        raise ValueError(
            f"Training length {len(values)} must be greater than window {window}"
        )

    X = np.stack([values[i : i + window] for i in range(len(values) - window)])
    y = values[window:].copy()
    return X.astype(np.float64), y.astype(np.float64)


def build_full_test_windows(
    train_values: np.ndarray,
    test_values: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-step-ahead evaluation over every test timestamp.

    For test step t, the input consists of the latest `window` observations
    available at that time: the training tail plus previously observed test
    values. The target is the current test value.
    """
    history = list(np.asarray(train_values, dtype=float))
    X_test: list[np.ndarray] = []
    y_test: list[float] = []

    for value in np.asarray(test_values, dtype=float):
        if len(history) < window:
            raise ValueError("Insufficient history for full test-window construction")

        X_test.append(np.asarray(history[-window:], dtype=float))
        y_test.append(float(value))

        # Walk-forward one-step-ahead protocol: the observed test value becomes
        # available before forecasting the next test timestamp.
        history.append(float(value))

    return np.stack(X_test), np.asarray(y_test, dtype=np.float64)


def check_scaling(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_mean: float,
    train_std: float,
    dataset: str,
) -> None:
    if not np.isfinite(train_std) or train_std <= 0:
        raise ValueError(f"{dataset}: invalid train_std={train_std}")

    train_expected = (train["y"].to_numpy(dtype=float) - train_mean) / train_std
    test_expected = (test["y"].to_numpy(dtype=float) - train_mean) / train_std

    train_err = float(
        np.max(np.abs(train_expected - train["y_scaled"].to_numpy(dtype=float)))
    )
    test_err = float(
        np.max(np.abs(test_expected - test["y_scaled"].to_numpy(dtype=float)))
    )

    tolerance = 1e-5
    if train_err > tolerance or test_err > tolerance:
        raise ValueError(
            f"{dataset}: y_scaled mismatch. "
            f"train max error={train_err:.6g}, test max error={test_err:.6g}. "
            "Inspect the upstream preprocessing before continuing."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild supervised windows with full test-horizon coverage."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing *_train_scaled.csv, *_test_scaled.csv, and old *_windows.npz",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for corrected CSV copies and *_windows.npz files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}

    for dataset in DATASETS:
        print(f"\n===== Rebuilding windows: {dataset.upper()} =====")

        train_path = input_dir / f"{dataset}_train_scaled.csv"
        test_path = input_dir / f"{dataset}_test_scaled.csv"
        old_npz_path = input_dir / f"{dataset}_windows.npz"

        for path in (train_path, test_path, old_npz_path):
            if not path.exists():
                raise FileNotFoundError(path)

        train = pd.read_csv(train_path, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)
        test = pd.read_csv(test_path, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)

        validate_frame(train, f"{dataset} train")
        validate_frame(test, f"{dataset} test")

        if train["ds"].max() >= test["ds"].min():
            raise ValueError(f"{dataset}: train/test chronology overlap")

        old = np.load(old_npz_path, allow_pickle=True)
        window_size = int(old["window_size"])
        train_mean = float(old["train_mean"])
        train_std = float(old["train_std"])

        check_scaling(train, test, train_mean, train_std, dataset)

        train_scaled = train["y_scaled"].to_numpy(dtype=float)
        test_scaled = test["y_scaled"].to_numpy(dtype=float)

        X_train, y_train = build_train_windows(train_scaled, window_size)
        X_test, y_test = build_full_test_windows(
            train_values=train_scaled,
            test_values=test_scaled,
            window=window_size,
        )

        if len(X_test) != len(test):
            raise AssertionError(
                f"{dataset}: expected {len(test)} test predictions, got {len(X_test)}"
            )

        if not np.allclose(y_test, test_scaled):
            raise AssertionError(f"{dataset}: y_test does not match all test rows")

        npz_out = output_dir / f"{dataset}_windows.npz"
        np.savez_compressed(
            npz_out,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            train_mean=np.asarray(train_mean),
            train_std=np.asarray(train_std),
            window_size=np.asarray(window_size),
            train_target_ds=train["ds"].iloc[window_size:].astype(str).to_numpy(),
            test_target_ds=test["ds"].astype(str).to_numpy(),
            protocol=np.asarray("full_test_walk_forward_context"),
        )

        train_out = output_dir / train_path.name
        test_out = output_dir / test_path.name
        train.to_csv(train_out, index=False)
        test.to_csv(test_out, index=False)

        manifest[dataset] = {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "window_size": window_size,
            "x_train_shape": list(X_train.shape),
            "y_train_shape": list(y_train.shape),
            "x_test_shape": list(X_test.shape),
            "y_test_shape": list(y_test.shape),
            "train_start": str(train["ds"].min()),
            "train_end": str(train["ds"].max()),
            "test_start": str(test["ds"].min()),
            "test_end": str(test["ds"].max()),
            "train_mean": train_mean,
            "train_std": train_std,
            "protocol": "full_test_walk_forward_context",
            "npz_sha256": sha256_file(npz_out),
        }

        print(
            f"window={window_size} | "
            f"X_train={X_train.shape} | X_test={X_test.shape} | "
            f"full test coverage={len(X_test)}/{len(test)}"
        )

    manifest_path = output_dir / "full_test_windows_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== Full-horizon window rebuild completed =====")
    print("Output directory:", output_dir)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
