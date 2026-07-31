from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ("cholera", "ilinet", "electricity")
REQUIRED_CSV_COLUMNS = {"ds", "y", "y_scaled"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_csv(path: Path) -> dict:
    frame = pd.read_csv(path, parse_dates=["ds"])
    missing = REQUIRED_CSV_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path}: empty")
    if frame["ds"].duplicated().any():
        raise ValueError(f"{path}: duplicate timestamps")
    if not frame["ds"].is_monotonic_increasing:
        raise ValueError(f"{path}: timestamps are not sorted")
    values = frame[["y", "y_scaled"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite values")
    return {
        "rows": len(frame),
        "start": str(frame["ds"].min()),
        "end": str(frame["ds"].max()),
        "sha256": sha256(path),
    }


def validate_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    required = {"X_train", "y_train", "X_test", "y_test", "train_mean", "train_std", "window_size"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{path}: missing arrays {sorted(missing)}")
    x_train, y_train = data["X_train"], data["y_train"]
    x_test, y_test = data["X_test"], data["y_test"]
    if x_train.shape[0] != y_train.shape[0] or x_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"{path}: X/y length mismatch")
    for name in ("X_train", "y_train", "X_test", "y_test"):
        if not np.isfinite(data[name]).all():
            raise ValueError(f"{path}: non-finite values in {name}")
    return {
        "x_train_shape": list(x_train.shape),
        "x_test_shape": list(x_test.shape),
        "window_size": int(data["window_size"]),
        "train_mean": float(data["train_mean"]),
        "train_std": float(data["train_std"]),
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/logs/prepared_data_manifest.json"))
    args = parser.parse_args()

    manifest = {}
    for dataset in DATASETS:
        train = args.ready_dir / f"{dataset}_train_scaled.csv"
        test = args.ready_dir / f"{dataset}_test_scaled.csv"
        windows = args.ready_dir / f"{dataset}_windows.npz"
        for path in (train, test, windows):
            if not path.exists():
                raise FileNotFoundError(path)
        manifest[dataset] = {
            "train": validate_csv(train),
            "test": validate_csv(test),
            "windows": validate_npz(windows),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
