from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DATASETS = ("cholera", "ilinet", "electricity")
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)
EPS = 1e-12


def load_nfga_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("nfga_stage6", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load NFGA implementation from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def round_window(value: float) -> int:
    return max(4, int(round(value)))


def build_train_windows(
    train_scaled: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(train_scaled, dtype=float)

    if len(values) <= window_size:
        raise ValueError(
            f"Training length {len(values)} must exceed window {window_size}"
        )

    X = np.stack(
        [
            values[index : index + window_size]
            for index in range(len(values) - window_size)
        ]
    )
    y = values[window_size:].copy()
    return X, y


def build_original_scenario(test: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": test["ds"],
            "y_clean": test["y"].to_numpy(dtype=float),
            "y_observed": test["y"].to_numpy(dtype=float),
            "y_observed_scaled": test["y_scaled"].to_numpy(dtype=float),
            "anomaly_truth": np.zeros(len(test), dtype=int),
        }
    )


def baseline_config(base_window: int) -> dict:
    return {
        "config_id": "baseline",
        "factor": "baseline",
        "level": "reference",
        "window_size": base_window,
        "K_min": 2,
        "K_max": 8,
        "population_size": 25,
        "maximum_generations": 40,
        "crossover_probability": 0.90,
        "center_mutation_probability": 0.25,
        "sigma_mutation_probability": 0.25,
        "consequent_mutation_probability": 0.30,
        "lambda_rules": 0.005,
        "lambda_weights": 1e-6,
    }


def sensitivity_configs(base_window: int) -> list[dict]:
    base = baseline_config(base_window)
    configs = [base]

    variants = [
        ("K_max", "4", {"K_max": 4}),
        ("K_max", "12", {"K_max": 12}),
        (
            "window_size",
            "half",
            {"window_size": round_window(base_window * 0.50)},
        ),
        (
            "window_size",
            "one_point_five",
            {"window_size": round_window(base_window * 1.50)},
        ),
        ("lambda_rules", "0", {"lambda_rules": 0.0}),
        ("lambda_rules", "0.02", {"lambda_rules": 0.02}),
        (
            "mutation_probability",
            "low_0.15",
            {
                "center_mutation_probability": 0.15,
                "sigma_mutation_probability": 0.15,
                "consequent_mutation_probability": 0.15,
            },
        ),
        (
            "mutation_probability",
            "high_0.45",
            {
                "center_mutation_probability": 0.45,
                "sigma_mutation_probability": 0.45,
                "consequent_mutation_probability": 0.45,
            },
        ),
        (
            "crossover_probability",
            "0.70",
            {"crossover_probability": 0.70},
        ),
        (
            "ga_budget",
            "small_15x20",
            {
                "population_size": 15,
                "maximum_generations": 20,
            },
        ),
        (
            "ga_budget",
            "large_40x60",
            {
                "population_size": 40,
                "maximum_generations": 60,
            },
        ),
    ]

    for factor, level, changes in variants:
        config = dict(base)
        config.update(changes)
        config["factor"] = factor
        config["level"] = level
        config["config_id"] = f"{factor}__{level}"
        configs.append(config)

    return configs


def apply_config(module: ModuleType, config: dict) -> None:
    module.K_MIN = int(config["K_min"])
    module.K_MAX = int(config["K_max"])
    module.POPULATION_SIZE = int(config["population_size"])
    module.MAX_GENERATIONS = int(config["maximum_generations"])
    module.CROSSOVER_PROBABILITY = float(
        config["crossover_probability"]
    )
    module.CENTER_MUTATION_PROBABILITY = float(
        config["center_mutation_probability"]
    )
    module.SIGMA_MUTATION_PROBABILITY = float(
        config["sigma_mutation_probability"]
    )
    module.CONSEQUENT_MUTATION_PROBABILITY = float(
        config["consequent_mutation_probability"]
    )
    module.LAMBDA_RULES = float(config["lambda_rules"])
    module.LAMBDA_WEIGHTS = float(config["lambda_weights"])


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)

    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p) - rank) * p[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(1.0, running_max)

    return adjusted.tolist()


def exact_wilcoxon(
    baseline: np.ndarray,
    alternative: np.ndarray,
) -> tuple[float, float, int]:
    baseline = np.asarray(baseline, dtype=float)
    alternative = np.asarray(alternative, dtype=float)
    difference = alternative - baseline
    nonzero = np.abs(difference) > EPS
    n_nonzero = int(nonzero.sum())

    if n_nonzero == 0:
        return 0.0, 1.0, 0

    method = (
        "exact"
        if n_nonzero == len(difference) and n_nonzero <= 25
        else "auto"
    )
    result = wilcoxon(
        alternative,
        baseline,
        alternative="two-sided",
        zero_method="wilcox",
        method=method,
    )
    return float(result.statistic), float(result.pvalue), n_nonzero


def paired_bootstrap_delta(
    baseline: np.ndarray,
    alternative: np.ndarray,
    *,
    seed: int,
    repetitions: int = 10000,
) -> tuple[float, float, float]:
    """
    Delta = alternative RMSE - baseline RMSE.
    Negative values favor the alternative configuration.
    """
    baseline = np.asarray(baseline, dtype=float)
    alternative = np.asarray(alternative, dtype=float)
    delta = alternative - baseline

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(delta),
        size=(repetitions, len(delta)),
    )
    means = delta[indices].mean(axis=1)

    return (
        float(np.mean(delta)),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def validate_baseline_against_stage6(
    sensitivity_raw: pd.DataFrame,
    stage6_raw_path: Path | None,
) -> pd.DataFrame:
    if stage6_raw_path is None:
        return pd.DataFrame()

    stage6 = pd.read_csv(stage6_raw_path)
    stage6 = stage6[
        (stage6["model"] == "NFGA-LINEAR")
        & (stage6["scenario"] == "forecast_original")
    ][["dataset", "seed", "RMSE"]].copy()
    stage6 = stage6.rename(columns={"RMSE": "stage6_RMSE"})

    baseline = sensitivity_raw[
        sensitivity_raw["config_id"] == "baseline"
    ][["dataset", "seed", "RMSE"]].copy()
    baseline = baseline.rename(columns={"RMSE": "sensitivity_RMSE"})

    merged = baseline.merge(
        stage6,
        on=["dataset", "seed"],
        how="outer",
        validate="one_to_one",
    )
    merged["absolute_difference"] = np.abs(
        merged["sensitivity_RMSE"] - merged["stage6_RMSE"]
    )
    merged["matches_within_1e-10"] = (
        merged["absolute_difference"] <= 1e-10
    )

    if not bool(merged["matches_within_1e-10"].all()):
        bad = merged[~merged["matches_within_1e-10"]]
        raise AssertionError(
            "Sensitivity baseline does not reproduce Stage 6 exactly:\n"
            + bad.to_string(index=False)
        )

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-factor-at-a-time sensitivity analysis for NFGA-LINEAR."
    )
    parser.add_argument("--ready-dir", required=True)
    parser.add_argument("--nfga-script", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--stage6-raw",
        default=None,
        help="Optional Stage 6 raw CSV for exact baseline reproducibility check",
    )
    args = parser.parse_args()

    ready_dir = Path(args.ready_dir)
    nfga_script = Path(args.nfga_script)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    module = load_nfga_module(nfga_script)
    raw_rows: list[dict] = []
    config_manifest: dict[str, list[dict]] = {}

    for dataset in DATASETS:
        print(f"\n================ SENSITIVITY: {dataset.upper()} ================")

        npz = np.load(
            ready_dir / f"{dataset}_windows.npz",
            allow_pickle=True,
        )
        base_window = int(npz["window_size"])
        train_mean = float(npz["train_mean"])
        train_std = float(npz["train_std"])

        train = (
            pd.read_csv(
                ready_dir / f"{dataset}_train_scaled.csv",
                parse_dates=["ds"],
            )
            .sort_values("ds")
            .reset_index(drop=True)
        )
        test = (
            pd.read_csv(
                ready_dir / f"{dataset}_test_scaled.csv",
                parse_dates=["ds"],
            )
            .sort_values("ds")
            .reset_index(drop=True)
        )

        train_scaled = train["y_scaled"].to_numpy(dtype=float)
        scenario = build_original_scenario(test)
        configs = sensitivity_configs(base_window)
        config_manifest[dataset] = configs

        for config_index, config in enumerate(configs):
            apply_config(module, config)
            window_size = int(config["window_size"])
            X_train, y_train = build_train_windows(
                train_scaled,
                window_size,
            )

            print(
                f"\n[{config_index + 1}/{len(configs)}] "
                f"{config['config_id']} | "
                f"window={window_size}, Kmax={config['K_max']}, "
                f"pop={config['population_size']}, "
                f"gen={config['maximum_generations']}"
            )

            for seed in SEEDS:
                individual, history, training_time = module.train_nfga(
                    X_train,
                    y_train,
                    seed=seed,
                    model_type="linear",
                )

                predictions, inference_latency = module.run_scenario(
                    individual,
                    model_type="linear",
                    train_scaled=train_scaled,
                    scenario=scenario,
                    window_size=window_size,
                    train_mean=train_mean,
                    train_std=train_std,
                    threshold=math.inf,
                )

                metrics = module.forecast_metrics(
                    predictions["y_clean"].to_numpy(dtype=float),
                    predictions["y_pred"].to_numpy(dtype=float),
                )

                raw_rows.append(
                    {
                        "dataset": dataset,
                        "config_id": config["config_id"],
                        "factor": config["factor"],
                        "level": config["level"],
                        "seed": seed,
                        "window_size": window_size,
                        "K_min": config["K_min"],
                        "K_max": config["K_max"],
                        "population_size": config["population_size"],
                        "maximum_generations": config["maximum_generations"],
                        "crossover_probability": (
                            config["crossover_probability"]
                        ),
                        "center_mutation_probability": (
                            config["center_mutation_probability"]
                        ),
                        "sigma_mutation_probability": (
                            config["sigma_mutation_probability"]
                        ),
                        "consequent_mutation_probability": (
                            config["consequent_mutation_probability"]
                        ),
                        "lambda_rules": config["lambda_rules"],
                        "lambda_weights": config["lambda_weights"],
                        "K_selected": int(individual["K"]),
                        "generations_ran": int(len(history)),
                        "training_time_sec": training_time,
                        "inference_latency_sec": inference_latency,
                        "model_size_bytes": module.model_size_bytes(individual),
                        **metrics,
                    }
                )

                print(
                    f"  seed={seed:>3} | "
                    f"K={individual['K']} | "
                    f"RMSE={metrics['RMSE']:.6g}"
                )

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(
        output_dir / "nfga_linear_sensitivity_raw.csv",
        index=False,
    )

    validation = validate_baseline_against_stage6(
        raw,
        Path(args.stage6_raw) if args.stage6_raw else None,
    )
    if not validation.empty:
        validation.to_csv(
            output_dir / "baseline_reproducibility_check.csv",
            index=False,
        )

    summary = (
        raw.groupby(
            ["dataset", "config_id", "factor", "level"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "count"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            sMAPE_mean=("sMAPE", "mean"),
            sMAPE_std=("sMAPE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            K_mean=("K_selected", "mean"),
            K_std=("K_selected", "std"),
            training_time_mean_sec=("training_time_sec", "mean"),
            inference_latency_mean_sec=("inference_latency_sec", "mean"),
            model_size_mean_bytes=("model_size_bytes", "mean"),
            generations_mean=("generations_ran", "mean"),
        )
    )

    baseline_summary = summary[
        summary["config_id"] == "baseline"
    ][
        [
            "dataset",
            "RMSE_mean",
            "MAE_mean",
            "sMAPE_mean",
            "training_time_mean_sec",
            "model_size_mean_bytes",
        ]
    ].rename(
        columns={
            "RMSE_mean": "baseline_RMSE_mean",
            "MAE_mean": "baseline_MAE_mean",
            "sMAPE_mean": "baseline_sMAPE_mean",
            "training_time_mean_sec": "baseline_training_time_mean_sec",
            "model_size_mean_bytes": "baseline_model_size_mean_bytes",
        }
    )

    summary = summary.merge(
        baseline_summary,
        on="dataset",
        how="left",
        validate="many_to_one",
    )
    summary["RMSE_change_percent_vs_baseline"] = (
        100.0
        * (summary["RMSE_mean"] - summary["baseline_RMSE_mean"])
        / np.maximum(np.abs(summary["baseline_RMSE_mean"]), EPS)
    )
    summary["training_time_change_percent_vs_baseline"] = (
        100.0
        * (
            summary["training_time_mean_sec"]
            - summary["baseline_training_time_mean_sec"]
        )
        / np.maximum(
            np.abs(summary["baseline_training_time_mean_sec"]),
            EPS,
        )
    )
    summary["model_size_change_percent_vs_baseline"] = (
        100.0
        * (
            summary["model_size_mean_bytes"]
            - summary["baseline_model_size_mean_bytes"]
        )
        / np.maximum(
            np.abs(summary["baseline_model_size_mean_bytes"]),
            EPS,
        )
    )
    summary.to_csv(
        output_dir / "nfga_linear_sensitivity_summary.csv",
        index=False,
    )

    inference_rows: list[dict] = []

    for dataset in DATASETS:
        dataset_raw = raw[raw["dataset"] == dataset]
        baseline = (
            dataset_raw[
                dataset_raw["config_id"] == "baseline"
            ]
            .set_index("seed")
            .sort_index()
        )

        local_rows: list[dict] = []
        local_p_values: list[float] = []

        for config_id in sorted(
            set(dataset_raw["config_id"]) - {"baseline"}
        ):
            alternative = (
                dataset_raw[
                    dataset_raw["config_id"] == config_id
                ]
                .set_index("seed")
                .sort_index()
            )

            if not baseline.index.equals(alternative.index):
                raise ValueError(
                    f"{dataset}/{config_id}: seed alignment failure"
                )

            statistic, p_value, n_nonzero = exact_wilcoxon(
                baseline["RMSE"].to_numpy(dtype=float),
                alternative["RMSE"].to_numpy(dtype=float),
            )
            delta_mean, delta_low, delta_high = paired_bootstrap_delta(
                baseline["RMSE"].to_numpy(dtype=float),
                alternative["RMSE"].to_numpy(dtype=float),
                seed=20260801 + len(local_rows),
            )

            first = alternative.iloc[0]
            local_rows.append(
                {
                    "dataset": dataset,
                    "config_id": config_id,
                    "factor": first["factor"],
                    "level": first["level"],
                    "n_pairs": len(baseline),
                    "baseline_RMSE_mean": float(
                        baseline["RMSE"].mean()
                    ),
                    "alternative_RMSE_mean": float(
                        alternative["RMSE"].mean()
                    ),
                    "mean_delta_alternative_minus_baseline": delta_mean,
                    "delta_ci95_low": delta_low,
                    "delta_ci95_high": delta_high,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_raw": p_value,
                    "n_nonzero_pairs": n_nonzero,
                }
            )
            local_p_values.append(p_value)

        adjusted = holm_adjust(local_p_values)
        for row, p_adjusted in zip(local_rows, adjusted):
            row["wilcoxon_p_holm_within_dataset"] = p_adjusted
            row["significant_holm_0_05"] = bool(p_adjusted < 0.05)
            inference_rows.append(row)

    inference = pd.DataFrame(inference_rows)
    inference.to_csv(
        output_dir / "nfga_linear_sensitivity_inference.csv",
        index=False,
    )

    metadata = {
        "analysis_type": "one_factor_at_a_time",
        "model": "NFGA-LINEAR",
        "seeds": list(SEEDS),
        "baseline": {
            "K_min": 2,
            "K_max": 8,
            "population_size": 25,
            "maximum_generations": 40,
            "crossover_probability": 0.90,
            "center_mutation_probability": 0.25,
            "sigma_mutation_probability": 0.25,
            "consequent_mutation_probability": 0.30,
            "lambda_rules": 0.005,
            "lambda_weights": 1e-6,
        },
        "important_limitation": (
            "One-factor-at-a-time analysis does not estimate interactions "
            "between hyperparameters."
        ),
        "configuration_manifest": config_manifest,
    }
    (output_dir / "sensitivity_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== Sensitivity analysis completed =====")
    print("\nSummary:")
    print(
        summary[
            [
                "dataset",
                "config_id",
                "RMSE_mean",
                "RMSE_std",
                "RMSE_change_percent_vs_baseline",
                "K_mean",
                "training_time_mean_sec",
                "model_size_mean_bytes",
            ]
        ].to_string(index=False)
    )
    print("\nInference:")
    print(inference.to_string(index=False))
    print("\nOutputs:", output_dir)


if __name__ == "__main__":
    main()
