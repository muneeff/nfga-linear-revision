from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import time
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

warnings.filterwarnings("ignore")


DATASETS = ("cholera", "ilinet", "electricity")
MODEL_TYPES: tuple[Literal["core", "linear"], ...] = ("core", "linear")
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)
CALIBRATION_SEED = 20260731

K_MIN = 2
K_MAX = 8
POPULATION_SIZE = 25
MAX_GENERATIONS = 40
TOURNAMENT_SIZE = 3
ELITE_COUNT = 2
NO_IMPROVEMENT_PATIENCE = 10
MIN_IMPROVEMENT = 1e-6

CROSSOVER_PROBABILITY = 0.90
CENTER_MUTATION_PROBABILITY = 0.25
SIGMA_MUTATION_PROBABILITY = 0.25
CONSEQUENT_MUTATION_PROBABILITY = 0.30
RULE_INSERTION_PROBABILITY = 0.10
RULE_DELETION_PROBABILITY = 0.08

CENTER_MUTATION_STD = 0.05
CONSEQUENT_MUTATION_STD = 0.05
SIGMA_LOGNORMAL_STD = 0.12
SIGMA_MIN = 0.05
SIGMA_MAX = 5.0

LAMBDA_RULES = 0.005
LAMBDA_WEIGHTS = 1e-6
ROBUST_K = 3.5
EPS = 1e-12


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("_").lower()


def model_display_name(model_type: str) -> str:
    if model_type == "core":
        return "NFGA-Core"
    if model_type == "linear":
        return "NFGA-LINEAR"
    raise ValueError(f"Unknown model type: {model_type}")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    terms = np.zeros_like(y_true, dtype=float)
    mask = denominator > EPS
    terms[mask] = np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
    return float(100.0 * np.mean(terms))


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Forecast metric shape mismatch: {y_true.shape} vs {y_pred.shape}"
        )

    result = {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "sMAPE": smape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }

    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    numerator = float(np.sum((y_true - y_pred) ** 2))
    r2_manual = np.nan if denominator <= EPS else 1.0 - numerator / denominator

    if np.isfinite(r2_manual) and not np.isclose(
        result["R2"], r2_manual, rtol=1e-10, atol=1e-10
    ):
        raise AssertionError(
            f"R² consistency failure: sklearn={result['R2']}, manual={r2_manual}"
        )

    result["R2_manual"] = float(r2_manual)
    return result


def anomaly_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float, float, str]:
    truth = np.asarray(truth, dtype=int)
    predicted = np.asarray(predicted, dtype=int)

    if truth.shape != predicted.shape:
        raise ValueError(
            f"Anomaly metric shape mismatch: {truth.shape} vs {predicted.shape}"
        )

    if int(truth.sum()) == 0:
        return np.nan, np.nan, np.nan, "not_applicable_no_positive_reference"

    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        predicted,
        average="binary",
        zero_division=0,
    )
    return float(precision), float(recall), float(f1), "applicable"


def robust_residual_threshold(
    residuals: np.ndarray,
    *,
    k: float = ROBUST_K,
) -> tuple[float, float, float, str]:
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    if residuals.size < 3:
        raise ValueError("At least three finite calibration residuals are required.")

    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    robust_scale = 1.4826 * mad

    if not np.isfinite(robust_scale) or robust_scale <= EPS:
        std = float(np.std(residuals, ddof=0))
        threshold = median + 3.0 * std
        method = "median_plus_3std_fallback"
    else:
        threshold = median + k * robust_scale
        method = "median_plus_k_scaled_mad"

    if not np.isfinite(threshold):
        raise ValueError("Non-finite residual threshold.")

    return float(threshold), median, mad, method


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    validation_size = max(20, int(round(0.20 * n)))
    validation_size = min(validation_size, n - 20)

    if validation_size <= 0:
        raise ValueError(f"Not enough samples for chronological validation: n={n}")

    split = n - validation_size
    return X[:split], y[:split], X[split:], y[split:]


def gaussian_activations(
    X: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)

    squared_distance = np.mean(
        (X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    sigma_squared = np.maximum(sigmas**2, EPS)
    return np.exp(
        -squared_distance / (2.0 * sigma_squared[np.newaxis, :])
    )


def predict_scaled(
    X: np.ndarray,
    individual: dict,
    model_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = gaussian_activations(
        X,
        individual["centers"],
        individual["sigmas"],
    )
    denominator = np.sum(mu, axis=1, keepdims=True) + EPS
    normalized = mu / denominator

    if model_type == "core":
        local_outputs = np.broadcast_to(
            individual["consequents"][:, 0],
            (len(X), individual["K"]),
        )
    elif model_type == "linear":
        features = np.column_stack(
            [np.asarray(X, dtype=float), np.ones(len(X), dtype=float)]
        )
        local_outputs = features @ individual["consequents"].T
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    prediction = np.sum(normalized * local_outputs, axis=1)
    max_membership = np.max(normalized, axis=1)
    mean_membership = np.mean(normalized, axis=1)
    return prediction, max_membership, mean_membership


def initial_consequent(
    rng: np.random.Generator,
    *,
    model_type: str,
    window_size: int,
    y_reference: float,
) -> np.ndarray:
    if model_type == "core":
        return np.asarray([y_reference + rng.normal(0.0, 0.05)], dtype=float)

    coefficients = np.zeros(window_size + 1, dtype=float)
    coefficients[-2] = rng.normal(1.0, 0.08)  # persistence initialization
    coefficients[-1] = rng.normal(0.0, 0.05)  # intercept
    return coefficients


def initialize_individual(
    rng: np.random.Generator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
) -> dict:
    n_samples, window_size = X_train.shape
    K = int(rng.integers(K_MIN, K_MAX + 1))
    chosen = rng.choice(n_samples, size=K, replace=False)

    centers = X_train[chosen].copy()
    sigmas = rng.uniform(0.25, 2.0, size=K)

    consequents = np.vstack(
        [
            initial_consequent(
                rng,
                model_type=model_type,
                window_size=window_size,
                y_reference=float(y_train[index]),
            )
            for index in chosen
        ]
    )

    return {
        "K": K,
        "centers": centers,
        "sigmas": sigmas,
        "consequents": consequents,
    }


def clone_individual(individual: dict) -> dict:
    return {
        "K": int(individual["K"]),
        "centers": individual["centers"].copy(),
        "sigmas": individual["sigmas"].copy(),
        "consequents": individual["consequents"].copy(),
    }


def objective(
    individual: dict,
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
) -> float:
    prediction, _, _ = predict_scaled(X, individual, model_type)
    error = rmse(y, prediction)
    rule_penalty = LAMBDA_RULES * (individual["K"] / K_MAX)
    weight_penalty = LAMBDA_WEIGHTS * float(
        np.mean(individual["consequents"] ** 2)
    )
    return float(error + rule_penalty + weight_penalty)


def tournament_select(
    rng: np.random.Generator,
    population: list[dict],
    fitnesses: np.ndarray,
) -> dict:
    indices = rng.choice(
        len(population),
        size=TOURNAMENT_SIZE,
        replace=False,
    )
    best_index = int(indices[np.argmin(fitnesses[indices])])
    return clone_individual(population[best_index])


def crossover(
    rng: np.random.Generator,
    parent_a: dict,
    parent_b: dict,
) -> dict:
    if rng.random() > CROSSOVER_PROBABILITY:
        return clone_individual(
            parent_a if rng.random() < 0.5 else parent_b
        )

    choose_a = rng.choice(
        parent_a["K"],
        size=max(1, int(rng.integers(1, parent_a["K"] + 1))),
        replace=False,
    )
    choose_b = rng.choice(
        parent_b["K"],
        size=max(1, int(rng.integers(1, parent_b["K"] + 1))),
        replace=False,
    )

    centers = np.vstack(
        [parent_a["centers"][choose_a], parent_b["centers"][choose_b]]
    )
    sigmas = np.concatenate(
        [parent_a["sigmas"][choose_a], parent_b["sigmas"][choose_b]]
    )
    consequents = np.vstack(
        [
            parent_a["consequents"][choose_a],
            parent_b["consequents"][choose_b],
        ]
    )

    if len(sigmas) > K_MAX:
        keep = rng.choice(len(sigmas), size=K_MAX, replace=False)
        centers = centers[keep]
        sigmas = sigmas[keep]
        consequents = consequents[keep]

    if len(sigmas) < K_MIN:
        return clone_individual(parent_a)

    return {
        "K": int(len(sigmas)),
        "centers": centers,
        "sigmas": sigmas,
        "consequents": consequents,
    }


def mutate(
    rng: np.random.Generator,
    individual: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
) -> dict:
    mutated = clone_individual(individual)
    window_size = X_train.shape[1]

    if rng.random() < CENTER_MUTATION_PROBABILITY:
        mutated["centers"] += rng.normal(
            0.0,
            CENTER_MUTATION_STD,
            size=mutated["centers"].shape,
        )

    if rng.random() < SIGMA_MUTATION_PROBABILITY:
        mutated["sigmas"] *= rng.lognormal(
            mean=0.0,
            sigma=SIGMA_LOGNORMAL_STD,
            size=mutated["K"],
        )
        mutated["sigmas"] = np.clip(
            mutated["sigmas"],
            SIGMA_MIN,
            SIGMA_MAX,
        )

    if rng.random() < CONSEQUENT_MUTATION_PROBABILITY:
        mutated["consequents"] += rng.normal(
            0.0,
            CONSEQUENT_MUTATION_STD,
            size=mutated["consequents"].shape,
        )

    if (
        rng.random() < RULE_INSERTION_PROBABILITY
        and mutated["K"] < K_MAX
    ):
        index = int(rng.integers(0, len(X_train)))
        new_center = X_train[index].reshape(1, -1)
        new_sigma = np.asarray([rng.uniform(0.25, 2.0)], dtype=float)
        new_consequent = initial_consequent(
            rng,
            model_type=model_type,
            window_size=window_size,
            y_reference=float(y_train[index]),
        ).reshape(1, -1)

        mutated["centers"] = np.vstack(
            [mutated["centers"], new_center]
        )
        mutated["sigmas"] = np.concatenate(
            [mutated["sigmas"], new_sigma]
        )
        mutated["consequents"] = np.vstack(
            [mutated["consequents"], new_consequent]
        )
        mutated["K"] += 1

    if (
        rng.random() < RULE_DELETION_PROBABILITY
        and mutated["K"] > K_MIN
    ):
        index = int(rng.integers(0, mutated["K"]))
        mutated["centers"] = np.delete(
            mutated["centers"],
            index,
            axis=0,
        )
        mutated["sigmas"] = np.delete(
            mutated["sigmas"],
            index,
            axis=0,
        )
        mutated["consequents"] = np.delete(
            mutated["consequents"],
            index,
            axis=0,
        )
        mutated["K"] -= 1

    return mutated


def model_size_bytes(individual: dict) -> int:
    return int(
        individual["centers"].nbytes
        + individual["sigmas"].nbytes
        + individual["consequents"].nbytes
    )


def train_nfga(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    model_type: str,
) -> tuple[dict, pd.DataFrame, float]:
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    population = [
        initialize_individual(
            rng,
            X_train,
            y_train,
            model_type,
        )
        for _ in range(POPULATION_SIZE)
    ]

    history_rows: list[dict] = []
    global_best: dict | None = None
    global_best_fitness = np.inf
    generations_without_improvement = 0

    started = time.perf_counter()

    for generation in range(MAX_GENERATIONS):
        fitnesses = np.asarray(
            [
                objective(individual, X_train, y_train, model_type)
                for individual in population
            ],
            dtype=float,
        )

        order = np.argsort(fitnesses)
        generation_best_index = int(order[0])
        generation_best_fitness = float(fitnesses[generation_best_index])
        generation_best = population[generation_best_index]

        if generation_best_fitness < (
            global_best_fitness - MIN_IMPROVEMENT
        ):
            global_best = clone_individual(generation_best)
            global_best_fitness = generation_best_fitness
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        history_rows.append(
            {
                "generation": generation,
                "generation_best_fitness": generation_best_fitness,
                "global_best_fitness": global_best_fitness,
                "generation_mean_fitness": float(np.mean(fitnesses)),
                "generation_std_fitness": float(np.std(fitnesses)),
                "best_K": int(generation_best["K"]),
            }
        )

        if generations_without_improvement >= NO_IMPROVEMENT_PATIENCE:
            break

        new_population = [
            clone_individual(population[int(index)])
            for index in order[:ELITE_COUNT]
        ]

        while len(new_population) < POPULATION_SIZE:
            parent_a = tournament_select(rng, population, fitnesses)
            parent_b = tournament_select(rng, population, fitnesses)
            child = crossover(rng, parent_a, parent_b)
            child = mutate(
                rng,
                child,
                X_train,
                y_train,
                model_type,
            )
            new_population.append(child)

        population = new_population

    training_time = time.perf_counter() - started

    if global_best is None:
        raise RuntimeError("GA failed to produce a best individual.")

    return global_best, pd.DataFrame(history_rows), training_time


def calibrate_threshold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    train_std: float,
    model_type: str,
) -> tuple[float, float, float, str, dict]:
    X_fit, y_fit, X_val, y_val = chronological_split(X_train, y_train)

    model, history, calibration_time = train_nfga(
        X_fit,
        y_fit,
        seed=CALIBRATION_SEED,
        model_type=model_type,
    )
    prediction, _, _ = predict_scaled(X_val, model, model_type)
    residuals_original = np.abs(y_val - prediction) * train_std

    threshold, median, mad, method = robust_residual_threshold(
        residuals_original
    )

    metadata = {
        "calibration_seed": CALIBRATION_SEED,
        "n_fit": len(X_fit),
        "n_validation": len(X_val),
        "selected_K": int(model["K"]),
        "calibration_time_sec": calibration_time,
        "calibration_generations": int(len(history)),
    }
    return threshold, median, mad, method, metadata


def load_scenarios(
    dataset: str,
    ready_dir: Path,
    anomaly_dir: Path,
    *,
    train_mean: float,
    train_std: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    test = (
        pd.read_csv(
            ready_dir / f"{dataset}_test_scaled.csv",
            parse_dates=["ds"],
        )
        .sort_values("ds")
        .reset_index(drop=True)
    )

    scenarios: dict[str, pd.DataFrame] = {
        "forecast_original": pd.DataFrame(
            {
                "ds": test["ds"],
                "y_clean": test["y"].to_numpy(dtype=float),
                "y_observed": test["y"].to_numpy(dtype=float),
                "y_observed_scaled": test["y_scaled"].to_numpy(dtype=float),
                "anomaly_truth": np.zeros(len(test), dtype=int),
            }
        )
    }

    for rate in (5, 10):
        frame = (
            pd.read_csv(
                anomaly_dir / f"{dataset}_test_injected_{rate}pct.csv",
                parse_dates=["ds"],
            )
            .sort_values("ds")
            .reset_index(drop=True)
        )

        if not np.array_equal(
            frame["ds"].to_numpy(dtype="datetime64[ns]"),
            test["ds"].to_numpy(dtype="datetime64[ns]"),
        ):
            raise ValueError(f"{dataset} injected_{rate}pct: timestamp mismatch")

        expected_scaled = (
            frame["y_anom"].to_numpy(dtype=float) - train_mean
        ) / train_std
        max_scaling_error = float(
            np.max(
                np.abs(
                    expected_scaled
                    - frame["y_anom_scaled"].to_numpy(dtype=float)
                )
            )
        )
        if max_scaling_error > 1e-5:
            raise ValueError(
                f"{dataset} injected_{rate}pct: y_anom_scaled mismatch "
                f"(max error={max_scaling_error:.6g})"
            )

        scenarios[f"injected_{rate}pct"] = pd.DataFrame(
            {
                "ds": frame["ds"],
                "y_clean": frame["y_clean"].to_numpy(dtype=float),
                "y_observed": frame["y_anom"].to_numpy(dtype=float),
                "y_observed_scaled": frame["y_anom_scaled"].to_numpy(dtype=float),
                "anomaly_truth": frame["anomaly_injected"].to_numpy(dtype=int),
            }
        )

    proxy = (
        pd.read_csv(
            anomaly_dir / f"{dataset}_test_proxy_robust_z.csv",
            parse_dates=["ds"],
        )
        .sort_values("ds")
        .reset_index(drop=True)
    )

    if not np.array_equal(
        proxy["ds"].to_numpy(dtype="datetime64[ns]"),
        test["ds"].to_numpy(dtype="datetime64[ns]"),
    ):
        raise ValueError(f"{dataset} proxy_robust_z: timestamp mismatch")

    scenarios["proxy_robust_z"] = pd.DataFrame(
        {
            "ds": proxy["ds"],
            "y_clean": proxy["y"].to_numpy(dtype=float),
            "y_observed": proxy["y"].to_numpy(dtype=float),
            "y_observed_scaled": proxy["y_scaled"].to_numpy(dtype=float),
            "anomaly_truth": proxy["anomaly_proxy_robust"].to_numpy(dtype=int),
        }
    )

    return test, scenarios


def run_scenario(
    individual: dict,
    *,
    model_type: str,
    train_scaled: np.ndarray,
    scenario: pd.DataFrame,
    window_size: int,
    train_mean: float,
    train_std: float,
    threshold: float,
) -> tuple[pd.DataFrame, float]:
    history = list(np.asarray(train_scaled, dtype=float))
    predictions_scaled: list[float] = []
    max_memberships: list[float] = []
    mean_memberships: list[float] = []
    latencies: list[float] = []

    for observed_scaled in scenario["y_observed_scaled"].to_numpy(dtype=float):
        features = np.asarray(
            history[-window_size:],
            dtype=float,
        ).reshape(1, window_size)

        started = time.perf_counter()
        prediction, max_mu, mean_mu = predict_scaled(
            features,
            individual,
            model_type,
        )
        latencies.append(time.perf_counter() - started)

        predictions_scaled.append(float(prediction[0]))
        max_memberships.append(float(max_mu[0]))
        mean_memberships.append(float(mean_mu[0]))
        history.append(float(observed_scaled))

    output = scenario.copy()
    output["y_pred_scaled"] = np.asarray(predictions_scaled, dtype=float)
    output["y_pred"] = output["y_pred_scaled"] * train_std + train_mean
    output["residual_abs"] = np.abs(
        output["y_observed"].to_numpy(dtype=float)
        - output["y_pred"].to_numpy(dtype=float)
    )
    output["anomaly_pred"] = (
        output["residual_abs"].to_numpy(dtype=float) > threshold
    ).astype(int)
    output["threshold"] = threshold
    output["max_membership"] = max_memberships
    output["mean_membership"] = mean_memberships

    return output, float(np.mean(latencies))


def save_model(
    path: Path,
    individual: dict,
    *,
    dataset: str,
    model_type: str,
    seed: int,
    window_size: int,
    train_mean: float,
    train_std: float,
) -> None:
    np.savez_compressed(
        path,
        dataset=np.asarray(dataset),
        model_type=np.asarray(model_type),
        seed=np.asarray(seed),
        K=np.asarray(individual["K"]),
        centers=individual["centers"],
        sigmas=individual["sigmas"],
        consequents=individual["consequents"],
        window_size=np.asarray(window_size),
        train_mean=np.asarray(train_mean),
        train_std=np.asarray(train_std),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run matched NFGA-Core and NFGA-LINEAR experiments."
    )
    parser.add_argument("--ready-dir", required=True)
    parser.add_argument("--anomaly-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    ready_dir = Path(args.ready_dir)
    anomaly_dir = Path(args.anomaly_dir)
    output_dir = Path(args.output_dir)
    predictions_dir = output_dir / "predictions"
    histories_dir = output_dir / "ga_histories"
    models_dir = output_dir / "models"

    for directory in (
        output_dir,
        predictions_dir,
        histories_dir,
        models_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    threshold_rows: list[dict] = []
    calibration_metadata: dict[str, dict] = {}

    for dataset in DATASETS:
        print(f"\n================ DATASET: {dataset.upper()} ================")

        npz = np.load(
            ready_dir / f"{dataset}_windows.npz",
            allow_pickle=True,
        )
        X_train = np.asarray(npz["X_train"], dtype=float)
        y_train = np.asarray(npz["y_train"], dtype=float)
        window_size = int(npz["window_size"])
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
        train_scaled = train["y_scaled"].to_numpy(dtype=float)

        test, scenarios = load_scenarios(
            dataset,
            ready_dir,
            anomaly_dir,
            train_mean=train_mean,
            train_std=train_std,
        )

        for model_type in MODEL_TYPES:
            display_name = model_display_name(model_type)
            print(f"\n---------------- {display_name} ----------------")

            (
                threshold,
                calibration_median,
                calibration_mad,
                threshold_method,
                threshold_metadata,
            ) = calibrate_threshold(
                X_train,
                y_train,
                train_std=train_std,
                model_type=model_type,
            )

            calibration_metadata[f"{dataset}:{display_name}"] = (
                threshold_metadata
            )

            threshold_rows.append(
                {
                    "dataset": dataset,
                    "model": display_name,
                    "threshold": threshold,
                    "calibration_residual_median": calibration_median,
                    "calibration_residual_mad": calibration_mad,
                    "threshold_method": threshold_method,
                    **threshold_metadata,
                }
            )

            print(
                f"Calibrated residual threshold: {threshold:.6g} | "
                f"calibration K={threshold_metadata['selected_K']}"
            )

            for seed in SEEDS:
                print(f"\n--- seed={seed} ---")

                individual, ga_history, training_time = train_nfga(
                    X_train,
                    y_train,
                    seed=seed,
                    model_type=model_type,
                )

                ga_history.to_csv(
                    histories_dir
                    / f"{dataset}_{slugify(display_name)}_seed_{seed}.csv",
                    index=False,
                )

                save_model(
                    models_dir
                    / f"{dataset}_{slugify(display_name)}_seed_{seed}.npz",
                    individual,
                    dataset=dataset,
                    model_type=model_type,
                    seed=seed,
                    window_size=window_size,
                    train_mean=train_mean,
                    train_std=train_std,
                )

                size_bytes = model_size_bytes(individual)

                for scenario_name, scenario in scenarios.items():
                    predictions, mean_latency = run_scenario(
                        individual,
                        model_type=model_type,
                        train_scaled=train_scaled,
                        scenario=scenario,
                        window_size=window_size,
                        train_mean=train_mean,
                        train_std=train_std,
                        threshold=threshold,
                    )

                    if len(predictions) != len(test):
                        raise AssertionError(
                            f"{dataset}/{display_name}/{seed}/{scenario_name}: "
                            f"expected {len(test)} rows, got {len(predictions)}"
                        )

                    predictions.to_csv(
                        predictions_dir
                        / (
                            f"{dataset}_{slugify(display_name)}_"
                            f"seed_{seed}_{scenario_name}.csv"
                        ),
                        index=False,
                    )

                    base_row = {
                        "dataset": dataset,
                        "model": display_name,
                        "model_type": model_type,
                        "seed": seed,
                        "scenario": scenario_name,
                        "n_eval": len(predictions),
                        "K": int(individual["K"]),
                        "training_time_sec": training_time,
                        "mean_inference_latency_sec": mean_latency,
                        "model_size_bytes": size_bytes,
                        "generations_ran": len(ga_history),
                        "final_objective": float(
                            ga_history["global_best_fitness"].iloc[-1]
                        ),
                        "threshold": threshold,
                        "mean_max_membership": float(
                            predictions["max_membership"].mean()
                        ),
                        "mean_membership": float(
                            predictions["mean_membership"].mean()
                        ),
                    }

                    if scenario_name == "forecast_original":
                        metrics = forecast_metrics(
                            predictions["y_clean"].to_numpy(dtype=float),
                            predictions["y_pred"].to_numpy(dtype=float),
                        )
                        raw_rows.append(
                            {
                                **base_row,
                                **metrics,
                                "n_reference_anomalies": 0,
                                "n_detected": int(
                                    predictions["anomaly_pred"].sum()
                                ),
                                "Precision": np.nan,
                                "Recall": np.nan,
                                "F1": np.nan,
                                "metric_status": "forecast_only",
                            }
                        )

                        print(
                            f"K={individual['K']} | "
                            f"generations={len(ga_history)} | "
                            f"RMSE={metrics['RMSE']:.6g}, "
                            f"MAE={metrics['MAE']:.6g}, "
                            f"sMAPE={metrics['sMAPE']:.3f}, "
                            f"R2={metrics['R2']:.6g}"
                        )
                    else:
                        precision, recall, f1, status = anomaly_metrics(
                            predictions["anomaly_truth"].to_numpy(dtype=int),
                            predictions["anomaly_pred"].to_numpy(dtype=int),
                        )
                        raw_rows.append(
                            {
                                **base_row,
                                "RMSE": np.nan,
                                "MAE": np.nan,
                                "sMAPE": np.nan,
                                "R2": np.nan,
                                "R2_manual": np.nan,
                                "n_reference_anomalies": int(
                                    predictions["anomaly_truth"].sum()
                                ),
                                "n_detected": int(
                                    predictions["anomaly_pred"].sum()
                                ),
                                "Precision": precision,
                                "Recall": recall,
                                "F1": f1,
                                "metric_status": status,
                            }
                        )

                        print(
                            f"{scenario_name} | "
                            f"reference={int(predictions['anomaly_truth'].sum())}, "
                            f"detected={int(predictions['anomaly_pred'].sum())}, "
                            f"F1={f1 if np.isfinite(f1) else 'N/A'}"
                        )

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(output_dir / "nfga_raw_results.csv", index=False)

    forecast_summary = (
        raw[raw["scenario"] == "forecast_original"]
        .groupby(["dataset", "model"], as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            K_mean=("K", "mean"),
            K_std=("K", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            sMAPE_mean=("sMAPE", "mean"),
            sMAPE_std=("sMAPE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            training_time_mean_sec=("training_time_sec", "mean"),
            training_time_std_sec=("training_time_sec", "std"),
            inference_latency_mean_sec=("mean_inference_latency_sec", "mean"),
            model_size_mean_bytes=("model_size_bytes", "mean"),
            generations_mean=("generations_ran", "mean"),
        )
    )
    forecast_summary.to_csv(
        output_dir / "nfga_forecast_summary.csv",
        index=False,
    )

    anomaly_summary = (
        raw[raw["scenario"] != "forecast_original"]
        .groupby(["dataset", "model", "scenario"], as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            n_reference_anomalies=("n_reference_anomalies", "first"),
            n_detected_mean=("n_detected", "mean"),
            Precision_mean=("Precision", "mean"),
            Precision_std=("Precision", "std"),
            Recall_mean=("Recall", "mean"),
            Recall_std=("Recall", "std"),
            F1_mean=("F1", "mean"),
            F1_std=("F1", "std"),
            metric_status=("metric_status", "first"),
        )
    )
    anomaly_summary.to_csv(
        output_dir / "nfga_anomaly_summary.csv",
        index=False,
    )

    pd.DataFrame(threshold_rows).to_csv(
        output_dir / "nfga_residual_thresholds.csv",
        index=False,
    )

    (output_dir / "nfga_calibration_metadata.json").write_text(
        json.dumps(
            calibration_metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    metadata = {
        "seeds": list(SEEDS),
        "calibration_seed": CALIBRATION_SEED,
        "models": {
            "NFGA-Core": (
                "Gaussian antecedents with zero-order constant consequents."
            ),
            "NFGA-LINEAR": (
                "Gaussian antecedents with full local linear consequents "
                "over all lagged window inputs plus intercept."
            ),
        },
        "shared_ga_configuration": {
            "K_min": K_MIN,
            "K_max": K_MAX,
            "population_size": POPULATION_SIZE,
            "maximum_generations": MAX_GENERATIONS,
            "tournament_size": TOURNAMENT_SIZE,
            "elite_count": ELITE_COUNT,
            "stopping_patience": NO_IMPROVEMENT_PATIENCE,
            "minimum_improvement": MIN_IMPROVEMENT,
            "crossover_probability": CROSSOVER_PROBABILITY,
            "center_mutation_probability": CENTER_MUTATION_PROBABILITY,
            "sigma_mutation_probability": SIGMA_MUTATION_PROBABILITY,
            "consequent_mutation_probability": (
                CONSEQUENT_MUTATION_PROBABILITY
            ),
            "rule_insertion_probability": RULE_INSERTION_PROBABILITY,
            "rule_deletion_probability": RULE_DELETION_PROBABILITY,
            "lambda_rules": LAMBDA_RULES,
            "lambda_weights": LAMBDA_WEIGHTS,
        },
        "fitness": (
            "training RMSE on scaled targets + normalized rule-count penalty "
            "+ small consequent-weight L2 penalty"
        ),
        "validation": (
            "last 20% of training windows is used only to calibrate the "
            "residual anomaly threshold with an independent calibration seed"
        ),
        "test_protocol": (
            "fixed trained model, full test one-step-ahead forecasting, "
            "using previously observed scenario values in subsequent windows"
        ),
    }
    (output_dir / "nfga_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== Matched NFGA experiments completed =====")
    print("\nForecast summary:")
    print(forecast_summary.to_string(index=False))
    print("\nAnomaly summary:")
    print(anomaly_summary.to_string(index=False))


if __name__ == "__main__":
    main()
