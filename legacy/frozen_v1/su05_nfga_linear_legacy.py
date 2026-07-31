import os
import time
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings("ignore")

BASE_DIR = "/content/drive/MyDrive/Colab_Projects/Cholera_Forecasting/data/processed"
READY_DIR = os.path.join(BASE_DIR, "ready_for_models")
ANOM_DIR = os.path.join(BASE_DIR, "anomaly_redesign")
OUT_DIR = os.path.join(BASE_DIR, "nfga_linear_results")
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = ["cholera", "ilinet", "electricity"]
SEEDS = [11, 22, 33, 44, 55]

K_MIN = 2
K_MAX = 12
POP_SIZE = 25
GENERATIONS = 40
LAMBDA_K = 0.01
MUTATION_RATE = 0.25
TOURNAMENT_SIZE = 3
EPS = 1e-8


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    out = np.zeros_like(y_true, dtype=float)
    mask = denom != 0
    out[mask] = np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return 100 * np.mean(out)


def forecast_metrics(y_true, y_pred):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    }


def anomaly_metrics(y_true_flags, y_pred_flags):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_flags,
        y_pred_flags,
        average="binary",
        zero_division=0
    )
    return p, r, f1


def make_rule_features(X):
    """
    Features used in each local linear consequent:
    [last value, mean window, slope, std window, bias]
    """
    X = np.asarray(X)

    last = X[:, -1]
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)

    first = X[:, 0]
    slope = (last - first) / max(1, X.shape[1] - 1)

    bias = np.ones(len(X))

    return np.vstack([last, mean, slope, std, bias]).T


def gaussian_activation(X, centers, sigmas):
    X = np.asarray(X)
    acts = []

    for i in range(centers.shape[0]):
        dist2 = np.mean((X - centers[i]) ** 2, axis=1)
        sigma2 = max(sigmas[i] ** 2, EPS)
        mu = np.exp(-dist2 / (2 * sigma2))
        acts.append(mu)

    return np.vstack(acts).T


def nfga_linear_predict(X, individual):
    centers = individual["centers"]
    sigmas = individual["sigmas"]
    weights = individual["weights"]

    mu = gaussian_activation(X, centers, sigmas)
    F = make_rule_features(X)

    local_outputs = F @ weights.T

    denom = np.sum(mu, axis=1) + EPS
    yhat = np.sum(mu * local_outputs, axis=1) / denom

    max_mu = np.max(mu, axis=1)
    mean_mu = np.mean(mu, axis=1)

    return yhat, max_mu, mean_mu


def init_individual(rng, X_train, y_train):
    K = rng.integers(K_MIN, K_MAX + 1)

    chosen = rng.choice(len(X_train), size=K, replace=False)
    centers = X_train[chosen].copy()

    sigmas = rng.uniform(0.2, 2.5, size=K)

    # Start local consequents near naive persistence:
    # y_hat ≈ last value
    weights = np.zeros((K, 5))
    weights[:, 0] = rng.normal(1.0, 0.10, size=K)  # last value coefficient
    weights[:, 1] = rng.normal(0.0, 0.05, size=K)  # mean
    weights[:, 2] = rng.normal(0.0, 0.10, size=K)  # slope
    weights[:, 3] = rng.normal(0.0, 0.05, size=K)  # std
    weights[:, 4] = rng.normal(0.0, 0.10, size=K)  # bias

    return {
        "K": int(K),
        "centers": centers,
        "sigmas": sigmas,
        "weights": weights
    }


def clone_individual(ind):
    return {
        "K": int(ind["K"]),
        "centers": ind["centers"].copy(),
        "sigmas": ind["sigmas"].copy(),
        "weights": ind["weights"].copy()
    }


def fitness(individual, X_val, y_val):
    yhat, _, _ = nfga_linear_predict(X_val, individual)
    error = rmse(y_val, yhat)
    penalty = LAMBDA_K * (individual["K"] / K_MAX)
    return error + penalty


def tournament_select(rng, population, fitnesses):
    idxs = rng.choice(len(population), size=TOURNAMENT_SIZE, replace=False)
    best_idx = idxs[np.argmin([fitnesses[i] for i in idxs])]
    return clone_individual(population[best_idx])


def crossover(rng, parent1, parent2):
    k1 = parent1["K"]
    k2 = parent2["K"]

    cut1 = rng.integers(1, k1 + 1)
    cut2 = rng.integers(1, k2 + 1)

    centers = np.vstack([parent1["centers"][:cut1], parent2["centers"][cut2:]])
    sigmas = np.concatenate([parent1["sigmas"][:cut1], parent2["sigmas"][cut2:]])
    weights = np.vstack([parent1["weights"][:cut1], parent2["weights"][cut2:]])

    if len(sigmas) < K_MIN:
        return clone_individual(parent1)

    if len(sigmas) > K_MAX:
        centers = centers[:K_MAX]
        sigmas = sigmas[:K_MAX]
        weights = weights[:K_MAX]

    return {
        "K": int(len(sigmas)),
        "centers": centers,
        "sigmas": sigmas,
        "weights": weights
    }


def mutate(rng, individual, X_train):
    ind = clone_individual(individual)
    K = ind["K"]

    if rng.random() < MUTATION_RATE:
        ind["centers"] += rng.normal(0, 0.03, size=ind["centers"].shape)

    if rng.random() < MUTATION_RATE:
        factor = rng.lognormal(mean=0, sigma=0.12, size=K)
        ind["sigmas"] *= factor
        ind["sigmas"] = np.clip(ind["sigmas"], 0.05, 5.0)

    if rng.random() < MUTATION_RATE:
        ind["weights"] += rng.normal(0, 0.04, size=ind["weights"].shape)

    if rng.random() < 0.12 and ind["K"] < K_MAX:
        idx = rng.integers(0, len(X_train))
        new_center = X_train[idx].reshape(1, -1)
        new_sigma = np.array([rng.uniform(0.2, 2.5)])

        new_weight = np.zeros((1, 5))
        new_weight[0, 0] = rng.normal(1.0, 0.10)
        new_weight[0, 1:] = rng.normal(0.0, 0.05, size=4)

        ind["centers"] = np.vstack([ind["centers"], new_center])
        ind["sigmas"] = np.concatenate([ind["sigmas"], new_sigma])
        ind["weights"] = np.vstack([ind["weights"], new_weight])
        ind["K"] += 1

    if rng.random() < 0.08 and ind["K"] > K_MIN:
        remove_idx = rng.integers(0, ind["K"])
        ind["centers"] = np.delete(ind["centers"], remove_idx, axis=0)
        ind["sigmas"] = np.delete(ind["sigmas"], remove_idx)
        ind["weights"] = np.delete(ind["weights"], remove_idx, axis=0)
        ind["K"] -= 1

    return ind


def train_nfga_linear(X_train, y_train, seed):
    rng = np.random.default_rng(seed)

    val_size = max(10, int(0.2 * len(X_train)))
    X_fit = X_train[:-val_size]
    y_fit = y_train[:-val_size]
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]

    population = [init_individual(rng, X_fit, y_fit) for _ in range(POP_SIZE)]
    history = []

    for gen in range(GENERATIONS):
        fitnesses = np.array([fitness(ind, X_val, y_val) for ind in population])

        best_idx = int(np.argmin(fitnesses))
        best_fit = float(fitnesses[best_idx])
        best_k = int(population[best_idx]["K"])

        history.append({
            "generation": gen,
            "best_fitness": best_fit,
            "best_K": best_k
        })

        new_population = [clone_individual(population[best_idx])]

        while len(new_population) < POP_SIZE:
            p1 = tournament_select(rng, population, fitnesses)
            p2 = tournament_select(rng, population, fitnesses)

            child = crossover(rng, p1, p2)
            child = mutate(rng, child, X_fit)

            new_population.append(child)

        population = new_population

    final_fitnesses = np.array([fitness(ind, X_val, y_val) for ind in population])
    best_idx = int(np.argmin(final_fitnesses))

    return clone_individual(population[best_idx]), pd.DataFrame(history)


def detect_anomalies_from_errors(train_errors, test_errors):
    train_errors = np.asarray(train_errors)
    test_errors = np.asarray(test_errors)

    med = np.median(train_errors)
    mad = np.median(np.abs(train_errors - med))

    if mad == 0:
        threshold = np.mean(train_errors) + 3 * np.std(train_errors)
    else:
        threshold = med + 3.5 * mad

    flags = (test_errors > threshold).astype(int)
    return flags, threshold


def load_anomaly_truth(dataset, scenario, test_len):
    if scenario == "real":
        path = os.path.join(ANOM_DIR, f"{dataset}_test_real_anomalies.csv")
        df = pd.read_csv(path)
        return df["anomaly_real_robust"].values[:test_len]

    if scenario == "injected_5pct":
        path = os.path.join(ANOM_DIR, f"{dataset}_test_injected_5pct.csv")
        df = pd.read_csv(path)
        return df["anomaly_injected"].values[:test_len]

    if scenario == "injected_10pct":
        path = os.path.join(ANOM_DIR, f"{dataset}_test_injected_10pct.csv")
        df = pd.read_csv(path)
        return df["anomaly_injected"].values[:test_len]

    raise ValueError("Unknown scenario")


all_results = []

for dataset in DATASETS:
    print(f"\n================ NFGA-LINEAR: {dataset.upper()} ================")

    data = np.load(os.path.join(READY_DIR, f"{dataset}_windows.npz"), allow_pickle=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test_scaled = data["y_test"]

    train_mean = float(data["train_mean"])
    train_std = float(data["train_std"])
    window_size = int(data["window_size"])

    y_test_original = y_test_scaled * train_std + train_mean

    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")

        start_train = time.time()
        best_model, ga_history = train_nfga_linear(X_train, y_train, seed)
        train_time = time.time() - start_train

        pred_scaled, max_mu, mean_mu = nfga_linear_predict(X_test, best_model)
        pred_original = pred_scaled * train_std + train_mean

        fm = forecast_metrics(y_test_original, pred_original)

        train_pred_scaled, _, _ = nfga_linear_predict(X_train, best_model)

        train_errors_original = np.abs(
            (y_train * train_std + train_mean)
            - (train_pred_scaled * train_std + train_mean)
        )

        test_errors_original = np.abs(y_test_original - pred_original)

        anomaly_pred, err_threshold = detect_anomalies_from_errors(
            train_errors_original,
            test_errors_original
        )

        avg_time = train_time / max(1, len(y_test_original))

        print(
            f"K={best_model['K']} | "
            f"RMSE={fm['RMSE']:.4f}, MAE={fm['MAE']:.4f}, "
            f"sMAPE={fm['sMAPE']:.2f}%, R2={fm['R2']:.4f}, "
            f"detected_anom={int(anomaly_pred.sum())}"
        )

        all_results.append({
            "dataset": dataset,
            "model": "NFGA_linear",
            "seed": seed,
            "scenario": "forecast_original",
            "K": best_model["K"],
            "RMSE": fm["RMSE"],
            "MAE": fm["MAE"],
            "sMAPE": fm["sMAPE"],
            "R2": fm["R2"],
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "avg_time_sec": avg_time,
            "error_threshold": err_threshold,
            "mean_max_mu": float(np.mean(max_mu)),
            "mean_mu": float(np.mean(mean_mu))
        })

        for scenario in ["real", "injected_5pct", "injected_10pct"]:
            truth = load_anomaly_truth(dataset, scenario, len(y_test_original))

            if truth.sum() == 0:
                print(f"Anomaly {scenario}: skipped because true anomalies = 0")
                continue

            p, r, f1 = anomaly_metrics(truth, anomaly_pred)

            all_results.append({
                "dataset": dataset,
                "model": "NFGA_linear",
                "seed": seed,
                "scenario": scenario,
                "K": best_model["K"],
                "RMSE": np.nan,
                "MAE": np.nan,
                "sMAPE": np.nan,
                "R2": np.nan,
                "Precision": p,
                "Recall": r,
                "F1": f1,
                "avg_time_sec": avg_time,
                "error_threshold": err_threshold,
                "mean_max_mu": float(np.mean(max_mu)),
                "mean_mu": float(np.mean(mean_mu))
            })

        pred_df = pd.DataFrame({
            "y_true": y_test_original,
            "y_pred": pred_original,
            "error_abs": test_errors_original,
            "anomaly_pred": anomaly_pred,
            "max_mu": max_mu,
            "mean_mu": mean_mu
        })

        pred_df.to_csv(
            os.path.join(OUT_DIR, f"{dataset}_nfga_linear_predictions_seed_{seed}.csv"),
            index=False
        )

        ga_history.to_csv(
            os.path.join(OUT_DIR, f"{dataset}_nfga_linear_ga_history_seed_{seed}.csv"),
            index=False
        )

        model_json = {
            "dataset": dataset,
            "seed": seed,
            "K": int(best_model["K"]),
            "sigmas": best_model["sigmas"].tolist(),
            "weights": best_model["weights"].tolist(),
            "window_size": window_size,
            "train_mean": train_mean,
            "train_std": train_std
        }

        with open(
            os.path.join(OUT_DIR, f"{dataset}_nfga_linear_model_seed_{seed}.json"),
            "w"
        ) as f:
            json.dump(model_json, f, indent=2)

results_df = pd.DataFrame(all_results)

raw_path = os.path.join(OUT_DIR, "nfga_linear_raw_results.csv")
results_df.to_csv(raw_path, index=False)

summary = (
    results_df
    .groupby(["dataset", "model", "scenario"], dropna=False)
    .agg({
        "K": ["mean", "std"],
        "RMSE": ["mean", "std"],
        "MAE": ["mean", "std"],
        "sMAPE": ["mean", "std"],
        "R2": ["mean", "std"],
        "Precision": ["mean", "std"],
        "Recall": ["mean", "std"],
        "F1": ["mean", "std"],
        "avg_time_sec": ["mean", "std"],
        "mean_max_mu": ["mean", "std"],
        "mean_mu": ["mean", "std"]
    })
)

summary_path = os.path.join(OUT_DIR, "nfga_linear_summary.csv")
summary.to_csv(summary_path)

print("\n===== NFGA-LINEAR completed =====")
print("Raw results saved to:", raw_path)
print("Summary saved to:", summary_path)