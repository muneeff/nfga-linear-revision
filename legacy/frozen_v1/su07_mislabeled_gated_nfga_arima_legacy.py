import os
import time
import warnings
import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings("ignore")

BASE_DIR = "/content/drive/MyDrive/Colab_Projects/Cholera_Forecasting/data/processed"
READY_DIR = os.path.join(BASE_DIR, "ready_for_models")
ANOM_DIR = os.path.join(BASE_DIR, "anomaly_redesign")
OUT_DIR = os.path.join(BASE_DIR, "gated_nfga_arima_results")
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

GATE_WINDOW = 8
GATE_RATIO = 0.90
MU_THRESHOLD = 0.25

ARIMA_ORDERS = [
    (1, 0, 0), (1, 1, 0),
    (2, 1, 0), (1, 1, 1),
    (2, 1, 1), (3, 1, 0)
]


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

    return yhat, np.max(mu, axis=1), np.mean(mu, axis=1)


def init_individual(rng, X_train):
    K = rng.integers(K_MIN, K_MAX + 1)
    chosen = rng.choice(len(X_train), size=K, replace=False)

    centers = X_train[chosen].copy()
    sigmas = rng.uniform(0.2, 2.5, size=K)

    weights = np.zeros((K, 5))
    weights[:, 0] = rng.normal(1.0, 0.10, size=K)
    weights[:, 1] = rng.normal(0.0, 0.05, size=K)
    weights[:, 2] = rng.normal(0.0, 0.10, size=K)
    weights[:, 3] = rng.normal(0.0, 0.05, size=K)
    weights[:, 4] = rng.normal(0.0, 0.10, size=K)

    return {"K": int(K), "centers": centers, "sigmas": sigmas, "weights": weights}


def clone_individual(ind):
    return {
        "K": int(ind["K"]),
        "centers": ind["centers"].copy(),
        "sigmas": ind["sigmas"].copy(),
        "weights": ind["weights"].copy()
    }


def fitness(individual, X_val, y_val):
    yhat, _, _ = nfga_linear_predict(X_val, individual)
    return rmse(y_val, yhat) + LAMBDA_K * (individual["K"] / K_MAX)


def tournament_select(rng, population, fitnesses):
    idxs = rng.choice(len(population), size=TOURNAMENT_SIZE, replace=False)
    best_idx = idxs[np.argmin([fitnesses[i] for i in idxs])]
    return clone_individual(population[best_idx])


def crossover(rng, p1, p2):
    k1, k2 = p1["K"], p2["K"]
    cut1 = rng.integers(1, k1 + 1)
    cut2 = rng.integers(1, k2 + 1)

    centers = np.vstack([p1["centers"][:cut1], p2["centers"][cut2:]])
    sigmas = np.concatenate([p1["sigmas"][:cut1], p2["sigmas"][cut2:]])
    weights = np.vstack([p1["weights"][:cut1], p2["weights"][cut2:]])

    if len(sigmas) < K_MIN:
        return clone_individual(p1)

    if len(sigmas) > K_MAX:
        centers = centers[:K_MAX]
        sigmas = sigmas[:K_MAX]
        weights = weights[:K_MAX]

    return {"K": int(len(sigmas)), "centers": centers, "sigmas": sigmas, "weights": weights}


def mutate(rng, ind, X_train):
    out = clone_individual(ind)
    K = out["K"]

    if rng.random() < MUTATION_RATE:
        out["centers"] += rng.normal(0, 0.03, size=out["centers"].shape)

    if rng.random() < MUTATION_RATE:
        out["sigmas"] *= rng.lognormal(mean=0, sigma=0.12, size=K)
        out["sigmas"] = np.clip(out["sigmas"], 0.05, 5.0)

    if rng.random() < MUTATION_RATE:
        out["weights"] += rng.normal(0, 0.04, size=out["weights"].shape)

    if rng.random() < 0.12 and out["K"] < K_MAX:
        idx = rng.integers(0, len(X_train))
        new_weight = np.zeros((1, 5))
        new_weight[0, 0] = rng.normal(1.0, 0.10)
        new_weight[0, 1:] = rng.normal(0.0, 0.05, size=4)

        out["centers"] = np.vstack([out["centers"], X_train[idx].reshape(1, -1)])
        out["sigmas"] = np.concatenate([out["sigmas"], [rng.uniform(0.2, 2.5)]])
        out["weights"] = np.vstack([out["weights"], new_weight])
        out["K"] += 1

    if rng.random() < 0.08 and out["K"] > K_MIN:
        j = rng.integers(0, out["K"])
        out["centers"] = np.delete(out["centers"], j, axis=0)
        out["sigmas"] = np.delete(out["sigmas"], j)
        out["weights"] = np.delete(out["weights"], j, axis=0)
        out["K"] -= 1

    return out


def train_nfga_linear(X_train, y_train, seed):
    rng = np.random.default_rng(seed)

    val_size = max(10, int(0.2 * len(X_train)))
    X_fit = X_train[:-val_size]
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]

    population = [init_individual(rng, X_fit) for _ in range(POP_SIZE)]
    history = []

    for gen in range(GENERATIONS):
        fits = np.array([fitness(ind, X_val, y_val) for ind in population])
        best_idx = int(np.argmin(fits))

        history.append({
            "generation": gen,
            "best_fitness": float(fits[best_idx]),
            "best_K": int(population[best_idx]["K"])
        })

        new_pop = [clone_individual(population[best_idx])]

        while len(new_pop) < POP_SIZE:
            a = tournament_select(rng, population, fits)
            b = tournament_select(rng, population, fits)
            child = mutate(rng, crossover(rng, a, b), X_fit)
            new_pop.append(child)

        population = new_pop

    fits = np.array([fitness(ind, X_val, y_val) for ind in population])
    best_idx = int(np.argmin(fits))
    return clone_individual(population[best_idx]), pd.DataFrame(history)


def select_arima_order(history):
    best_order, best_aic = None, np.inf

    for order in ARIMA_ORDERS:
        try:
            fit = ARIMA(history, order=order).fit()
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_order = order
        except Exception:
            continue

    return best_order if best_order is not None else (1, 1, 0)


def arima_one_step(history, order):
    try:
        fit = ARIMA(history, order=order).fit()
        return float(fit.forecast()[0])
    except Exception:
        return float(history[-1])


def detect_anomalies_from_errors(train_errors, test_errors):
    med = np.median(train_errors)
    mad = np.median(np.abs(train_errors - med))

    if mad == 0:
        threshold = np.mean(train_errors) + 3 * np.std(train_errors)
    else:
        threshold = med + 3.5 * mad

    return (np.asarray(test_errors) > threshold).astype(int), threshold


def load_anomaly_truth(dataset, scenario, test_len):
    if scenario == "real":
        path = os.path.join(ANOM_DIR, f"{dataset}_test_real_anomalies.csv")
        col = "anomaly_real_robust"
    elif scenario == "injected_5pct":
        path = os.path.join(ANOM_DIR, f"{dataset}_test_injected_5pct.csv")
        col = "anomaly_injected"
    elif scenario == "injected_10pct":
        path = os.path.join(ANOM_DIR, f"{dataset}_test_injected_10pct.csv")
        col = "anomaly_injected"
    else:
        raise ValueError("Unknown scenario")

    df = pd.read_csv(path)
    return df[col].values[:test_len]


def gated_forecast(dataset, seed):
    data = np.load(os.path.join(READY_DIR, f"{dataset}_windows.npz"), allow_pickle=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test_scaled = data["y_test"]

    train_mean = float(data["train_mean"])
    train_std = float(data["train_std"])

    train_df = pd.read_csv(os.path.join(READY_DIR, f"{dataset}_train_scaled.csv"))
    test_df = pd.read_csv(os.path.join(READY_DIR, f"{dataset}_test_scaled.csv"))

    train_original = train_df["y"].values.astype(float)
    test_original_full = test_df["y"].values.astype(float)

    # Align full test with X_test/y_test produced after windowing inside test split.
    offset = len(test_original_full) - len(y_test_scaled)
    test_original = test_original_full[offset:]

    start = time.time()
    nfga_model, ga_history = train_nfga_linear(X_train, y_train, seed)
    train_time = time.time() - start

    nfga_pred_scaled, max_mu, mean_mu = nfga_linear_predict(X_test, nfga_model)
    nfga_pred_original = nfga_pred_scaled * train_std + train_mean

    y_true_original = y_test_scaled * train_std + train_mean

    train_pred_scaled, _, _ = nfga_linear_predict(X_train, nfga_model)
    train_pred_original = train_pred_scaled * train_std + train_mean
    train_y_original = y_train * train_std + train_mean

    train_errors = np.abs(train_y_original - train_pred_original)

    arima_history = list(train_original)
    arima_order = select_arima_order(arima_history)

    final_preds = []
    expert_used = []
    nfga_recent_errors = []
    naive_recent_errors = []

    for t in range(len(y_true_original)):
        arima_pred = arima_one_step(arima_history, arima_order)
        nfga_pred = float(nfga_pred_original[t])

        true_val = float(y_true_original[t])
        naive_pred = arima_history[-1]

        if t < GATE_WINDOW:
            use_arima = False
        else:
            nfga_rmse_recent = rmse(
                y_true_original[t-GATE_WINDOW:t],
                nfga_pred_original[t-GATE_WINDOW:t]
            )
            naive_rmse_recent = rmse(
                y_true_original[t-GATE_WINDOW:t],
                np.array([arima_history[-GATE_WINDOW+i] for i in range(GATE_WINDOW)])
            )

            bad_recent_nfga = nfga_rmse_recent > GATE_RATIO * naive_rmse_recent
            low_membership = max_mu[t] < MU_THRESHOLD

            use_arima = bad_recent_nfga or low_membership

        if use_arima:
            final_pred = arima_pred
            expert_used.append("ARIMA")
        else:
            final_pred = nfga_pred
            expert_used.append("NFGA")

        final_preds.append(final_pred)

        arima_history.append(float(test_original[t]))

        nfga_recent_errors.append(abs(true_val - nfga_pred))
        naive_recent_errors.append(abs(true_val - naive_pred))

    final_preds = np.array(final_preds)

    test_errors = np.abs(y_true_original - final_preds)
    anomaly_pred, threshold = detect_anomalies_from_errors(train_errors, test_errors)

    pred_df = pd.DataFrame({
        "y_true": y_true_original,
        "y_pred_gated": final_preds,
        "y_pred_nfga": nfga_pred_original,
        "max_mu": max_mu,
        "mean_mu": mean_mu,
        "expert_used": expert_used,
        "error_abs": test_errors,
        "anomaly_pred": anomaly_pred
    })

    return {
        "pred_df": pred_df,
        "ga_history": ga_history,
        "model": nfga_model,
        "train_time": train_time,
        "avg_time": train_time / max(1, len(y_true_original)),
        "K": nfga_model["K"],
        "arima_order": arima_order,
        "error_threshold": threshold,
        "mean_max_mu": float(np.mean(max_mu)),
        "mean_mu": float(np.mean(mean_mu)),
        "n_arima": int(sum(np.array(expert_used) == "ARIMA")),
        "n_nfga": int(sum(np.array(expert_used) == "NFGA"))
    }


all_results = []

for dataset in DATASETS:
    print(f"\n================ GATED NFGA-ARIMA: {dataset.upper()} ================")

    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")

        out = gated_forecast(dataset, seed)
        pred_df = out["pred_df"]

        fm = forecast_metrics(pred_df["y_true"].values, pred_df["y_pred_gated"].values)

        print(
            f"K={out['K']} | ARIMA={out['arima_order']} | "
            f"NFGA_used={out['n_nfga']} ARIMA_used={out['n_arima']} | "
            f"RMSE={fm['RMSE']:.4f}, MAE={fm['MAE']:.4f}, "
            f"sMAPE={fm['sMAPE']:.2f}%, R2={fm['R2']:.4f}, "
            f"detected_anom={int(pred_df['anomaly_pred'].sum())}"
        )

        all_results.append({
            "dataset": dataset,
            "model": "Gated_NFGA_ARIMA",
            "seed": seed,
            "scenario": "forecast_original",
            "K": out["K"],
            "arima_order": str(out["arima_order"]),
            "NFGA_used": out["n_nfga"],
            "ARIMA_used": out["n_arima"],
            "RMSE": fm["RMSE"],
            "MAE": fm["MAE"],
            "sMAPE": fm["sMAPE"],
            "R2": fm["R2"],
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "avg_time_sec": out["avg_time"],
            "error_threshold": out["error_threshold"],
            "mean_max_mu": out["mean_max_mu"],
            "mean_mu": out["mean_mu"]
        })

        for scenario in ["real", "injected_5pct", "injected_10pct"]:
            truth = load_anomaly_truth(dataset, scenario, len(pred_df))

            if truth.sum() == 0:
                print(f"Anomaly {scenario}: skipped because true anomalies = 0")
                continue

            p, r, f1 = anomaly_metrics(truth, pred_df["anomaly_pred"].values)

            all_results.append({
                "dataset": dataset,
                "model": "Gated_NFGA_ARIMA",
                "seed": seed,
                "scenario": scenario,
                "K": out["K"],
                "arima_order": str(out["arima_order"]),
                "NFGA_used": out["n_nfga"],
                "ARIMA_used": out["n_arima"],
                "RMSE": np.nan,
                "MAE": np.nan,
                "sMAPE": np.nan,
                "R2": np.nan,
                "Precision": p,
                "Recall": r,
                "F1": f1,
                "avg_time_sec": out["avg_time"],
                "error_threshold": out["error_threshold"],
                "mean_max_mu": out["mean_max_mu"],
                "mean_mu": out["mean_mu"]
            })

        pred_df.to_csv(
            os.path.join(OUT_DIR, f"{dataset}_gated_predictions_seed_{seed}.csv"),
            index=False
        )

        out["ga_history"].to_csv(
            os.path.join(OUT_DIR, f"{dataset}_gated_ga_history_seed_{seed}.csv"),
            index=False
        )


results_df = pd.DataFrame(all_results)
raw_path = os.path.join(OUT_DIR, "gated_nfga_arima_raw_results.csv")
results_df.to_csv(raw_path, index=False)

summary = (
    results_df
    .groupby(["dataset", "model", "scenario"], dropna=False)
    .agg({
        "K": ["mean", "std"],
        "NFGA_used": ["mean", "std"],
        "ARIMA_used": ["mean", "std"],
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

summary_path = os.path.join(OUT_DIR, "gated_nfga_arima_summary.csv")
summary.to_csv(summary_path)

print("\n===== GATED NFGA-ARIMA completed =====")
print("Raw results saved to:", raw_path)
print("Summary saved to:", summary_path)