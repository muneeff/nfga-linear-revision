# NFGA-LINEAR Revision and Reproducibility Repository

This repository rebuilds the experiments for the manuscript:

**NFGA-LINEAR: An Explainable Neuro-Fuzzy Genetic Framework for Forecasting Anomaly-Affected Data-Scarce Time Series**

## Status

The original scripts were recovered from Google Colab and preserved unchanged under `legacy/frozen_v1/`.
They are **not** the authoritative implementation because the audit identified methodological and reporting defects.
The corrected implementation is being rebuilt incrementally under `src/` and `scripts/`.

## Scientific rules for the rebuild

1. No result is copied from the submitted manuscript.
2. Every table and figure must be generated from raw per-timestamp predictions.
3. All forecasting models use the same timestamps and the same `y_true` values.
4. Isolation Forest is evaluated only as an anomaly detector, not as a forecaster.
5. Synthetic anomaly scenarios are evaluated on scenario-specific corrupted series.
6. Training-derived statistics only are used for scaling and anomaly thresholds.
7. All stochastic models save per-seed outputs.
8. Statistical claims are limited to what the tests support.

## Planned execution order

```text
00_validate_environment.py
01_build_anomaly_scenarios.py
02_validate_prepared_data.py
03_run_naive_arima_prophet.py
04_run_xgboost.py
05_run_lstm.py
06_run_nfga_core.py
07_run_nfga_linear.py
08_run_anomaly_benchmarks.py
09_run_ablation.py
10_run_sensitivity.py
11_run_cost_analysis.py
12_run_statistics.py
13_build_tables.py
14_build_figures.py
15_validate_artifacts.py
```

See `docs/EXECUTION_PLAN.md`, `docs/LEGACY_AUDIT.md`, and `paper/reviewer_notes/REVIEWER_MATRIX.md`.

## Data policy

Raw datasets are not committed unless their licenses permit redistribution. Place them in `data/raw/` and document source URLs, versions, retrieval dates, checksums, and preprocessing in `data/README.md`.

## GitHub publication

After replacing `YOUR_USERNAME`:

```bash
git remote add origin https://github.com/YOUR_USERNAME/nfga-linear-revision.git
git push -u origin main
```
