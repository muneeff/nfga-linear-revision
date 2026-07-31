# Step-by-Step Execution Plan

## Phase 0 — Freeze and audit

- Preserve original scripts unchanged.
- Record hashes and environment information.
- Identify missing scripts and datasets.
- Create reviewer-to-experiment matrix.

**Gate:** No experiment runs until prepared data and timestamps pass validation.

## Phase 1 — Data provenance and validation

- Recover raw Cholera, ILINet, and Electricity files.
- Record source, license, retrieval date, checksum, date range, missing values, and aggregation.
- Rebuild train/test splits and windows from one canonical preprocessing module.
- Verify identical test timestamps and targets for every model.

**Outputs:** data profile, split manifest, timestamp manifest, checksums.

## Phase 2 — Anomaly scenario reconstruction

- Generate exact 5% and 10% synthetic anomaly labels.
- Use training-derived robust scale.
- Save clean and corrupted targets in original and scaled units.
- Treat robust-z flags as proxy labels, not verified real anomalies.

**Outputs:** scenario CSV files and metadata JSON.

## Phase 3 — Forecasting baselines

- Seasonal naive/persistence baseline.
- ARIMA with order selection restricted to training/validation.
- Prophet with documented weekly settings and nonnegative handling where justified.
- XGBoost and LSTM with chronological validation.

**Gate:** Prediction alignment and metric-identity tests pass.

## Phase 4 — NFGA-Core and NFGA-LINEAR

- Reconcile equations and implementation.
- Train chromosomes only on fit data.
- Use validation for model selection and early stopping.
- Document population, crossover probability, mutation rates, elitism, stopping rule, bounds, and lambda.
- Save rules, parameters, convergence logs, predictions, timing, and memory.

## Phase 5 — Anomaly detection evaluation

- Generate scenario-specific residuals.
- Compare MAD residual detector, Isolation Forest, LOF, and one additional modern detector.
- Report N/A only when ground-truth positives are absent.
- Report precision, recall, F1, and counts with uncertainty over seeds.

## Phase 6 — Reviewer experiments

- Expanded component-wise ablation.
- Sensitivity analysis for window size, rule count, lambda, crossover, and mutation.
- Computational cost analysis.
- 10–20 seeds when computationally feasible.
- Paired statistical tests plus effect sizes and confidence intervals.

## Phase 7 — Paper artifacts

- Build validated tables and vector figures.
- Rewrite claims from generated results only.
- Correct equations and LaTeX artifacts.
- Add limitations and avoid unsupported superiority claims.

## Phase 8 — Response package

- Point-by-point reviewer response.
- Page/line locations for every change.
- Color-coded revised manuscript.
- Clean manuscript and tracked/highlighted manuscript.
- Reproducibility repository release and archive DOI if available.
