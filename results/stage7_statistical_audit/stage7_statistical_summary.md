# Stage 7 Statistical Audit

## Overall RMSE ranking

### cholera
- 1. Persistence: 96.0345
- 2. ARIMA: 103.009
- 3. NFGA-LINEAR: 136.072 ± 41.7894
- 4. LSTM: 193.804 ± 31.8407
- 5. XGBoost: 391.631 ± 24.7069
- 6. NFGA-Core: 459.729 ± 118.382
- 7. SeasonalNaive: 618.738

### electricity
- 1. XGBoost: 0.130019 ± 0.00156037
- 2. LSTM: 0.138409 ± 0.00460908
- 3. NFGA-LINEAR: 0.147863 ± 0.00312199
- 4. ARIMA: 0.14867
- 5. SeasonalNaive: 0.152623
- 6. NFGA-Core: 0.155309 ± 0.0105783
- 7. Persistence: 0.160625

### ilinet
- 1. ARIMA: 0.395146
- 2. NFGA-LINEAR: 0.442183 ± 0.0142823
- 3. Persistence: 0.501735
- 4. XGBoost: 0.700853 ± 0.0168706
- 5. NFGA-Core: 0.741354 ± 0.0448555
- 6. LSTM: 0.753887 ± 0.0700788
- 7. SeasonalNaive: 1.02479

## Matched NFGA ablation

Positive benefit means NFGA-LINEAR has lower error than NFGA-Core.

- cholera: Core=459.729, LINEAR=136.072, LINEAR better in 10/10 seeds, Holm-adjusted p=0.00585938.
- electricity: Core=0.155309, LINEAR=0.147863, LINEAR better in 8/10 seeds, Holm-adjusted p=0.111328.
- ilinet: Core=0.741354, LINEAR=0.442183, LINEAR better in 10/10 seeds, Holm-adjusted p=0.00585938.

## Interpretation rule

- Statistical superiority is claimed only when the Holm-adjusted p-value is below 0.05 and the effect direction favors NFGA-LINEAR.
- Deterministic single-run models are ranked descriptively; they are not included in seed-level Wilcoxon tests.
- Proxy robust-z anomaly labels remain exploratory and are not treated as verified ground truth.

Supported RMSE comparisons:
- cholera: NFGA-LINEAR vs NFGA-Core, p=0.00585938, rank-biserial=1.000.
- cholera: NFGA-LINEAR vs XGBoost, p=0.00585938, rank-biserial=1.000.
- cholera: NFGA-LINEAR vs LSTM, p=0.0195312, rank-biserial=0.818.
- electricity: NFGA-LINEAR vs NFGA-Core, p=0.0371094, rank-biserial=0.745.
- ilinet: NFGA-LINEAR vs NFGA-Core, p=0.00585938, rank-biserial=1.000.
- ilinet: NFGA-LINEAR vs XGBoost, p=0.00585938, rank-biserial=1.000.
- ilinet: NFGA-LINEAR vs LSTM, p=0.00585938, rank-biserial=1.000.