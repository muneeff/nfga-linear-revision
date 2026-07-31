# Final corrected result audit

## Forecasting winners by RMSE

### cholera
- 1. Persistence: RMSE=96.0345 (single_deterministic_run).
- 2. ARIMA: RMSE=103.009 (single_deterministic_run).
- 3. NFGA-LINEAR: RMSE=136.072 ± 41.7894 (multiseed_mean).
- 4. LSTM: RMSE=193.804 ± 31.8407 (multiseed_mean).
- 5. XGBoost: RMSE=391.631 ± 24.7069 (multiseed_mean).
- 6. NFGA-Core: RMSE=459.729 ± 118.382 (multiseed_mean).
- 7. SeasonalNaive: RMSE=618.738 (single_deterministic_run).
- 8. Prophet: RMSE=1158.88 (single_deterministic_run).

### electricity
- 1. Prophet: RMSE=0.130018 (single_deterministic_run).
- 2. XGBoost: RMSE=0.130019 ± 0.00156037 (multiseed_mean).
- 3. LSTM: RMSE=0.138409 ± 0.00460908 (multiseed_mean).
- 4. NFGA-LINEAR: RMSE=0.147863 ± 0.00312199 (multiseed_mean).
- 5. ARIMA: RMSE=0.14867 (single_deterministic_run).
- 6. SeasonalNaive: RMSE=0.152623 (single_deterministic_run).
- 7. NFGA-Core: RMSE=0.155309 ± 0.0105783 (multiseed_mean).
- 8. Persistence: RMSE=0.160625 (single_deterministic_run).

### ilinet
- 1. ARIMA: RMSE=0.395146 (single_deterministic_run).
- 2. NFGA-LINEAR: RMSE=0.442183 ± 0.0142823 (multiseed_mean).
- 3. Persistence: RMSE=0.501735 (single_deterministic_run).
- 4. XGBoost: RMSE=0.700853 ± 0.0168706 (multiseed_mean).
- 5. NFGA-Core: RMSE=0.741354 ± 0.0448555 (multiseed_mean).
- 6. LSTM: RMSE=0.753887 ± 0.0700788 (multiseed_mean).
- 7. SeasonalNaive: RMSE=1.02479 (single_deterministic_run).
- 8. Prophet: RMSE=1.52696 (single_deterministic_run).

## Synthetic injected anomaly winners

- cholera / injected_10pct: NFGA-LINEAR residual detector (F1=0.5078, FPR=0.0500).
- cholera / injected_5pct: Robust-Z (F1=0.6667, FPR=0.0000).
- electricity / injected_10pct: Prophet residual detector (F1=1.0000, FPR=0.0000).
- electricity / injected_5pct: Prophet residual detector (F1=1.0000, FPR=0.0000).
- ilinet / injected_10pct: NFGA-LINEAR residual detector (F1=0.5740, FPR=0.1532).
- ilinet / injected_5pct: Prophet residual detector (F1=0.4000, FPR=0.1020).

## Macro anomaly winners

- cholera: NFGA-LINEAR residual detector (macro-F1=0.5678, macro-FPR=0.0550).
- electricity: Prophet residual detector (macro-F1=1.0000, macro-FPR=0.0000).
- ilinet: NFGA-LINEAR residual detector (macro-F1=0.4669, macro-FPR=0.1858).

## Claim control

- Allowed: No forecasting model was best on all datasets.
- Allowed: NFGA-LINEAR offered a compact accuracy-complexity trade-off and significantly improved over NFGA-Core in the primary matched RMSE analysis.
- Allowed: After adding corrected Prophet results, NFGA-LINEAR was the best synthetic-anomaly detector in two of six scenarios and the best macro-F1 detector on Cholera and ILINet.
- Allowed: Prophet achieved perfect detection on the small synthetic Electricity scenarios, but this does not establish broad real-world superiority.
- Do not claim: NFGA-LINEAR was universally best in forecasting.
- Do not claim: NFGA-LINEAR was best in five of six anomaly scenarios.
- Do not claim: Prophet was significantly better than XGBoost on Electricity.
- Do not claim: Synthetic anomaly results prove real-world anomaly detection.
- Do not claim: Robust-Z proxy-label performance is independent validation.