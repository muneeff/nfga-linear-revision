# Anomaly-threshold sensitivity

Prespecified baseline: k = 3.5.

The best-k entries below are descriptive only because the injected test scenarios were used to compute them. They must not replace the prespecified baseline in the main comparison.

## cholera
- ARIMA: baseline macro-F1=0.3405; descriptive best k=2.0, macro-F1=0.3810, gain=+0.0405.
- LSTM: baseline macro-F1=0.1919; descriptive best k=5.0, macro-F1=0.2151, gain=+0.0232.
- NFGA-Core: baseline macro-F1=0.5526; descriptive best k=5.0, macro-F1=0.6008, gain=+0.0482.
- NFGA-LINEAR: baseline macro-F1=0.5678; descriptive best k=5.0, macro-F1=0.5899, gain=+0.0220.
- Persistence: baseline macro-F1=0.5641; descriptive best k=2.0, macro-F1=0.6000, gain=+0.0359.
- SeasonalNaive: baseline macro-F1=0.5333; descriptive best k=2.0, macro-F1=0.7727, gain=+0.2394.
- XGBoost: baseline macro-F1=0.2129; descriptive best k=5.0, macro-F1=0.2524, gain=+0.0395.

## ilinet
- ARIMA: baseline macro-F1=0.3538; descriptive best k=5.0, macro-F1=0.4048, gain=+0.0510.
- LSTM: baseline macro-F1=0.4390; descriptive best k=5.0, macro-F1=0.4649, gain=+0.0258.
- NFGA-Core: baseline macro-F1=0.2723; descriptive best k=2.0, macro-F1=0.5166, gain=+0.2443.
- NFGA-LINEAR: baseline macro-F1=0.4669; descriptive best k=4.5, macro-F1=0.4693, gain=+0.0024.
- Persistence: baseline macro-F1=0.3478; descriptive best k=5.0, macro-F1=0.3772, gain=+0.0294.
- SeasonalNaive: baseline macro-F1=0.1429; descriptive best k=2.0, macro-F1=0.4818, gain=+0.3390.
- XGBoost: baseline macro-F1=0.4048; descriptive best k=3.5, macro-F1=0.4048, gain=+0.0000.

## electricity
- ARIMA: baseline macro-F1=0.6667; descriptive best k=4.0, macro-F1=0.7333, gain=+0.0667.
- LSTM: baseline macro-F1=0.7358; descriptive best k=5.0, macro-F1=0.9178, gain=+0.1820.
- NFGA-Core: baseline macro-F1=0.4346; descriptive best k=5.0, macro-F1=0.5595, gain=+0.1249.
- NFGA-LINEAR: baseline macro-F1=0.6396; descriptive best k=5.0, macro-F1=0.7297, gain=+0.0901.
- Persistence: baseline macro-F1=0.6667; descriptive best k=3.5, macro-F1=0.6667, gain=+0.0000.
- SeasonalNaive: baseline macro-F1=1.0000; descriptive best k=3.0, macro-F1=1.0000, gain=+0.0000.
- XGBoost: baseline macro-F1=0.6684; descriptive best k=5.0, macro-F1=0.9733, gain=+0.3049.

## Interpretation rule

- Main-paper model comparisons retain k = 3.5.
- Sensitivity results quantify threshold dependence but do not constitute post-hoc tuning.
- Proxy robust-z labels remain exploratory rather than verified ground truth.
