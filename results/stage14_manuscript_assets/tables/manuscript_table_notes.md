# Table and figure notes

1. RMSE is the prespecified primary forecasting endpoint.
2. Prophet and XGBoost on Electricity differ by approximately 1.06e-6 RMSE and
   must be described as numerically tied at the reported precision.
3. Statistical comparisons among stochastic models use matched seeds. The
   deterministic models are ranked descriptively only.
4. Synthetic anomaly scenarios contain few positives; detector rankings must
   not be generalized to verified real-world anomalies.
5. Proxy robust-z labels are exploratory and excluded from the main detector
   ranking.
6. Model-size measurements use different serialization/parameter definitions
   and are approximate rather than strictly commensurate.
7. The sensitivity study is one-factor-at-a-time and does not estimate
   hyperparameter interactions.
