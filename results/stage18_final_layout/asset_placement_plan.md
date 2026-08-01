# Manuscript asset placement plan

## Main manuscript

1. `fig1_forecast_rmse_overview`
2. `fig2_nfga_primary_rmse_ablation`
3. `fig3_detector_macro_f1`
4. `fig4_online_latency_overview`

## Supplementary material

1. `figS1_nfga_ofat_sensitivity`
2. `figS2_threshold_sensitivity`

## Statistical reporting correction

The Stage 14 ablation asset used Holm correction across RMSE, MAE, and sMAPE
within each dataset. The main manuscript claim is defined on the prespecified
primary endpoint RMSE. The corrected multiplicity family therefore contains
the three matched NFGA-LINEAR versus NFGA-Core RMSE tests, one per dataset.

Corrected Holm-adjusted p-values:

- Cholera: 0.0059
- ILINet: 0.0059
- Electricity: 0.0371
