# Final detector audit

## Injected-scenario ranking

### cholera
- injected_5pct:
  - 1. Robust-Z: F1=0.6667, precision=1.0000, recall=0.5000, FPR=0.0000
  - 2. NFGA-LINEAR residual detector: F1=0.6278, precision=0.4589, recall=1.0000, FPR=0.0600
  - 3. IsolationForest: F1=0.1106, precision=0.0585, recall=1.0000, FPR=0.8088
  - 4. LocalOutlierFactor: F1=0.0909, precision=0.0476, recall=1.0000, FPR=1.0000
  - 4. OneClassSVM: F1=0.0909, precision=0.0476, recall=1.0000, FPR=1.0000
- injected_10pct:
  - 1. NFGA-LINEAR residual detector: F1=0.5078, precision=0.5190, recall=0.5000, FPR=0.0500
  - 2. Robust-Z: F1=0.4000, precision=1.0000, recall=0.2500, FPR=0.0000
  - 3. IsolationForest: F1=0.2150, precision=0.1261, recall=0.7875, FPR=0.6118
  - 4. LocalOutlierFactor: F1=0.1739, precision=0.0952, recall=1.0000, FPR=1.0000
  - 4. OneClassSVM: F1=0.1739, precision=0.0952, recall=1.0000, FPR=1.0000

### electricity
- injected_5pct:
  - 1. NFGA-LINEAR residual detector: F1=0.6476, precision=0.4800, recall=1.0000, FPR=0.0550
  - 2. LocalOutlierFactor: F1=0.1600, precision=0.0870, recall=1.0000, FPR=0.5250
  - 3. OneClassSVM: F1=0.1212, precision=0.0645, recall=1.0000, FPR=0.7250
  - 4. Robust-Z: F1=0.0000, precision=0.0000, recall=0.0000, FPR=0.0000
  - 4. IsolationForest: F1=0.0000, precision=0.0000, recall=0.0000, FPR=0.3050
- injected_10pct:
  - 1. NFGA-LINEAR residual detector: F1=0.6315, precision=0.4622, recall=1.0000, FPR=0.1237
  - 2. Robust-Z: F1=0.4000, precision=1.0000, recall=0.2500, FPR=0.0000
  - 3. LocalOutlierFactor: F1=0.2000, precision=0.1111, recall=1.0000, FPR=0.8421
  - 4. IsolationForest: F1=0.1881, precision=0.1076, recall=0.7500, FPR=0.6579
  - 5. OneClassSVM: F1=0.1860, precision=0.1026, recall=1.0000, FPR=0.9211

### ilinet
- injected_5pct:
  - 1. NFGA-LINEAR residual detector: F1=0.3598, precision=0.2195, recall=1.0000, FPR=0.2184
  - 2. LocalOutlierFactor: F1=0.1333, precision=0.0741, recall=0.6667, FPR=0.5102
  - 3. OneClassSVM: F1=0.1111, precision=0.0606, recall=0.6667, FPR=0.6327
  - 4. IsolationForest: F1=0.1049, precision=0.0623, recall=0.3333, FPR=0.3102
  - 5. Robust-Z: F1=0.0000, precision=0.0000, recall=0.0000, FPR=0.2245
- injected_10pct:
  - 1. NFGA-LINEAR residual detector: F1=0.5740, precision=0.4066, recall=0.9800, FPR=0.1532
  - 2. IsolationForest: F1=0.1126, precision=0.0793, recall=0.2000, FPR=0.2574
  - 3. OneClassSVM: F1=0.0606, precision=0.0357, recall=0.2000, FPR=0.5745
  - 4. Robust-Z: F1=0.0000, precision=0.0000, recall=0.0000, FPR=0.1702
  - 4. LocalOutlierFactor: F1=0.0000, precision=0.0000, recall=0.0000, FPR=0.3404

## Macro injected results

- cholera / NFGA-LINEAR residual detector: macro-F1=0.5678, macro-FPR=0.0550, rank=1.
- cholera / Robust-Z: macro-F1=0.5333, macro-FPR=0.0000, rank=2.
- cholera / IsolationForest: macro-F1=0.1628, macro-FPR=0.7103, rank=3.
- cholera / LocalOutlierFactor: macro-F1=0.1324, macro-FPR=1.0000, rank=4.
- cholera / OneClassSVM: macro-F1=0.1324, macro-FPR=1.0000, rank=4.
- electricity / NFGA-LINEAR residual detector: macro-F1=0.6396, macro-FPR=0.0893, rank=1.
- electricity / Robust-Z: macro-F1=0.2000, macro-FPR=0.0000, rank=2.
- electricity / LocalOutlierFactor: macro-F1=0.1800, macro-FPR=0.6836, rank=3.
- electricity / OneClassSVM: macro-F1=0.1536, macro-FPR=0.8230, rank=4.
- electricity / IsolationForest: macro-F1=0.0941, macro-FPR=0.4814, rank=5.
- ilinet / NFGA-LINEAR residual detector: macro-F1=0.4669, macro-FPR=0.1858, rank=1.
- ilinet / IsolationForest: macro-F1=0.1087, macro-FPR=0.2838, rank=2.
- ilinet / OneClassSVM: macro-F1=0.0859, macro-FPR=0.6036, rank=3.
- ilinet / LocalOutlierFactor: macro-F1=0.0667, macro-FPR=0.4253, rank=4.
- ilinet / Robust-Z: macro-F1=0.0000, macro-FPR=0.1974, rank=5.

## Claim control

- NFGA-LINEAR ranked first in 5/6 synthetic injected scenarios.
- Allowed: NFGA-LINEAR's residual detector achieved the highest F1 in most synthetic injected scenarios, while performance remained scenario-dependent.
- Proxy robust-z labels are not verified ground truth.
- Robust-Z performance on the robust-z proxy scenario is circular and must not be reported as independent validation.
