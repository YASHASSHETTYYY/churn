# Fairness Mitigation Trade-off

Best Phase 1 model: **CatBoost [threshold_tuning]**.

Mitigation baseline: **ExponentiatedGradient + DemographicParity** applied to the worst proxy group feature, **state**.

| Setting | Overall AUC-ROC | Max TPR disparity |
| --- | --- | --- |
| Before mitigation | 0.9527 | 0.8917 |
| After mitigation | 0.9301 | 0.7333 |

Interpretation:
- AUC-ROC changed by **-0.0225** after the fairness constraint.
- Maximum TPR disparity changed by **-0.1583**.
- This baseline is intended as a fairness-aware reference point rather than the final deployment model.
