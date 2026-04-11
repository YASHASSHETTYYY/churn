# Model Ablation

Best PR-AUC combination: **CatBoost [threshold_tuning]** with PR-AUC **0.9229**.

Top configurations by AUC-ROC with bootstrap 95% confidence intervals:

| Model | SMOTE | class_weight | AUC-ROC (95% CI) | F1 | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| CatBoost [smote] | Yes | No | 0.9553 (0.9263, 0.9800) | 0.8811 | 0.9205 |
| XGBoost [smote] | Yes | No | 0.9538 (0.9232, 0.9791) | 0.8850 | 0.9178 |
| CatBoost [threshold_tuning] | No | No | 0.9527 (0.9213, 0.9792) | 0.8880 | 0.9229 |
