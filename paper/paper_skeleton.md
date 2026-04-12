# A Production-Ready MLOps Framework for Customer Churn Prediction with Explainability and Drift Detection

## Abstract (250 words — to be filled manually)

## 1. Introduction
Customer churn prediction is a high-impact supervised learning problem because retention interventions are materially cheaper than reacquisition in telecom, banking, and subscription businesses. However, benchmark-style churn projects often stop at offline accuracy and do not address reproducibility, explainability, deployment health, drift, or fairness. This project reframes churn modeling as a full MLOps problem by combining competitive tabular models with experiment tracking, data versioning, online serving, monitoring, fairness auditing, and rollback readiness.

Contributions:
- A reproducible churn experimentation stack built around MLflow and DVC for lineage, reruns, and publishable artifacts.
- A production-serving layer with async FastAPI inference, rate limiting, Docker health checks, Prometheus telemetry, and Grafana dashboards.
- A research evaluation suite covering ablation benchmarking, bootstrap confidence intervals, SHAP explainability, drift detection, and group fairness analysis.
- A cross-dataset generalization workflow harmonizing telecom and banking churn datasets into a shared schema for transfer evaluation.

## 2. Related Work
Placeholder: "Compare to [X, Y, Z] papers in churn prediction and MLOps"

## 3. Dataset & Preprocessing
The primary telecom churn dataset contains **4,250 samples**, **19 predictive features**, and an observed churn rate of **14.07%**. To improve external validity, the project also incorporates **IBM Telco Customer Churn** (**7,043 samples**, **26.54% churn**) and a **bank churn dataset** (**10,000 samples**, **20.37% churn**). A shared schema was defined in `data/schema.json` with normalized fields for geography, tenure, spending, support interactions, product count, international-plan proxy, and contract grouping.

Preprocessing steps:
- Median imputation for numeric columns.
- Most-frequent imputation for categorical columns.
- One-hot encoding for categorical features.
- Stratified train/test splitting for the primary telecom benchmark.
- Cross-dataset harmonization into a common schema for transfer experiments.

## 4. Methodology
Model architectures used in Phase 1 were Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, and MLPClassifier. Class imbalance was handled with three strategies: SMOTE oversampling, class weighting, and threshold tuning to optimize F1 on a validation split.

The MLOps workflow combines MLflow for experiment tracking, DVC for data and pipeline versioning, FastAPI for async model serving, Docker for packaging and health checks, and Prometheus/Grafana for operational monitoring. Explainability is handled with SHAP artifacts, and governance extensions include fairness auditing and rollback documentation.

## 5. Experimental Results

### Ablation Table

# Model Ablation

Best PR-AUC combination: **CatBoost [threshold_tuning]** with PR-AUC **0.9229**.

Top configurations by AUC-ROC with bootstrap 95% confidence intervals:

| Model | SMOTE | class_weight | AUC-ROC (95% CI) | F1 | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| CatBoost [smote] | Yes | No | 0.9553 (0.9263, 0.9800) | 0.8811 | 0.9205 |
| XGBoost [smote] | Yes | No | 0.9538 (0.9232, 0.9791) | 0.8850 | 0.9178 |
| CatBoost [threshold_tuning] | No | No | 0.9527 (0.9213, 0.9792) | 0.8880 | 0.9229 |

### Bootstrap Confidence Intervals

| Run ID | Model | Strategy | SMOTE | class_weight | Threshold | AUC-ROC | F1 Positive Class | Bootstrap Samples | AUC-ROC CI Lower | AUC-ROC CI Upper | F1 CI Lower | F1 CI Upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catboost__smote | CatBoost | smote | True | False | 0.5 | 0.9553 | 0.8811 | 1000 | 0.9263 | 0.9800 | 0.8326 | 0.9258 |
| xgboost__smote | XGBoost | smote | True | False | 0.5 | 0.9538 | 0.8850 | 1000 | 0.9232 | 0.9791 | 0.8374 | 0.9264 |
| catboost__threshold_tuning | CatBoost | threshold_tuning | False | False | 0.2731 | 0.9527 | 0.8880 | 1000 | 0.9213 | 0.9792 | 0.8450 | 0.9250 |

### Cross-Dataset Generalization

The best in-domain model family from Phase 1 was reused for transfer experiments: **CatBoost with threshold tuning**. Cross-dataset results show meaningful dataset shift and limited portability:

| Train Dataset | Test Dataset | AUC-ROC | PR-AUC | Accuracy | F1 Positive Class |
| --- | --- | --- | --- | --- | --- |
| IBM Telco | SyriaTel Telecom | 0.6754 | 0.3181 | 0.8586 | 0.0099 |
| SyriaTel Telecom | Bank Churn | 0.5981 | 0.2577 | 0.5708 | 0.3950 |
| SyriaTel Telecom | IBM Telco | 0.5323 | 0.2951 | 0.7346 | 0.0000 |
| IBM Telco | Bank Churn | 0.5266 | 0.2357 | 0.7814 | 0.1157 |
| Bank Churn | SyriaTel Telecom | 0.4748 | 0.1331 | 0.4831 | 0.2106 |
| Bank Churn | IBM Telco | 0.4728 | 0.2438 | 0.4384 | 0.4119 |

These results suggest that strong in-domain churn performance does not automatically translate into robust cross-industry transfer.

## 6. Explainability Analysis
SHAP analysis indicated that churn is primarily driven by service friction, plan fit, and usage intensity. The most influential features were `number_customer_service_calls`, `international_plan`, `area_code`, `total_day_minutes`, and `total_day_charge`. Higher customer service contact frequency and expensive daytime or international usage patterns consistently pushed predictions toward churn, while sticky product-engagement signals such as voice-mail adoption were associated with lower churn risk. The analysis supports retention strategies focused on proactive service recovery, usage-plan redesign, and region-specific interventions.

## 7. Drift Detection Evaluation
The drift evaluation compared an Evidently-based detector with a PSI baseline across gradual, sudden, and seasonal drift scenarios. Evidently detected drift in all **3/3** injected scenarios, with an average detection latency of **206.67 records** and mean drift score **0.0332**. The PSI baseline detected **0/3** scenarios and produced a lower average drift score (**0.0105**), indicating lower sensitivity in this setup.

| Detector | Scenarios | Detections | Mean Latency Records | Mean Drift Score |
| --- | --- | --- | --- | --- |
| Evidently | 3 | 3 | 206.6667 | 0.0332 |
| PSI | 3 | 0 | n/a | 0.0105 |

## 8. Fairness Analysis
Fairness was evaluated across three proxy-sensitive features: `state`, `area_code`, and `international_plan`. For each group, the audit computed selection rate, true positive rate, false positive rate, and AUC-ROC using Fairlearn `MetricFrame`. Potential disparate impact was flagged when group TPR differed from the overall TPR by more than **10 percentage points**.

| Sensitive Feature | Number of Groups | Flagged Groups | Max TPR Gap vs Overall | Mean Selection Rate |
| --- | --- | --- | --- | --- |
| state | 51 | 49 | 0.8917 | 0.1450 |
| area_code | 3 | 0 | 0.0789 | 0.1418 |
| international_plan | 2 | 1 | 0.1083 | 0.2804 |

As a mitigation baseline, ExponentiatedGradient with a DemographicParity constraint was applied to the worst proxy grouping (`state`). This reduced the maximum TPR disparity from **0.8917** to **0.7333**, while decreasing overall AUC-ROC from **0.9527** to **0.9301**. The result illustrates a measurable fairness-performance trade-off.

## 9. Conclusion & Future Work
This project demonstrates that a churn system becomes substantially more publishable when predictive modeling is paired with reproducibility, monitoring, explainability, fairness checks, and operational safeguards. In-domain benchmark performance remained strong, but cross-dataset evaluation showed that transfer is challenging and fairness disparities can remain substantial even in high-AUC models.

Future work should prioritize stronger domain adaptation across industries, better proxy-sensitive feature handling, richer protected-attribute fairness studies, cost-aware intervention modeling, and migration from file-based MLflow storage to a database-backed tracking service for longer-lived deployments.

## References (placeholder list of 15 relevant papers)
1. Placeholder: survey of customer churn prediction methods.
2. Placeholder: telecom churn prediction with ensemble learning.
3. Placeholder: explainable AI for customer attrition modeling.
4. Placeholder: fairness in machine learning classification systems.
5. Placeholder: Fairlearn and constrained optimization for group fairness.
6. Placeholder: monitoring model drift in production ML systems.
7. Placeholder: MLOps lifecycle design for reproducible ML.
8. Placeholder: CI/CD patterns for machine learning deployment.
9. Placeholder: SHAP for interpretable churn modeling.
10. Placeholder: benchmarking tree-based models for tabular classification.
11. Placeholder: cross-domain generalization in tabular ML.
12. Placeholder: class imbalance handling with SMOTE and threshold moving.
13. Placeholder: production observability for machine learning APIs.
14. Placeholder: model governance, rollback, and registry workflows.
15. Placeholder: responsible AI auditing in high-impact business applications.
