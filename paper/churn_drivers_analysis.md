# Key Churn Drivers

SHAP analysis of the Phase 1 best model, **CatBoost** trained with the **smote** imbalance strategy, shows that churn risk is driven primarily by a mix of service friction, plan design, and usage intensity. The most influential features were Number Customer Service Calls, International Plan, Area Code, Total Day Minutes, Total Day Charge. Across the top-ranked features, customers with signals of repeated support issues, expensive usage patterns, or poor product-plan fit tended to receive positive SHAP contributions that pushed the model toward a churn prediction, while protective engagement signals tended to pull predictions back toward retention.

## Top SHAP Drivers

| Feature | Mean |SHAP| | Direction | Business Interpretation |
| --- | --- | --- | --- |
| Number Customer Service Calls | 0.9979 | Higher values increase churn risk | Repeated support contacts suggest unresolved service friction and a higher chance of switching. |
| International Plan | 0.7884 | yes is associated with higher churn risk | International-plan customers may face pricing or package-fit concerns that make churn more likely. |
| Area Code | 0.6441 | Mixed or non-linear effect | Regional variation may reflect local competition, coverage quality, or pricing pressure. |
| Total Day Minutes | 0.5615 | Higher values increase churn risk | Heavy daytime usage increases bill exposure and can signal plan mismatch or price sensitivity. |
| Total Day Charge | 0.4434 | Higher values increase churn risk | Heavy daytime usage increases bill exposure and can signal plan mismatch or price sensitivity. |
| Voice Mail Plan | 0.3689 | no is associated with higher churn risk | Voice-mail adoption often reflects product engagement and can separate sticky users from low-engagement accounts. |
| Total Intl Calls | 0.2660 | Higher values reduce churn risk | International calling frequency captures how central the service is to the customer's routine and whether the plan still fits that need. |
| State | 0.2563 | CA is associated with higher churn risk | Regional variation may reflect local competition, coverage quality, or pricing pressure. |
| Total Eve Charge | 0.1628 | Higher values increase churn risk | Evening usage captures engagement intensity and may reveal whether the tariff structure fits customer habits. |
| Total Eve Minutes | 0.1406 | Higher values increase churn risk | Evening usage captures engagement intensity and may reveal whether the tariff structure fits customer habits. |

## Business Recommendations

1. Prioritize proactive retention outreach for customers with repeated customer-service contacts, and route them to specialist resolution teams before renewal windows.
2. Redesign or personalize plan offers for heavy day-time and international users, especially when current usage patterns imply avoidable bill shock.
3. Investigate region-specific churn pockets to determine whether local competition, coverage, or pricing requires market-level intervention.
