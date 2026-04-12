# Model Rollback Guide

## Model stage tags in MLflow

Use the MLflow Model Registry UI or CLI to move model versions between lifecycle stages:

- `Staging`: candidate model undergoing validation.
- `Production`: model version currently approved for live traffic.

Example CLI commands:

```bash
mlflow models transition-stage --name churn-model --version 7 --stage Staging
mlflow models transition-stage --name churn-model --version 6 --stage Production
```

## Rollback procedure

1. Identify the bad model version from alerts, drift, or degraded production KPIs.
2. Open MLflow and confirm the currently promoted `Production` version.
3. Find the previous stable version in the registry history.
4. Transition the stable version back to `Production`.
5. Move the faulty version to `Staging` or `Archived`.
6. Redeploy the service so it pulls the restored production model artifact.
7. Verify `/health`, prediction latency, and core business metrics after redeploy.

## Recommended workflow

1. Tag the candidate model as `Staging`.
2. Promote to `Production` only after validation checks pass.
3. Keep the last known-good version documented in release notes or deployment metadata.
4. If an incident occurs, promote the previous version and redeploy immediately.

