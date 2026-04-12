# MLflow Runs

Training runs are tracked in the local `mlruns/` directory by default.

Launch the UI from the project root:

```powershell
mlflow ui --backend-store-uri .\mlruns --port 5000
```

Then open `http://127.0.0.1:5000` and inspect the `telecom-churn` experiment.

