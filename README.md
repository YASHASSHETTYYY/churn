# Customer Churn MLOps Project

Production-oriented customer churn prediction project with optimized training,
FastAPI serving, monitoring, Docker deployment, CI, and an interactive
dashboard.

## What Changed

- FastAPI serving layer with `/predict`, `/predict/batch`, `/explain`,
  `/health`, and `/metrics`
- Optuna-based hyperparameter optimization for the Random Forest model
- Shared prediction service used by the API, monitoring job, and Streamlit app
- SHAP-based local explanations for churn predictions
- Drift monitoring job that generates `reports/drift_report.html`
- Prometheus metrics plus a Grafana dashboard for request volume, latency, and
  errors
- Dockerfile and `docker-compose.yml` for API, dashboard, Prometheus, and
  Grafana
- GitHub Actions workflow for linting, tests, model build, drift report
  generation, and Docker image build

## Stack

- FastAPI
- Uvicorn
- Streamlit
- Optuna
- SHAP
- Evidently
- Prometheus
- Grafana
- scikit-learn
- DVC
- pytest
- flake8

## Recommended Python Version

Use Python 3.11 for local development and deployment. The included Docker and
GitHub Actions setup already targets Python 3.11.

## Project Layout

```text
src/
├── api/app.py                 # FastAPI serving layer
├── config.py                  # Shared config/path helpers
├── data/                      # Data ingestion and splitting
├── models/predict.py          # Shared predictor + SHAP explanations
├── models/train_model.py      # Optuna training pipeline
└── monitoring/drift_report.py # Drift report generation

dashboard/streamlit_app.py     # Interactive prediction dashboard
monitoring/prometheus.yml      # Prometheus scrape config
monitoring/grafana/            # Grafana provisioning + dashboard
docker-compose.yml             # API + dashboard + monitoring stack
```

## Training

```bash
python src/data/load_data.py --config params.yaml
python src/data/split_data.py --config params.yaml
python src/models/train_model.py --config params.yaml --n-trials 50
```

Artifacts are written to:

- `models/churn_model.joblib`
- `reports/model_metrics.json`
- `reports/best_params.json`

## API

Run the serving layer:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"number_vmail_messages\":12,\"total_day_calls\":112,\"total_eve_minutes\":175.5,\"total_eve_charge\":14.92,\"total_intl_minutes\":10.4,\"number_customer_service_calls\":2}"
```

## Drift Monitoring

Generate the drift report:

```bash
python src/monitoring/drift_report.py --config params.yaml
```

Outputs:

- `reports/drift_report.html`
- `reports/drift_report.json`

The script uses Evidently when the runtime supports it and falls back to a
lightweight HTML summary if Evidently cannot initialize.

## Interactive Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

The dashboard supports:

- single-customer prediction
- SHAP top-factor display
- batch CSV scoring and download

## Monitoring Stack

The API exports:

- `prediction_requests_total`
- `prediction_errors_total`
- `model_latency_seconds`
- `model_error_rate`

Start the full local stack:

```bash
docker-compose up --build
```

Services:

- API: `http://localhost:8000`
- Streamlit: `http://localhost:8501`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## Tests and Lint

```bash
python -m flake8 --jobs 1 src tests dashboard app.py
python -m pytest -q tests --basetemp .pytest_run -p no:cacheprovider
```

## CI

`.github/workflows/ci-cd.yaml` now runs:

- dependency installation
- data preparation
- optimized model training
- drift report generation
- flake8
- pytest
- Docker image build
