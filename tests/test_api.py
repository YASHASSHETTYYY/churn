from __future__ import annotations

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from src.api import app as api_module
from src.models.predict import ChurnPredictor

FEATURES = [
    "number_vmail_messages",
    "total_day_calls",
    "total_eve_minutes",
    "total_eve_charge",
    "total_intl_minutes",
    "number_customer_service_calls",
]


@pytest.fixture()
def sample_customer() -> dict[str, float]:
    return {
        "number_vmail_messages": 12.0,
        "total_day_calls": 112.0,
        "total_eve_minutes": 175.5,
        "total_eve_charge": 14.92,
        "total_intl_minutes": 10.4,
        "number_customer_service_calls": 2.0,
    }


@pytest.fixture()
def predictor(tmp_path) -> ChurnPredictor:
    training_frame = pd.DataFrame(
        [
            [0, 80, 140.0, 12.0, 8.0, 1, "no"],
            [5, 95, 150.0, 12.8, 9.0, 1, "no"],
            [10, 110, 180.0, 15.3, 11.0, 2, "yes"],
            [20, 130, 210.0, 17.9, 12.5, 4, "yes"],
            [25, 145, 225.0, 19.1, 13.4, 5, "yes"],
            [2, 90, 145.0, 12.3, 8.7, 1, "no"],
        ],
        columns=FEATURES + ["churn"],
    )
    model = RandomForestClassifier(random_state=7, n_estimators=25)
    model.fit(training_frame[FEATURES], training_frame["churn"])

    artifact_path = tmp_path / "model.joblib"
    bundle = {
        "model": model,
        "metadata": {
            "feature_names": FEATURES,
            "target": "churn",
            "positive_label": "yes",
        },
    }
    joblib.dump(bundle, artifact_path)
    return ChurnPredictor(artifact_path=artifact_path)


@pytest.fixture()
def client(predictor):
    api_module._predictor = predictor
    api_module._request_count = 0
    api_module._error_count = 0
    return TestClient(api_module.app)


def test_predictor_returns_probability(predictor, sample_customer):
    result = predictor.predict_one(sample_customer)
    assert result["churn"] in {"yes", "no"}
    assert 0.0 <= result["churn_probability"] <= 1.0


def test_predict_endpoint_returns_prediction(client, sample_customer):
    response = client.post("/predict", json=sample_customer)

    assert response.status_code == 200
    payload = response.json()
    assert payload["churn"] in {"yes", "no"}
    assert 0.0 <= payload["churn_probability"] <= 1.0


def test_batch_predict_endpoint_returns_multiple_predictions(client, sample_customer):
    response = client.post(
        "/predict/batch",
        json={"customers": [sample_customer, sample_customer]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["predictions"]) == 2


def test_explain_endpoint_returns_top_factors(client, sample_customer):
    response = client.post("/explain", json=sample_customer)

    assert response.status_code == 200
    payload = response.json()
    assert "top_factors" in payload
    assert len(payload["top_factors"]) > 0


def test_predict_validation_rejects_bad_payload(client, sample_customer):
    invalid = dict(sample_customer)
    invalid["unexpected"] = 1

    response = client.post("/predict", json=invalid)

    assert response.status_code == 422


def test_metrics_endpoint_exposes_custom_metrics(client, sample_customer):
    client.post("/predict", json=sample_customer)
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "prediction_requests_total" in response.text
    assert "model_latency_seconds" in response.text
