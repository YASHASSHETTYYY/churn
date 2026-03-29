from __future__ import annotations

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from src.api import app as api_module
from src.models.predict import ChurnPredictor

FEATURES = [
    "state",
    "account_length",
    "area_code",
    "international_plan",
    "voice_mail_plan",
    "number_vmail_messages",
    "total_day_minutes",
    "total_day_calls",
    "total_day_charge",
    "total_eve_minutes",
    "total_eve_calls",
    "total_eve_charge",
    "total_night_minutes",
    "total_night_calls",
    "total_night_charge",
    "total_intl_minutes",
    "total_intl_calls",
    "total_intl_charge",
    "number_customer_service_calls",
]


@pytest.fixture()
def sample_customer() -> dict[str, float]:
    return {
        "state": "KS",
        "account_length": 128.0,
        "area_code": "415",
        "international_plan": "no",
        "voice_mail_plan": "yes",
        "number_vmail_messages": 12.0,
        "total_day_minutes": 265.1,
        "total_day_calls": 112.0,
        "total_day_charge": 45.07,
        "total_eve_minutes": 175.5,
        "total_eve_calls": 99.0,
        "total_eve_charge": 14.92,
        "total_night_minutes": 220.3,
        "total_night_calls": 91.0,
        "total_night_charge": 9.91,
        "total_intl_minutes": 10.4,
        "total_intl_calls": 3.0,
        "total_intl_charge": 2.81,
        "number_customer_service_calls": 2.0,
    }


@pytest.fixture()
def predictor(tmp_path) -> ChurnPredictor:
    training_frame = pd.DataFrame(
        [
            ["KS", 100, "415", "no", "yes", 0, 180.0, 80, 30.6, 140.0, 90, 11.9, 180.0, 100, 8.1, 8.0, 2, 2.16, 1, "no"],
            ["OH", 110, "408", "no", "yes", 5, 190.0, 95, 32.3, 150.0, 95, 12.8, 185.0, 98, 8.33, 9.0, 3, 2.43, 1, "no"],
            ["NJ", 120, "415", "yes", "no", 10, 240.0, 110, 40.8, 180.0, 100, 15.3, 210.0, 105, 9.45, 11.0, 4, 2.97, 2, "yes"],
            ["CA", 130, "510", "yes", "no", 20, 260.0, 130, 44.2, 210.0, 110, 17.9, 220.0, 108, 9.9, 12.5, 5, 3.38, 4, "yes"],
            ["TX", 145, "415", "yes", "yes", 25, 280.0, 145, 47.6, 225.0, 120, 19.1, 230.0, 112, 10.35, 13.4, 6, 3.62, 5, "yes"],
            ["WA", 90, "408", "no", "yes", 2, 185.0, 90, 31.45, 145.0, 92, 12.3, 182.0, 96, 8.19, 8.7, 2, 2.35, 1, "no"],
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
