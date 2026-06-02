from __future__ import annotations

import joblib
import pandas as pd

from src.models.predict import ChurnPredictor
from src.models.train_model import build_training_pipeline
from src.retention import build_retention_intelligence


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


def sample_customer() -> dict[str, float | str]:
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
    model = build_training_pipeline(
        train_x=training_frame[FEATURES],
        random_state=7,
        n_jobs=1,
        estimator_params={
            "n_estimators": 25,
            "max_depth": 4,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
    )
    model.fit(training_frame[FEATURES], training_frame["churn"])
    artifact_path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "metadata": {
                "feature_names": FEATURES,
                "feature_dtypes": {
                    column: str(dtype)
                    for column, dtype in training_frame[FEATURES].dtypes.items()
                },
                "target": "churn",
                "positive_label": "yes",
            },
        },
        artifact_path,
    )
    return ChurnPredictor(artifact_path=artifact_path)


def test_retention_intelligence_builds_plain_english_analysis(tmp_path):
    model = predictor(tmp_path)
    customer = sample_customer()
    intelligence = build_retention_intelligence(model)

    result = intelligence.analyze_customer(customer, monthly_revenue=70.0)

    assert result["analyst"]["summary"]
    assert len(result["analyst"]["plain_english_reasons"]) > 0
    assert len(result["analyst"]["recommendations"]) == 3
    assert result["agents"]["offer_agent"]["offer"]


def test_digital_twin_applies_interventions(tmp_path):
    model = predictor(tmp_path)
    customer = sample_customer()
    intelligence = build_retention_intelligence(model)

    result = intelligence.simulate_digital_twin(
        customer,
        {
            "service_calls_delta": -2,
            "discount_percent": 10,
            "plan_changes": {"international_plan": "no"},
            "day_usage_delta_percent": -20,
        },
        monthly_revenue=70.0,
    )

    assert result["baseline"]["prediction"]["churn_probability"] >= 0.0
    assert result["intervention"]["customer"]["number_customer_service_calls"] == 0.0
    assert result["intervention"]["customer"]["total_day_minutes"] < customer["total_day_minutes"]
    assert "net_revenue_saved" in result["impact"]
