from __future__ import annotations

import os
import threading
import time
from typing import Annotated

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from src.models.predict import ChurnPredictor, ModelNotTrainedError

APP_CONFIG_PATH_ENV = "APP_CONFIG_PATH"
MODEL_ARTIFACT_PATH_ENV = "MODEL_ARTIFACT_PATH"

prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total number of prediction requests served by the churn API.",
)
prediction_errors_total = Counter(
    "prediction_errors_total",
    "Total number of failed prediction requests.",
)
model_latency_seconds = Histogram(
    "model_latency_seconds",
    "Prediction latency in seconds.",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)
model_error_rate = Gauge(
    "model_error_rate",
    "Fraction of prediction requests that resulted in an error.",
)

_metrics_lock = threading.Lock()
_request_count = 0
_error_count = 0
_predictor: ChurnPredictor | None = None

app = FastAPI(
    title="Customer Churn Serving API",
    version="2.0.0",
    description="FastAPI serving layer with batch scoring, explanations, and metrics.",
)


class CustomerData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Annotated[str, Field(min_length=1)]
    account_length: Annotated[float, Field(ge=0)]
    area_code: Annotated[str, Field(min_length=1)]
    international_plan: Annotated[str, Field(min_length=1)]
    voice_mail_plan: Annotated[str, Field(min_length=1)]
    number_vmail_messages: Annotated[float, Field(ge=0)]
    total_day_minutes: Annotated[float, Field(ge=0)]
    total_day_calls: Annotated[float, Field(ge=0)]
    total_day_charge: Annotated[float, Field(ge=0)]
    total_eve_minutes: Annotated[float, Field(ge=0)]
    total_eve_calls: Annotated[float, Field(ge=0)]
    total_eve_charge: Annotated[float, Field(ge=0)]
    total_night_minutes: Annotated[float, Field(ge=0)]
    total_night_calls: Annotated[float, Field(ge=0)]
    total_night_charge: Annotated[float, Field(ge=0)]
    total_intl_minutes: Annotated[float, Field(ge=0)]
    total_intl_calls: Annotated[float, Field(ge=0)]
    total_intl_charge: Annotated[float, Field(ge=0)]
    number_customer_service_calls: Annotated[float, Field(ge=0)]


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customers: list[CustomerData] = Field(..., min_length=1, max_length=1000)


def get_predictor() -> ChurnPredictor:
    global _predictor
    if _predictor is None:
        config_path = os.getenv(APP_CONFIG_PATH_ENV, "params.yaml")
        artifact_path = os.getenv(MODEL_ARTIFACT_PATH_ENV)
        _predictor = ChurnPredictor(
            artifact_path=artifact_path,
            config_path=config_path,
        )
    return _predictor


def _record_metrics(start_time: float, failed: bool) -> None:
    global _request_count, _error_count
    prediction_requests_total.inc()
    if failed:
        prediction_errors_total.inc()
    model_latency_seconds.observe(time.perf_counter() - start_time)

    with _metrics_lock:
        _request_count += 1
        if failed:
            _error_count += 1
        model_error_rate.set(_error_count / _request_count)


def _customer_to_dict(customer: CustomerData) -> dict[str, float]:
    return customer.model_dump()


@app.exception_handler(ModelNotTrainedError)
async def model_not_found_handler(_, exc: ModelNotTrainedError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/")
def root():
    predictor = get_predictor()
    return {
        "service": "customer-churn-api",
        "status": "ok",
        "features": predictor.feature_names,
        "endpoints": ["/predict", "/predict/batch", "/explain", "/metrics"],
    }


@app.get("/health")
def health():
    predictor = get_predictor()
    return {"status": "healthy", "model_path": str(predictor.artifact_path)}


@app.post("/predict")
def predict(customer: CustomerData):
    started_at = time.perf_counter()
    failed = False
    try:
        result = get_predictor().predict_one(_customer_to_dict(customer))
        return result
    except Exception:
        failed = True
        raise
    finally:
        _record_metrics(started_at, failed)


@app.post("/predict/batch")
def predict_batch(payload: BatchPredictionRequest):
    started_at = time.perf_counter()
    failed = False
    try:
        customers = [_customer_to_dict(customer) for customer in payload.customers]
        return {"predictions": get_predictor().predict(customers)}
    except Exception:
        failed = True
        raise
    finally:
        _record_metrics(started_at, failed)


@app.post("/explain")
def explain(customer: CustomerData):
    started_at = time.perf_counter()
    failed = False
    try:
        return get_predictor().explain(_customer_to_dict(customer))
    except Exception:
        failed = True
        raise
    finally:
        _record_metrics(started_at, failed)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
