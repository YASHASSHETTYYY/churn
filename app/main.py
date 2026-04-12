from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

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
prediction_confidence_mean = Gauge(
    "prediction_confidence_mean",
    "Mean churn probability across served predictions.",
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
model_drift_score = Gauge(
    "model_drift_score",
    "Latest drift score derived from the drift report summary.",
)

_metrics_lock = threading.Lock()
_predictor_lock = threading.Lock()
_request_count = 0
_error_count = 0
_confidence_sum = 0.0
_confidence_observation_count = 0
_predictor: ChurnPredictor | None = None

limiter = Limiter(key_func=get_remote_address, default_limits=[])

app = FastAPI(
    title="Customer Churn Serving API",
    version="2.1.0",
    description="FastAPI serving layer with batch scoring, explanations, and metrics.",
)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


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


def _get_drift_report_path() -> Path:
    return Path("reports/drift_report.json").resolve()


def _load_drift_score() -> float:
    report_path = _get_drift_report_path()
    if not report_path.exists():
        return 0.0

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.0

    drifts = []
    for column in payload.get("columns", []):
        mean_delta = column.get("mean_delta")
        if isinstance(mean_delta, (int, float)):
            drifts.append(abs(float(mean_delta)))
    return max(drifts, default=0.0)


def get_predictor_sync() -> ChurnPredictor:
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                config_path = os.getenv(APP_CONFIG_PATH_ENV, "params.yaml")
                artifact_path = os.getenv(MODEL_ARTIFACT_PATH_ENV)
                _predictor = ChurnPredictor(
                    artifact_path=artifact_path,
                    config_path=config_path,
                )
    return _predictor


async def get_predictor() -> ChurnPredictor:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_predictor_sync)


def _record_metrics(
    start_time: float,
    failed: bool,
    *,
    probabilities: list[float] | None = None,
) -> None:
    global _request_count, _error_count, _confidence_sum, _confidence_observation_count
    prediction_requests_total.inc()
    if failed:
        prediction_errors_total.inc()
    model_latency_seconds.observe(time.perf_counter() - start_time)
    model_drift_score.set(_load_drift_score())

    with _metrics_lock:
        _request_count += 1
        if failed:
            _error_count += 1
        if probabilities:
            _confidence_sum += sum(probabilities)
            _confidence_observation_count += len(probabilities)
            prediction_confidence_mean.set(
                _confidence_sum / _confidence_observation_count
            )
        model_error_rate.set(_error_count / _request_count)


def _customer_to_dict(customer: CustomerData) -> dict[str, Any]:
    return customer.model_dump()


@app.exception_handler(ModelNotTrainedError)
async def model_not_found_handler(_, exc: ModelNotTrainedError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(_, __: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded: maximum 100 requests per minute per IP."},
    )


@app.get("/")
async def root():
    predictor = await get_predictor()
    return {
        "service": "customer-churn-api",
        "status": "ok",
        "features": predictor.feature_names,
        "endpoints": ["/predict", "/predict/batch", "/explain", "/health", "/metrics"],
    }


@app.get("/health")
async def health():
    predictor = await get_predictor()
    return {"status": "ok", "model_version": predictor.model_version}


@app.post("/predict")
@limiter.limit("100/minute")
async def predict(request: Request, customer: CustomerData):
    del request
    started_at = time.perf_counter()
    failed = False
    probabilities: list[float] | None = None
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: get_predictor_sync().predict_one(_customer_to_dict(customer)),
        )
        probabilities = [float(result["churn_probability"])]
        return result
    except Exception:
        failed = True
        raise
    finally:
        _record_metrics(started_at, failed, probabilities=probabilities)


@app.post("/predict/batch")
async def predict_batch(payload: BatchPredictionRequest):
    started_at = time.perf_counter()
    failed = False
    probabilities: list[float] | None = None
    try:
        customers = [_customer_to_dict(customer) for customer in payload.customers]
        loop = asyncio.get_running_loop()
        predictions = await loop.run_in_executor(
            None,
            lambda: get_predictor_sync().predict(customers),
        )
        probabilities = [float(item["churn_probability"]) for item in predictions]
        return {"predictions": predictions}
    except Exception:
        failed = True
        raise
    finally:
        _record_metrics(started_at, failed, probabilities=probabilities)


@app.post("/explain")
async def explain(customer: CustomerData):
    started_at = time.perf_counter()
    failed = False
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: get_predictor_sync().explain(_customer_to_dict(customer)),
        )
    except Exception:
        failed = True
        raise
    finally:
        _record_metrics(started_at, failed)


@app.get("/metrics")
async def metrics():
    model_drift_score.set(_load_drift_score())
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
