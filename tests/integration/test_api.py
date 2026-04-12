from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app import main as api_module
from tests.test_api import sample_customer as base_sample_customer
from tests.test_api import predictor as base_predictor


@pytest.fixture()
def sample_customer():
    return base_sample_customer.__wrapped__()


@pytest.fixture()
def predictor(tmp_path):
    return base_predictor.__wrapped__(tmp_path)


@pytest_asyncio.fixture()
async def async_client(predictor):
    api_module._predictor = predictor
    api_module._request_count = 0
    api_module._error_count = 0
    api_module._confidence_sum = 0.0
    api_module._confidence_observation_count = 0
    api_module.prediction_confidence_mean.set(0.0)
    api_module.model_error_rate.set(0.0)
    api_module.model_drift_score.set(0.0)
    if hasattr(api_module.limiter, "reset"):
        api_module.limiter.reset()
    elif hasattr(api_module.limiter, "_storage") and hasattr(api_module.limiter._storage, "reset"):
        api_module.limiter._storage.reset()

    transport = ASGITransport(app=api_module.app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.mark.asyncio
async def test_predict_returns_probability(async_client, sample_customer):
    response = await async_client.post("/predict", json=sample_customer)

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["churn_probability"], float)
    assert 0.0 <= payload["churn_probability"] <= 1.0


@pytest.mark.asyncio
async def test_predict_missing_field_returns_422(async_client, sample_customer):
    invalid_payload = dict(sample_customer)
    invalid_payload.pop("state")

    response = await async_client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_returns_429_on_101st_request(async_client, sample_customer):
    last_response = None
    for _ in range(101):
        last_response = await async_client.post("/predict", json=sample_customer)

    assert last_response is not None
    assert last_response.status_code == 429
    assert "Rate limit exceeded" in last_response.json()["detail"]


@pytest.mark.asyncio
async def test_health_returns_model_version(async_client):
    response = await async_client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_version"]
