"""Service API tests for forecaster request validation and health endpoints."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol, cast

import pytest
from httpx import ASGITransport, AsyncClient

from forecaster import service

HTTP_OK = 200
HTTP_UNPROCESSABLE_ENTITY = 422
HTTP_SERVICE_UNAVAILABLE = 503
FACILITY_ID_BATCH_COUNT = 2
FACILITY_ID_SINGLE = 5
FORECAST_HORIZON = 14
TIME_DELTA_EPSILON = 1e-9


class _BatchLike(Protocol):
    shape: tuple[int, ...]


class _CoercePredictions(Protocol):
    def __call__(self, value: object) -> object: ...


if TYPE_CHECKING:
    from pathlib import Path


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


def _valid_lookback() -> list[list[float]]:
    return [[float(i + j) for j in range(6)] for i in range(90)]


@pytest.mark.asyncio
async def test_predict_returns_503_without_backend() -> None:
    """Predict endpoint should return unavailable when backend is not loaded."""
    service.backend = None
    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json={"facility_id": 1, "lookback": _valid_lookback()})
    _check(response.status_code == HTTP_SERVICE_UNAVAILABLE)


@pytest.mark.asyncio
async def test_predict_returns_200_with_mock_backend() -> None:
    """Predict endpoint should return forecast payload when backend responds."""
    class MockBackend:
        device = service.DeviceType.CPU

        def warmup(self) -> None:
            return None

        def health_check(self) -> dict[str, object]:
            return {"status": "ok", "model_loaded": True}

        def predict(self, lookback: object) -> list[list[list[float]]]:
            _ = lookback
            return [[[10.0, 12.0, 14.0] for _ in range(FORECAST_HORIZON)]]

    service.backend = MockBackend()
    service.device_type = service.DeviceType.CPU

    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json={"facility_id": 5, "lookback": _valid_lookback()})
    _check(response.status_code == HTTP_OK)
    data = response.json()
    _check(data["facility_id"] == FACILITY_ID_SINGLE)
    _check(len(data["forecast"]) == FORECAST_HORIZON)


@pytest.mark.asyncio
async def test_predict_rejects_invalid_payload() -> None:
    """Predict endpoint should reject malformed lookback payloads."""
    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json={"facility_id": 1, "lookback": [[1.0, 2.0]]})
    _check(response.status_code == HTTP_UNPROCESSABLE_ENTITY)


@pytest.mark.asyncio
async def test_predict_rejects_non_finite_payload() -> None:
    """Predict endpoint should reject non-finite values in lookback matrix."""
    bad = _valid_lookback()
    bad[0][0] = float("nan")
    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json={"facility_id": 1, "lookback": bad})
    _check(response.status_code == HTTP_UNPROCESSABLE_ENTITY)


@pytest.mark.asyncio
async def test_predict_batch_contract() -> None:
    """Batch predict endpoint should return one response per request."""
    class MockBackend:
        device = service.DeviceType.CPU

        def warmup(self) -> None:
            return None

        def health_check(self) -> dict[str, object]:
            return {"status": "ok", "model_loaded": True}

        def predict(self, lookback: object) -> list[list[list[float]]]:
            batch_like = cast("_BatchLike", lookback)
            batch_size = int(batch_like.shape[0])
            return [[[10.0, 12.0, 14.0] for _ in range(FORECAST_HORIZON)] for _ in range(batch_size)]

    service.backend = MockBackend()
    service.device_type = service.DeviceType.CPU

    payload: dict[str, object] = {
        "requests": [
            {"facility_id": 1, "lookback": _valid_lookback()},
            {"facility_id": 2, "lookback": _valid_lookback()},
        ],
    }

    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict/batch", json=payload)
    _check(response.status_code == HTTP_OK)
    data = response.json()
    _check(len(data["responses"]) == FACILITY_ID_BATCH_COUNT)
    total_ms = float(data["total_inference_time_ms"])
    first_ms = float(data["responses"][0]["inference_time_ms"])
    second_ms = float(data["responses"][1]["inference_time_ms"])
    _check(abs(first_ms - total_ms) < TIME_DELTA_EPSILON)
    _check(abs(second_ms - total_ms) < TIME_DELTA_EPSILON)


@pytest.mark.asyncio
async def test_predict_batch_empty_requests_returns_200() -> None:
    """Batch predict endpoint should accept empty request lists."""
    class MockBackend:
        device = service.DeviceType.CPU

        def warmup(self) -> None:
            return None

        def health_check(self) -> dict[str, object]:
            return {"status": "ok"}

        def predict(self, lookback: object) -> list[list[list[float]]]:
            _ = lookback
            return []

    service.backend = MockBackend()
    service.device_type = service.DeviceType.CPU
    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict/batch", json={"requests": []})
    _check(response.status_code == HTTP_OK)
    _check(response.json()["responses"] == [])


@pytest.mark.asyncio
async def test_predict_batch_rejects_non_finite_payload() -> None:
    """Batch predict endpoint should reject non-finite lookback values."""
    class MockBackend:
        device = service.DeviceType.CPU

        def warmup(self) -> None:
            return None

        def health_check(self) -> dict[str, object]:
            return {"status": "ok", "model_loaded": True}

        def predict(self, lookback: object) -> list[list[list[float]]]:
            batch_like = cast("_BatchLike", lookback)
            batch_size = int(batch_like.shape[0])
            return [[[10.0, 12.0, 14.0] for _ in range(FORECAST_HORIZON)] for _ in range(batch_size)]

    service.backend = MockBackend()
    service.device_type = service.DeviceType.CPU
    bad = _valid_lookback()
    bad[1][1] = float("inf")
    payload: dict[str, object] = {
        "requests": [
            {"facility_id": 1, "lookback": bad},
        ],
    }
    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict/batch", json=payload)
    _check(response.status_code == HTTP_UNPROCESSABLE_ENTITY)


def test_coerce_predictions_rejects_invalid_shape() -> None:
    """Prediction coercion helper should reject flat prediction arrays."""
    coerce = cast("_CoercePredictions", service.__dict__["_coerce_predictions"])
    with pytest.raises(ValueError, match="shape"):
        coerce([1.0, 2.0, 3.0])


@pytest.mark.asyncio
async def test_lifespan_handles_missing_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Service lifespan should tolerate backend resolution failures."""
    def _missing_backend(_name: str, _device: service.DeviceType) -> object:
        _ = _name, _device
        msg = "missing backend"
        raise RuntimeError(msg)

    monkeypatch.setattr("forecaster.service.resolve_device", lambda: service.DeviceType.CPU)
    monkeypatch.setattr("forecaster.service.get_backend", _missing_backend)

    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    _check(response.status_code == HTTP_OK)


@pytest.mark.asyncio
async def test_lifespan_load_model_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Lifespan startup should load model path from environment when available."""
    class MockBackend:
        loaded = False

        def load_model(self, path: str) -> None:
            self.loaded = bool(path)

        def health_check(self) -> dict[str, object]:
            return {"status": "ok"}

    backend = MockBackend()
    model_path = tmp_path / "model.pt"
    model_path.write_text("placeholder")

    def _get_backend(_name: str, _device: service.DeviceType) -> MockBackend:
        _ = _name, _device
        return backend

    monkeypatch.setattr("forecaster.service.resolve_device", lambda: service.DeviceType.CPU)
    monkeypatch.setattr("forecaster.service.get_backend", _get_backend)
    monkeypatch.setenv("MODEL_PATH", os.fspath(model_path))

    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    _check(response.status_code == HTTP_OK)


@pytest.mark.asyncio
async def test_health_and_metrics_endpoints() -> None:
    """Health and metrics endpoints should both return HTTP OK."""
    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        metrics = await client.get("/metrics")
    _check(health.status_code == HTTP_OK)
    _check(metrics.status_code == HTTP_OK)
    _check("text/plain" in metrics.headers["content-type"])
