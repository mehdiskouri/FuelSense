"""API tests for anomaly FastAPI service behavior and metrics exposure."""

from __future__ import annotations

import re
from typing import cast
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from anomaly import service

STATUS_OK = 200
STATUS_UNAVAILABLE = 503
STATUS_UNPROCESSABLE = 422
IFOREST_MODEL_PATH = "artifacts/iforest.joblib"
CLASSIFIER_MODEL_PATH = "artifacts/classifier.joblib"


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


@pytest.mark.asyncio
async def test_lifespan_loads_detector_when_paths_set() -> None:
    """Lifespan should load detector artifacts when env paths are configured."""

    class _Detector:
        def __init__(self) -> None:
            self.is_loaded = False
            self.loaded_paths: tuple[str, str] | None = None

        def load(self, forest_path: str, classifier_path: str) -> None:
            self.is_loaded = True
            self.loaded_paths = (forest_path, classifier_path)

    class _Module:
        AnomalyDetector = _Detector

    def _import_module(name: str) -> object:
        _check(name == "anomaly.detector")
        return _Module()

    with (
        patch.dict(
            "os.environ",
            {
                "ANOMALY_IFOREST_PATH": IFOREST_MODEL_PATH,
                "ANOMALY_CLASSIFIER_PATH": CLASSIFIER_MODEL_PATH,
            },
            clear=False,
        ),
        patch("anomaly.service.importlib.import_module", side_effect=_import_module),
    ):
        async with service.lifespan(service.app):
            det = service.RUNTIME_STATE.detector
            _check(isinstance(det, _Detector))
            typed_det = cast("_Detector", det)
            _check(typed_det.is_loaded is True)
            _check(typed_det.loaded_paths == (IFOREST_MODEL_PATH, CLASSIFIER_MODEL_PATH))


@pytest.mark.asyncio
async def test_health_returns_ok_when_detector_loaded() -> None:
    """Health endpoint should report backend loaded when detector exists."""

    class _LoadedDetector:
        is_loaded = True

    with patch.object(service.RUNTIME_STATE, "detector", _LoadedDetector()):
        transport = ASGITransport(app=service.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

    _check(response.status_code == STATUS_OK)
    body = response.json()
    _check(body["status"] == "ok")
    _check(body["device"]["backend_loaded"] is True)


@pytest.mark.asyncio
async def test_detect_returns_503_without_detector() -> None:
    """Detect endpoint should return 503 when detector is unavailable."""
    payload: dict[str, object] = {
        "facility_id": 1,
        "actual_consumption": 120.0,
        "predicted_consumption": 100.0,
        "rolling_std": 5.0,
        "features": {
            "z_score_rolling_3d": 2.3,
            "consumption_delta_pct": 0.2,
            "temperature_residual": 1.0,
            "day_of_week": 3.0,
            "hours_since_delivery": 30.0,
            "inventory_level_pct": 0.5,
        },
    }
    with patch.object(service.RUNTIME_STATE, "detector", None):
        transport = ASGITransport(app=service.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/detect", json=payload)
    _check(response.status_code == STATUS_UNAVAILABLE)


@pytest.mark.asyncio
async def test_detect_returns_200_with_mock_detector() -> None:
    """Detect endpoint should return model payload when detector is present."""

    class MockDetector:
        is_loaded = True

        def detect(
            self,
            actual: float,
            predicted: float,
            rolling_std: float,
            features: dict[str, float],
        ) -> dict[str, object]:
            _ = actual, predicted, rolling_std, features
            return {
                "is_anomaly": True,
                "z_score": 3.5,
                "anomaly_type": "LEAK",
                "confidence": 0.9,
                "if_score": -0.2,
                "stage": 2,
            }

    payload: dict[str, object] = {
        "facility_id": 1,
        "actual_consumption": 120.0,
        "predicted_consumption": 100.0,
        "rolling_std": 5.0,
        "features": {
            "z_score_rolling_3d": 2.3,
            "consumption_delta_pct": 0.2,
            "temperature_residual": 1.0,
            "day_of_week": 3.0,
            "hours_since_delivery": 30.0,
            "inventory_level_pct": 0.5,
        },
    }
    with patch.object(service.RUNTIME_STATE, "detector", MockDetector()):
        transport = ASGITransport(app=service.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/detect", json=payload)
    _check(response.status_code == STATUS_OK)
    _check(response.json()["anomaly_type"] == "LEAK")


@pytest.mark.asyncio
async def test_detect_returns_non_anomaly_response() -> None:
    """Non-anomaly predictions should be returned unchanged by the endpoint."""

    class MockDetector:
        is_loaded = True

        def detect(
            self,
            actual: float,
            predicted: float,
            rolling_std: float,
            features: dict[str, float],
        ) -> dict[str, object]:
            _ = actual, predicted, rolling_std, features
            return {
                "is_anomaly": False,
                "z_score": 0.5,
                "anomaly_type": None,
                "confidence": 0.0,
                "if_score": None,
                "stage": 1,
            }

    payload: dict[str, object] = {
        "facility_id": 1,
        "actual_consumption": 100.5,
        "predicted_consumption": 100.0,
        "rolling_std": 2.0,
        "features": {
            "z_score_rolling_3d": 0.9,
            "consumption_delta_pct": 0.01,
            "temperature_residual": 0.2,
            "day_of_week": 2.0,
            "hours_since_delivery": 30.0,
            "inventory_level_pct": 0.7,
        },
    }
    with patch.object(service.RUNTIME_STATE, "detector", MockDetector()):
        transport = ASGITransport(app=service.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/detect", json=payload)
    _check(response.status_code == STATUS_OK)
    _check(response.json()["is_anomaly"] is False)
    _check(response.json()["anomaly_type"] is None)


@pytest.mark.asyncio
async def test_detect_invalid_request_returns_422() -> None:
    """Validation errors should produce HTTP 422 for malformed payloads."""
    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/detect", json={"facility_id": 1})
    _check(response.status_code == STATUS_UNPROCESSABLE)


@pytest.mark.asyncio
async def test_health_and_metrics_endpoints() -> None:
    """Health and metrics endpoints should both be reachable."""
    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        metrics = await client.get("/metrics")
    _check(health.status_code == STATUS_OK)
    _check(metrics.status_code == STATUS_OK)
    _check("text/plain" in metrics.headers["content-type"])


@pytest.mark.asyncio
async def test_detect_increments_detection_counter() -> None:
    """A detect call should increase anomaly detection metric counters."""

    class MockDetector:
        is_loaded = True

        def detect(
            self,
            actual: float,
            predicted: float,
            rolling_std: float,
            features: dict[str, float],
        ) -> dict[str, object]:
            _ = actual, predicted, rolling_std, features
            return {
                "is_anomaly": True,
                "z_score": 4.1,
                "anomaly_type": "LEAK",
                "confidence": 0.8,
                "if_score": -0.4,
                "stage": 2,
            }

    payload: dict[str, object] = {
        "facility_id": 1,
        "actual_consumption": 120.0,
        "predicted_consumption": 100.0,
        "rolling_std": 4.0,
        "features": {
            "z_score_rolling_3d": 2.3,
            "consumption_delta_pct": 0.2,
            "temperature_residual": 1.0,
            "day_of_week": 3.0,
            "hours_since_delivery": 30.0,
            "inventory_level_pct": 0.5,
        },
    }

    with patch.object(service.RUNTIME_STATE, "detector", MockDetector()):
        transport = ASGITransport(app=service.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/detect", json=payload)
            metrics = await client.get("/metrics")

    body = metrics.text
    _check(re.search(r'anomaly_detections_total\{anomaly_type="LEAK"\}\s+[0-9.]+', body) is not None)
