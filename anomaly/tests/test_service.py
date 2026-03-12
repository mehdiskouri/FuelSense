from __future__ import annotations

import re

import pytest
from httpx import ASGITransport, AsyncClient

from anomaly import service


@pytest.mark.asyncio
async def test_lifespan_loads_detector_when_paths_set(monkeypatch: pytest.MonkeyPatch) -> None:
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
        assert name == "anomaly.detector"
        return _Module()

    monkeypatch.setenv("ANOMALY_IFOREST_PATH", "/tmp/iforest.joblib")
    monkeypatch.setenv("ANOMALY_CLASSIFIER_PATH", "/tmp/classifier.joblib")
    monkeypatch.setattr("anomaly.service.importlib.import_module", _import_module)

    async with service.lifespan(service.app):
        assert service.detector is not None
        det = service.detector
        assert bool(getattr(det, "is_loaded", False)) is True
        assert getattr(det, "loaded_paths", None) == ("/tmp/iforest.joblib", "/tmp/classifier.joblib")


@pytest.mark.asyncio
async def test_health_returns_ok_when_detector_loaded() -> None:
    class _LoadedDetector:
        is_loaded = True

    service.detector = _LoadedDetector()
    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["device"]["backend_loaded"] is True


@pytest.mark.asyncio
async def test_detect_returns_503_without_detector() -> None:
    service.detector = None
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
    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/detect", json=payload)
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_detect_returns_200_with_mock_detector() -> None:
    class MockDetector:
        is_loaded = True

        def detect(
            self, actual: float, predicted: float, rolling_std: float, features: dict[str, float],
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

    service.detector = MockDetector()
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
    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/detect", json=payload)
    assert response.status_code == 200
    assert response.json()["anomaly_type"] == "LEAK"


@pytest.mark.asyncio
async def test_detect_returns_non_anomaly_response() -> None:
    class MockDetector:
        is_loaded = True

        def detect(
            self, actual: float, predicted: float, rolling_std: float, features: dict[str, float],
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

    service.detector = MockDetector()
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
    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/detect", json=payload)
    assert response.status_code == 200
    assert response.json()["is_anomaly"] is False
    assert response.json()["anomaly_type"] is None


@pytest.mark.asyncio
async def test_detect_invalid_request_returns_422() -> None:
    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/detect", json={"facility_id": 1})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_health_and_metrics_endpoints() -> None:
    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        metrics = await client.get("/metrics")
    assert health.status_code == 200
    assert metrics.status_code == 200
    assert "text/plain" in metrics.headers["content-type"]


@pytest.mark.asyncio
async def test_detect_increments_detection_counter() -> None:
    class MockDetector:
        is_loaded = True

        def detect(
            self, actual: float, predicted: float, rolling_std: float, features: dict[str, float],
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

    service.detector = MockDetector()
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

    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/detect", json=payload)
        metrics = await client.get("/metrics")

    body = metrics.text
    assert re.search(r'anomaly_detections_total\{anomaly_type="LEAK"\}\s+[0-9.]+', body) is not None
