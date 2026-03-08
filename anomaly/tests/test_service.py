from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from anomaly import service


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

		def detect(self, actual: float, predicted: float, rolling_std: float, features: dict[str, float]) -> dict[str, object]:
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
