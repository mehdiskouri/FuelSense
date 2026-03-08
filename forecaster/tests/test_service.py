from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from forecaster import service


def _valid_lookback() -> list[list[float]]:
	return [[float(i + j) for j in range(6)] for i in range(90)]


@pytest.mark.asyncio
async def test_predict_returns_503_without_backend() -> None:
	service.backend = None
	transport = ASGITransport(app=service.app)
	async with AsyncClient(transport=transport, base_url="http://test") as client:
		response = await client.post("/predict", json={"facility_id": 1, "lookback": _valid_lookback()})
	assert response.status_code == 503


@pytest.mark.asyncio
async def test_predict_returns_200_with_mock_backend() -> None:
	class MockBackend:
		device = service.DeviceType.CPU

		def warmup(self) -> None:
			return None

		def health_check(self) -> dict[str, object]:
			return {"status": "ok", "model_loaded": True}

		def predict(self, lookback: Any) -> list[list[list[float]]]:
			_ = lookback
			return [[[10.0, 12.0, 14.0] for _ in range(14)]]

	service.backend = MockBackend()
	service.device_type = service.DeviceType.CPU

	transport = ASGITransport(app=service.app)
	async with AsyncClient(transport=transport, base_url="http://test") as client:
		response = await client.post("/predict", json={"facility_id": 5, "lookback": _valid_lookback()})
	assert response.status_code == 200
	data = response.json()
	assert data["facility_id"] == 5
	assert len(data["forecast"]) == 14


@pytest.mark.asyncio
async def test_predict_rejects_invalid_payload() -> None:
	transport = ASGITransport(app=service.app)
	async with AsyncClient(transport=transport, base_url="http://test") as client:
		response = await client.post("/predict", json={"facility_id": 1, "lookback": [[1.0, 2.0]]})
	assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_batch_contract() -> None:
	class MockBackend:
		device = service.DeviceType.CPU

		def warmup(self) -> None:
			return None

		def health_check(self) -> dict[str, object]:
			return {"status": "ok", "model_loaded": True}

		def predict(self, lookback: Any) -> list[list[list[float]]]:
			batch_size = int(lookback.shape[0])
			return [[[10.0, 12.0, 14.0] for _ in range(14)] for _ in range(batch_size)]

	service.backend = MockBackend()
	service.device_type = service.DeviceType.CPU

	payload = {
		"requests": [
			{"facility_id": 1, "lookback": _valid_lookback()},
			{"facility_id": 2, "lookback": _valid_lookback()},
		]
	}

	transport = ASGITransport(app=service.app)
	async with AsyncClient(transport=transport, base_url="http://test") as client:
		response = await client.post("/predict/batch", json=payload)
	assert response.status_code == 200
	data = response.json()
	assert len(data["responses"]) == 2


@pytest.mark.asyncio
async def test_health_and_metrics_endpoints() -> None:
	transport = ASGITransport(app=service.app)
	async with AsyncClient(transport=transport, base_url="http://test") as client:
		health = await client.get("/health")
		metrics = await client.get("/metrics")
	assert health.status_code == 200
	assert metrics.status_code == 200
	assert "text/plain" in metrics.headers["content-type"]
