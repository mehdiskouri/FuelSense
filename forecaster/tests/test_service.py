from __future__ import annotations

import os
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
async def test_predict_batch_empty_requests_returns_200() -> None:
	class MockBackend:
		device = service.DeviceType.CPU

		def warmup(self) -> None:
			return None

		def health_check(self) -> dict[str, object]:
			return {"status": "ok"}

		def predict(self, lookback: Any) -> list[list[list[float]]]:
			_ = lookback
			return []

	service.backend = MockBackend()
	service.device_type = service.DeviceType.CPU
	transport = ASGITransport(app=service.app)
	async with AsyncClient(transport=transport, base_url="http://test") as client:
		response = await client.post("/predict/batch", json={"requests": []})
	assert response.status_code == 200
	assert response.json()["responses"] == []


def test_coerce_predictions_rejects_invalid_shape() -> None:
	with pytest.raises(ValueError):
		service._coerce_predictions([1.0, 2.0, 3.0])


@pytest.mark.asyncio
async def test_lifespan_handles_missing_backend(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("forecaster.service.resolve_device", lambda: service.DeviceType.CPU)
	monkeypatch.setattr(
		"forecaster.service.get_backend",
		lambda _name, _device: (_ for _ in ()).throw(RuntimeError("missing backend")),
	)

	transport = ASGITransport(app=service.app)
	async with AsyncClient(transport=transport, base_url="http://test") as client:
		response = await client.get("/health")
	assert response.status_code == 200


@pytest.mark.asyncio
async def test_lifespan_load_model_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
	class MockBackend:
		loaded = False

		def load_model(self, path: str) -> None:
			self.loaded = bool(path)

		def health_check(self) -> dict[str, object]:
			return {"status": "ok"}

	backend = MockBackend()
	model_path = tmp_path / "model.pt"
	model_path.write_text("placeholder")

	monkeypatch.setattr("forecaster.service.resolve_device", lambda: service.DeviceType.CPU)
	monkeypatch.setattr("forecaster.service.get_backend", lambda _name, _device: backend)
	monkeypatch.setenv("MODEL_PATH", os.fspath(model_path))

	transport = ASGITransport(app=service.app)
	async with AsyncClient(transport=transport, base_url="http://test") as client:
		response = await client.get("/health")
	assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_and_metrics_endpoints() -> None:
	transport = ASGITransport(app=service.app)
	async with AsyncClient(transport=transport, base_url="http://test") as client:
		health = await client.get("/health")
		metrics = await client.get("/metrics")
	assert health.status_code == 200
	assert metrics.status_code == 200
	assert "text/plain" in metrics.headers["content-type"]
