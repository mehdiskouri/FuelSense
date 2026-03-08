from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from optimizer import service


def _valid_optimize_payload() -> dict[str, object]:
	return {
		"depot_lat": 24.7,
		"depot_lng": 46.7,
		"vehicles": [{"capacity": 1000.0, "cost_per_km": 2.2}],
		"stops": [
			{
				"facility_index": 1,
				"demand": 120.0,
				"time_window_start": 0,
				"time_window_end": 240,
			}
		],
		"distance_matrix": [[0.0, 10.0], [10.0, 0.0]],
	}


@pytest.mark.asyncio
async def test_optimize_returns_503_without_backend() -> None:
	service.backend = None
	transport = ASGITransport(app=service.app)
	async with AsyncClient(transport=transport, base_url="http://test") as client:
		response = await client.post("/optimize", json=_valid_optimize_payload())
	assert response.status_code == 503


@pytest.mark.asyncio
async def test_optimize_returns_200_with_mock_backend() -> None:
	class MockBackend:
		device = service.DeviceType.CPU

		def warmup(self) -> None:
			return None

		def health_check(self) -> dict[str, object]:
			return {"status": "ok", "model_loaded": True}

		def solve(self, **kwargs: Any) -> dict[str, object]:
			_ = kwargs
			return {
				"status": "optimal",
				"routes": [
					{
						"vehicle_index": 0,
						"stops": [
							{
								"facility_index": 1,
								"demand": 120.0,
								"arrival_min": 30,
								"sequence": 1,
							}
						],
						"distance_km": 10.0,
						"cost": 22.0,
					}
				],
				"total_distance_km": 10.0,
				"total_cost": 22.0,
				"vehicles_used": 1,
				"solver_time_ms": 12.0,
				"baseline_cost": 30.0,
				"cost_reduction_pct": 26.7,
			}

	service.backend = MockBackend()
	service.device_type = service.DeviceType.CPU

	transport = ASGITransport(app=service.app)
	async with AsyncClient(transport=transport, base_url="http://test") as client:
		response = await client.post("/optimize", json=_valid_optimize_payload())
	assert response.status_code == 200
	assert response.json()["status"] == "optimal"


@pytest.mark.asyncio
async def test_optimize_invalid_request_returns_422() -> None:
	transport = ASGITransport(app=service.app)
	async with AsyncClient(transport=transport, base_url="http://test") as client:
		response = await client.post("/optimize", json={"depot_lat": 1.0})
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
