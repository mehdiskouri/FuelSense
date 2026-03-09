from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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


@pytest.mark.asyncio
async def test_lifespan_loads_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Backend:
        warmed = False

        def warmup(self) -> None:
            self.warmed = True

        def health_check(self) -> dict[str, object]:
            return {"solver": "mock"}

        def solve(self, **kwargs: Any) -> dict[str, object]:
            _ = kwargs
            return {
                "status": "optimal",
                "routes": [],
                "total_distance_km": 0.0,
                "total_cost": 0.0,
                "vehicles_used": 0,
                "solver_time_ms": 0.0,
                "baseline_cost": 0.0,
                "cost_reduction_pct": 0.0,
            }

    backend = _Backend()
    monkeypatch.setattr("optimizer.service.resolve_device", lambda: service.DeviceType.CPU)
    monkeypatch.setattr("optimizer.service.get_backend", lambda name, device: backend)

    @asynccontextmanager
    async def _run() -> AsyncIterator[None]:
        async with service.lifespan(service.app):
            yield

    async with _run():
        assert service.backend is backend
        assert backend.warmed is True


@pytest.mark.asyncio
async def test_health_degraded_when_backend_none() -> None:
    service.backend = None
    service.device_type = service.DeviceType.CPU
    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "degraded"


@pytest.mark.asyncio
async def test_lifespan_runtimeerror_keeps_backend_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("optimizer.service.resolve_device", lambda: service.DeviceType.CUDA)

    def _raise(name: str, device: object) -> object:
        _ = name, device
        raise RuntimeError("no backend")

    monkeypatch.setattr("optimizer.service.get_backend", _raise)

    @asynccontextmanager
    async def _run() -> AsyncIterator[None]:
        async with service.lifespan(service.app):
            yield

    async with _run():
        assert service.backend is None


@pytest.mark.asyncio
async def test_optimize_returns_503_when_backend_has_no_solve() -> None:
    class NoSolveBackend:
        def health_check(self) -> dict[str, object]:
            return {"status": "ok"}

    service.backend = NoSolveBackend()  # type: ignore[assignment]
    transport = ASGITransport(app=service.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/optimize", json=_valid_optimize_payload())
    assert response.status_code == 503
