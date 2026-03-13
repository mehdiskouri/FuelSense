"""FastAPI service entrypoint for route optimizer."""

from __future__ import annotations

from contextlib import asynccontextmanager
from importlib import import_module
from time import perf_counter
from typing import TYPE_CHECKING, Protocol, cast

from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, Histogram, generate_latest
from starlette.responses import Response

from fuelsense_common.compute import ComputeBackend, DeviceType, resolve_device
from fuelsense_common.registry import get_backend
from fuelsense_common.schemas import HealthResponse, OptimizeRequest, OptimizeResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

SOLVER_TIME = Histogram(
    "optimizer_solver_time_ms",
    "Optimizer solver latency in milliseconds",
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 5000),
)
COST_REDUCTION = Histogram(
    "optimizer_cost_reduction_pct",
    "Route optimization cost reduction percentage",
    buckets=(0, 5, 10, 15, 20, 25, 30, 40, 50),
)
VEHICLES_USED = Histogram(
    "optimizer_vehicles_used",
    "Distribution of vehicles used by route optimizer",
    buckets=(1, 2, 3, 5, 10, 20, 30, 40, 50),
)
DEVICE_INFO = Gauge("optimizer_device_gpu", "1 if CUDA is active, 0 for CPU")


class _RuntimeState:
    def __init__(self) -> None:
        self.backend: ComputeBackend | None = None
        self.device_type: DeviceType = DeviceType.CPU


RUNTIME_STATE = _RuntimeState()


class _SolverBackend(Protocol):
    def solve(self, **kwargs: object) -> dict[str, object]: ...


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Resolve optimizer backend and warm it during application startup."""
    RUNTIME_STATE.device_type = resolve_device()
    DEVICE_INFO.set(1 if RUNTIME_STATE.device_type == DeviceType.CUDA else 0)
    try:
        _ = import_module("optimizer.backends")
        RUNTIME_STATE.backend = get_backend("route_optimizer", RUNTIME_STATE.device_type)
        RUNTIME_STATE.backend.warmup()
    except RuntimeError:
        RUNTIME_STATE.backend = None
    yield


app = FastAPI(title="FuelSense Optimizer", version="0.2.0", lifespan=lifespan)


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(request: OptimizeRequest) -> OptimizeResponse:
    """Run route optimization for one depot payload and return normalized response."""
    if RUNTIME_STATE.backend is None or not hasattr(RUNTIME_STATE.backend, "solve"):
        raise HTTPException(status_code=503, detail="Optimizer backend unavailable")

    start = perf_counter()
    solver_backend = cast("_SolverBackend", RUNTIME_STATE.backend)
    result = solver_backend.solve(
        depot_lat=request.depot_lat,
        depot_lng=request.depot_lng,
        vehicles=[v.model_dump() for v in request.vehicles],
        stops=[s.model_dump() for s in request.stops],
        distance_matrix=request.distance_matrix,
        max_route_duration=request.max_route_duration,
    )
    elapsed_ms = (perf_counter() - start) * 1000
    SOLVER_TIME.observe(elapsed_ms)

    response = OptimizeResponse.model_validate(result)
    COST_REDUCTION.observe(response.cost_reduction_pct)
    VEHICLES_USED.observe(response.vehicles_used)
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report optimizer backend availability and backend-specific health fields."""
    if RUNTIME_STATE.backend is None:
        return HealthResponse(
            status="degraded",
            device={"device": RUNTIME_STATE.device_type.value, "backend_loaded": False},
        )
    details = RUNTIME_STATE.backend.health_check()
    details.setdefault("device", RUNTIME_STATE.device_type.value)
    details["backend_loaded"] = True
    return HealthResponse(status="ok", device=details)


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics for optimizer service runtime and solver behavior."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
