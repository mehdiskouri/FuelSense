"""FastAPI service entrypoint for route optimizer."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any, cast

from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, Histogram, generate_latest
from starlette.responses import Response

from fuelsense_common.compute import ComputeBackend, DeviceType, resolve_device
from fuelsense_common.registry import get_backend
from fuelsense_common.schemas import HealthResponse, OptimizeRequest, OptimizeResponse
import optimizer.backends  # noqa: F401

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

backend: ComputeBackend | None = None
device_type: DeviceType = DeviceType.CPU


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global backend, device_type
    device_type = resolve_device()
    DEVICE_INFO.set(1 if device_type == DeviceType.CUDA else 0)
    try:
        backend = get_backend("route_optimizer", device_type)
        backend.warmup()
    except RuntimeError:
        backend = None
    yield


app = FastAPI(title="FuelSense Optimizer", version="0.2.0", lifespan=lifespan)


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(request: OptimizeRequest) -> OptimizeResponse:
    if backend is None or not hasattr(backend, "solve"):
        raise HTTPException(status_code=503, detail="Optimizer backend unavailable")

    start = perf_counter()
    result = cast(Any, backend).solve(
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
    if backend is None:
        return HealthResponse(status="degraded", device={"device": device_type.value, "backend_loaded": False})
    details = cast(dict[str, object], backend.health_check())
    details.setdefault("device", device_type.value)
    details["backend_loaded"] = True
    return HealthResponse(status="ok", device=details)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
