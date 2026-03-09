"""FastAPI service entrypoint for demand forecaster."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any, cast

import numpy as np
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, Histogram, generate_latest
from starlette.responses import Response

from fuelsense_common.compute import ComputeBackend, DeviceType, resolve_device
from fuelsense_common.registry import get_backend
from fuelsense_common.schemas import (
    BatchForecastRequest,
    BatchForecastResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    QuantilePrediction,
)

INFERENCE_LATENCY = Histogram(
    "forecaster_inference_latency_ms",
    "Inference latency in milliseconds for demand forecaster",
    buckets=(0.5, 1, 2, 5, 10, 25, 50, 100),
)
MODEL_VERSION = Gauge("forecaster_model_version", "Model version gauge for demand forecaster")
DEVICE_INFO = Gauge("forecaster_device_gpu", "1 if CUDA is active, 0 for CPU")

backend: ComputeBackend | None = None
device_type: DeviceType = DeviceType.CPU


def _coerce_predictions(predictions: Any) -> np.ndarray:
    arr = np.asarray(predictions, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("backend prediction output must have shape [n, horizon, quantiles]")
    return arr


def _to_forecast_response(facility_id: int, output: np.ndarray, inference_time_ms: float) -> ForecastResponse:
    day_predictions = [
        QuantilePrediction(day=i + 1, p10=float(row[0]), p50=float(row[1]), p90=float(row[2]))
        for i, row in enumerate(output)
    ]
    return ForecastResponse(
        facility_id=facility_id,
        forecast=day_predictions,
        model_version="unknown",
        inference_time_ms=float(inference_time_ms),
        device=device_type.value,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global backend, device_type
    device_type = resolve_device()
    DEVICE_INFO.set(1 if device_type == DeviceType.CUDA else 0)

    try:
        backend = get_backend("demand_forecaster", device_type)
        model_path = os.environ.get("MODEL_PATH", "").strip()
        if model_path and hasattr(backend, "load_model"):
            cast(Any, backend).load_model(model_path)
            MODEL_VERSION.set(1)
    except RuntimeError:
        backend = None

    yield


app = FastAPI(title="FuelSense Forecaster", version="0.2.0", lifespan=lifespan)


@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest) -> ForecastResponse:
    if backend is None or not hasattr(backend, "predict"):
        raise HTTPException(status_code=503, detail="Forecaster backend unavailable")

    lookback = np.asarray(request.lookback, dtype=np.float32).reshape(1, 90, 6)
    start = perf_counter()
    raw_predictions = cast(Any, backend).predict(lookback)
    elapsed_ms = (perf_counter() - start) * 1000
    INFERENCE_LATENCY.observe(elapsed_ms)

    output = _coerce_predictions(raw_predictions)
    return _to_forecast_response(request.facility_id, output[0], elapsed_ms)


@app.post("/predict/batch", response_model=BatchForecastResponse)
def predict_batch(request: BatchForecastRequest) -> BatchForecastResponse:
    if backend is None or not hasattr(backend, "predict"):
        raise HTTPException(status_code=503, detail="Forecaster backend unavailable")

    if not request.requests:
        return BatchForecastResponse(responses=[], total_inference_time_ms=0.0, device=device_type.value)

    batch_input = np.stack([np.asarray(item.lookback, dtype=np.float32) for item in request.requests], axis=0)
    start = perf_counter()
    raw_predictions = cast(Any, backend).predict(batch_input)
    elapsed_ms = (perf_counter() - start) * 1000
    INFERENCE_LATENCY.observe(elapsed_ms)

    outputs = _coerce_predictions(raw_predictions)
    responses = [
        _to_forecast_response(item.facility_id, outputs[idx], elapsed_ms / max(len(request.requests), 1))
        for idx, item in enumerate(request.requests)
    ]
    return BatchForecastResponse(
        responses=responses,
        total_inference_time_ms=elapsed_ms,
        device=device_type.value,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if backend is None:
        return HealthResponse(
            status="degraded",
            device={"device": device_type.value, "backend_loaded": False},
        )
    details = cast(dict[str, object], backend.health_check())
    details.setdefault("device", device_type.value)
    details["backend_loaded"] = True
    return HealthResponse(status="ok", device=details)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
