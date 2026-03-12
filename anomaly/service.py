"""FastAPI service entrypoint for anomaly detector."""

from __future__ import annotations

import importlib
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any, cast

from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from fuelsense_common.schemas import AnomalyDetectRequest, AnomalyDetectResponse, HealthResponse

INFERENCE_LATENCY = Histogram(
    "anomaly_detector_inference_latency_ms",
    "Inference latency in milliseconds for anomaly detector",
    buckets=(0.5, 1, 2, 5, 10, 25, 50, 100),
)
DETECTIONS_TOTAL = Counter(
    "anomaly_detections_total",
    "Number of anomaly detections by type",
    labelnames=("anomaly_type",),
)

detector: Any | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global detector
    detector_cls: type[Any] | None = None
    module = importlib.import_module("anomaly.detector")
    candidate = getattr(module, "AnomalyDetector", None)
    if isinstance(candidate, type):
        detector_cls = candidate

    if detector_cls is not None:
        detector = detector_cls()
        forest_path = os.environ.get("ANOMALY_IFOREST_PATH", "").strip()
        classifier_path = os.environ.get("ANOMALY_CLASSIFIER_PATH", "").strip()
        if forest_path and classifier_path and hasattr(detector, "load"):
            cast("Any", detector).load(forest_path, classifier_path)
    else:
        detector = None

    yield


app = FastAPI(title="FuelSense Anomaly", version="0.2.0", lifespan=lifespan)


@app.post("/detect", response_model=AnomalyDetectResponse)
def detect(request: AnomalyDetectRequest) -> AnomalyDetectResponse:
    if detector is None or not hasattr(detector, "detect"):
        raise HTTPException(status_code=503, detail="Anomaly detector unavailable")

    start = perf_counter()
    result = cast("Any", detector).detect(
        actual=request.actual_consumption,
        predicted=request.predicted_consumption,
        rolling_std=request.rolling_std,
        features=request.features,
    )
    elapsed_ms = (perf_counter() - start) * 1000
    INFERENCE_LATENCY.observe(elapsed_ms)

    response = AnomalyDetectResponse.model_validate(result)
    anomaly_label = response.anomaly_type or "none"
    DETECTIONS_TOTAL.labels(anomaly_type=anomaly_label).inc()
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if detector is None:
        return HealthResponse(status="degraded", device={"backend_loaded": False, "service": "anomaly"})

    loaded = bool(getattr(detector, "is_loaded", False))
    return HealthResponse(
        status="ok" if loaded else "degraded",
        device={"backend_loaded": loaded, "service": "anomaly"},
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
