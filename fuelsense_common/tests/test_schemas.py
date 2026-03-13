"""Schema validation tests for forecast, anomaly, optimizer, and health contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fuelsense_common.schemas import (
    AnomalyDetectRequest,
    AnomalyDetectResponse,
    BatchForecastRequest,
    BatchForecastResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    OptimizeRequest,
    OptimizeResponse,
    OptimizeStop,
    OptimizeVehicle,
    QuantilePrediction,
    Route,
    RouteStop,
)


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


def _valid_lookback() -> list[list[float]]:
    return [[float(i + j) for j in range(6)] for i in range(90)]


LOOKBACK_LENGTH = 90
ANOMALY_FACILITY_ID = 7
DEFAULT_SERVICE_TIME = 30
DEFAULT_MAX_ROUTE_DURATION = 480


def test_forecast_request_accepts_valid_shape() -> None:
    """Forecast request should accept a correctly shaped lookback matrix."""
    payload = ForecastRequest(facility_id=1, lookback=_valid_lookback())
    _check(payload.facility_id == 1)
    _check(len(payload.lookback) == LOOKBACK_LENGTH)


def test_forecast_request_rejects_short_lookback() -> None:
    """Forecast request should reject lookback arrays shorter than required."""
    with pytest.raises(ValidationError):
        ForecastRequest(facility_id=1, lookback=_valid_lookback()[:89])


def test_forecast_request_rejects_long_lookback() -> None:
    """Forecast request should reject lookback arrays longer than required."""
    with pytest.raises(ValidationError):
        ForecastRequest(facility_id=1, lookback=[*_valid_lookback(), [0, 0, 0, 0, 0, 0]])


def test_forecast_request_rejects_wrong_feature_width() -> None:
    """Forecast request should reject rows with incorrect feature counts."""
    bad = _valid_lookback()
    bad[0] = [1.0, 2.0]
    with pytest.raises(ValidationError):
        ForecastRequest(facility_id=1, lookback=bad)


def test_quantile_prediction_requires_all_fields() -> None:
    """Quantile prediction should require all quantile fields."""
    model = QuantilePrediction(day=1, p10=1.0, p50=2.0, p90=3.0)
    _check(model.day == 1)
    with pytest.raises(ValidationError):
        QuantilePrediction(day=1, p10=1.0, p50=2.0)  # type: ignore[call-arg]


def test_forecast_response_validates() -> None:
    """Forecast response schema should validate a minimal successful payload."""
    response = ForecastResponse(
        facility_id=5,
        forecast=[QuantilePrediction(day=1, p10=1, p50=2, p90=3)],
        model_version="v1",
        inference_time_ms=5.2,
        device="cpu",
    )
    _check(response.model_version == "v1")


def test_batch_forecast_schemas_validate() -> None:
    """Batch request and response schemas should validate nested forecast models."""
    req = BatchForecastRequest(requests=[ForecastRequest(facility_id=1, lookback=_valid_lookback())])
    _check(len(req.requests) == 1)

    resp = BatchForecastResponse(
        responses=[
            ForecastResponse(
                facility_id=1,
                forecast=[QuantilePrediction(day=1, p10=1, p50=2, p90=3)],
                model_version="v1",
                inference_time_ms=1.0,
                device="cpu",
            ),
        ],
        total_inference_time_ms=1.0,
        device="cpu",
    )
    _check(resp.total_inference_time_ms == 1.0)


def test_anomaly_detect_request_and_response_validate() -> None:
    """Anomaly detect request/response models should validate expected fields."""
    request = AnomalyDetectRequest(
        facility_id=ANOMALY_FACILITY_ID,
        actual_consumption=120.0,
        predicted_consumption=100.0,
        rolling_std=5.0,
        features={
            "z_score_rolling_3d": 2.3,
            "consumption_delta_pct": 0.2,
            "temperature_residual": 1.5,
            "day_of_week": 4.0,
            "hours_since_delivery": 30.0,
            "inventory_level_pct": 0.42,
        },
    )
    _check(request.facility_id == ANOMALY_FACILITY_ID)

    response_stage1 = AnomalyDetectResponse(is_anomaly=False, z_score=1.2, stage=1)
    _check(response_stage1.anomaly_type is None)

    response_stage2 = AnomalyDetectResponse(
        is_anomaly=True,
        z_score=4.2,
        anomaly_type="LEAK",
        confidence=0.91,
        if_score=-0.24,
        stage=2,
    )
    _check(response_stage2.anomaly_type == "LEAK")


def test_anomaly_detect_request_rejects_missing_fields() -> None:
    """Anomaly request should reject missing required nested feature payloads."""
    with pytest.raises(ValidationError):
        AnomalyDetectRequest(
            facility_id=1,
            actual_consumption=1.0,
            predicted_consumption=1.0,
            rolling_std=1.0,
        )  # type: ignore[call-arg]


def test_optimizer_stop_default_service_time() -> None:
    """Optimizer stop schema should apply default service-time value."""
    stop = OptimizeStop(facility_index=1, demand=10.0, time_window_start=0, time_window_end=120)
    _check(stop.service_time == DEFAULT_SERVICE_TIME)


def test_optimizer_request_and_response_validate() -> None:
    """Optimizer request and response schemas should validate routing payloads."""
    request = OptimizeRequest(
        depot_lat=24.7,
        depot_lng=46.7,
        vehicles=[OptimizeVehicle(capacity=1000.0, cost_per_km=2.2)],
        stops=[OptimizeStop(facility_index=1, demand=120.0, time_window_start=0, time_window_end=180)],
        distance_matrix=[[0.0, 10.0], [10.0, 0.0]],
    )
    _check(request.max_route_duration == DEFAULT_MAX_ROUTE_DURATION)

    route = Route(
        vehicle_index=0,
        stops=[RouteStop(facility_index=1, demand=120.0, arrival_min=30, sequence=1)],
        distance_km=10.0,
        cost=22.0,
    )
    response = OptimizeResponse(
        status="optimal",
        routes=[route],
        total_distance_km=10.0,
        total_cost=22.0,
        vehicles_used=1,
        solver_time_ms=14.0,
        baseline_cost=30.0,
        cost_reduction_pct=26.7,
    )
    _check(response.status == "optimal")

    infeasible = OptimizeResponse(
        status="infeasible",
        routes=[],
        total_distance_km=0.0,
        total_cost=0.0,
        vehicles_used=0,
        solver_time_ms=5.0,
        baseline_cost=30.0,
        cost_reduction_pct=0.0,
    )
    _check(infeasible.status == "infeasible")


def test_health_response_validate() -> None:
    """Health schema should validate status and device metadata payload."""
    health = HealthResponse(status="ok", device={"device": "cpu", "backend_loaded": True})
    _check(health.status == "ok")
