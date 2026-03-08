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


def _valid_lookback() -> list[list[float]]:
    return [[float(i + j) for j in range(6)] for i in range(90)]


def test_forecast_request_accepts_valid_shape() -> None:
    payload = ForecastRequest(facility_id=1, lookback=_valid_lookback())
    assert payload.facility_id == 1
    assert len(payload.lookback) == 90


def test_forecast_request_rejects_short_lookback() -> None:
    with pytest.raises(ValidationError):
        ForecastRequest(facility_id=1, lookback=_valid_lookback()[:89])


def test_forecast_request_rejects_long_lookback() -> None:
    with pytest.raises(ValidationError):
        ForecastRequest(facility_id=1, lookback=_valid_lookback() + [[0, 0, 0, 0, 0, 0]])


def test_forecast_request_rejects_wrong_feature_width() -> None:
    bad = _valid_lookback()
    bad[0] = [1.0, 2.0]
    with pytest.raises(ValidationError):
        ForecastRequest(facility_id=1, lookback=bad)


def test_quantile_prediction_requires_all_fields() -> None:
    model = QuantilePrediction(day=1, p10=1.0, p50=2.0, p90=3.0)
    assert model.day == 1
    with pytest.raises(ValidationError):
        QuantilePrediction(day=1, p10=1.0, p50=2.0)  # type: ignore[call-arg]


def test_forecast_response_validates() -> None:
    response = ForecastResponse(
        facility_id=5,
        forecast=[QuantilePrediction(day=1, p10=1, p50=2, p90=3)],
        model_version="v1",
        inference_time_ms=5.2,
        device="cpu",
    )
    assert response.model_version == "v1"


def test_batch_forecast_schemas_validate() -> None:
    req = BatchForecastRequest(requests=[ForecastRequest(facility_id=1, lookback=_valid_lookback())])
    assert len(req.requests) == 1

    resp = BatchForecastResponse(
        responses=[
            ForecastResponse(
                facility_id=1,
                forecast=[QuantilePrediction(day=1, p10=1, p50=2, p90=3)],
                model_version="v1",
                inference_time_ms=1.0,
                device="cpu",
            )
        ],
        total_inference_time_ms=1.0,
        device="cpu",
    )
    assert resp.total_inference_time_ms == 1.0


def test_anomaly_detect_request_and_response_validate() -> None:
    request = AnomalyDetectRequest(
        facility_id=7,
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
    assert request.facility_id == 7

    response_stage1 = AnomalyDetectResponse(is_anomaly=False, z_score=1.2, stage=1)
    assert response_stage1.anomaly_type is None

    response_stage2 = AnomalyDetectResponse(
        is_anomaly=True,
        z_score=4.2,
        anomaly_type="LEAK",
        confidence=0.91,
        if_score=-0.24,
        stage=2,
    )
    assert response_stage2.anomaly_type == "LEAK"


def test_anomaly_detect_request_rejects_missing_fields() -> None:
    with pytest.raises(ValidationError):
        AnomalyDetectRequest(
            facility_id=1,
            actual_consumption=1.0,
            predicted_consumption=1.0,
            rolling_std=1.0,
        )  # type: ignore[call-arg]


def test_optimizer_stop_default_service_time() -> None:
    stop = OptimizeStop(facility_index=1, demand=10.0, time_window_start=0, time_window_end=120)
    assert stop.service_time == 30


def test_optimizer_request_and_response_validate() -> None:
    request = OptimizeRequest(
        depot_lat=24.7,
        depot_lng=46.7,
        vehicles=[OptimizeVehicle(capacity=1000.0, cost_per_km=2.2)],
        stops=[OptimizeStop(facility_index=1, demand=120.0, time_window_start=0, time_window_end=180)],
        distance_matrix=[[0.0, 10.0], [10.0, 0.0]],
    )
    assert request.max_route_duration == 480

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
    assert response.status == "optimal"

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
    assert infeasible.status == "infeasible"


def test_health_response_validate() -> None:
    health = HealthResponse(status="ok", device={"device": "cpu", "backend_loaded": True})
    assert health.status == "ok"
