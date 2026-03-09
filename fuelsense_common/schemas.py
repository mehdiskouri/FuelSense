"""Shared request and response schemas for FuelSense ML services."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ForecastRequest(BaseModel):
    facility_id: int
    lookback: list[list[float]] = Field(..., min_length=90, max_length=90)

    @field_validator("lookback")
    @classmethod
    def validate_feature_width(cls, value: list[list[float]]) -> list[list[float]]:
        if any(len(row) != 6 for row in value):
            raise ValueError("lookback must be shaped as 90x6")
        return value


class QuantilePrediction(BaseModel):
    day: int
    p10: float
    p50: float
    p90: float


class ForecastResponse(BaseModel):
    facility_id: int
    forecast: list[QuantilePrediction]
    model_version: str
    inference_time_ms: float
    device: str


class BatchForecastRequest(BaseModel):
    requests: list[ForecastRequest]


class BatchForecastResponse(BaseModel):
    responses: list[ForecastResponse]
    total_inference_time_ms: float
    device: str


class AnomalyDetectRequest(BaseModel):
    facility_id: int
    actual_consumption: float
    predicted_consumption: float
    rolling_std: float
    features: dict[str, float]


class AnomalyDetectResponse(BaseModel):
    is_anomaly: bool
    z_score: float
    anomaly_type: str | None = None
    confidence: float | None = None
    if_score: float | None = None
    stage: int


class OptimizeStop(BaseModel):
    facility_index: int
    demand: float
    time_window_start: int
    time_window_end: int
    service_time: int = 30


class OptimizeVehicle(BaseModel):
    capacity: float
    cost_per_km: float


class OptimizeRequest(BaseModel):
    depot_lat: float
    depot_lng: float
    vehicles: list[OptimizeVehicle]
    stops: list[OptimizeStop]
    distance_matrix: list[list[float]]
    max_route_duration: int = 480


class RouteStop(BaseModel):
    facility_index: int
    demand: float
    arrival_min: int
    sequence: int


class Route(BaseModel):
    vehicle_index: int
    stops: list[RouteStop]
    distance_km: float
    cost: float


class OptimizeResponse(BaseModel):
    status: str
    routes: list[Route]
    total_distance_km: float
    total_cost: float
    vehicles_used: int
    solver_time_ms: float
    baseline_cost: float
    cost_reduction_pct: float


class HealthResponse(BaseModel):
    status: str
    device: dict[str, object]
