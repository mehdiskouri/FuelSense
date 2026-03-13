"""Shared request and response schemas for FuelSense ML services."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

LOOKBACK_DAYS = 90
LOOKBACK_FEATURES = 6


class ForecastRequest(BaseModel):
    """Single-entity forecast request payload with fixed lookback shape."""

    facility_id: int
    lookback: list[list[float]] = Field(..., min_length=LOOKBACK_DAYS, max_length=LOOKBACK_DAYS)

    @field_validator("lookback")
    @classmethod
    def validate_feature_width(cls, value: list[list[float]]) -> list[list[float]]:
        """Validate that every lookback row contains the expected feature width."""
        if any(len(row) != LOOKBACK_FEATURES for row in value):
            msg = "lookback must be shaped as 90x6"
            raise ValueError(msg)
        return value


class QuantilePrediction(BaseModel):
    """Quantile forecast for a single future day."""

    day: int
    p10: float
    p50: float
    p90: float


class ForecastResponse(BaseModel):
    """Forecast response payload for one facility."""

    facility_id: int
    forecast: list[QuantilePrediction]
    model_version: str
    inference_time_ms: float
    device: str


class BatchForecastRequest(BaseModel):
    """Batch request containing multiple forecast inputs."""

    requests: list[ForecastRequest]


class BatchForecastResponse(BaseModel):
    """Batch response for forecast inference requests."""

    responses: list[ForecastResponse]
    total_inference_time_ms: float
    device: str


class AnomalyDetectRequest(BaseModel):
    """Input payload for anomaly detection scoring."""

    facility_id: int
    actual_consumption: float
    predicted_consumption: float
    rolling_std: float
    features: dict[str, float]


class AnomalyDetectResponse(BaseModel):
    """Anomaly detection result and confidence metadata."""

    is_anomaly: bool
    z_score: float
    anomaly_type: str | None = None
    confidence: float | None = None
    if_score: float | None = None
    stage: int


class OptimizeStop(BaseModel):
    """Stop-level constraint used by the optimizer service."""

    facility_index: int
    demand: float
    time_window_start: int
    time_window_end: int
    service_time: int = 30


class OptimizeVehicle(BaseModel):
    """Vehicle capacity and cost profile for route optimization."""

    capacity: float
    cost_per_km: float


class OptimizeRequest(BaseModel):
    """Route-optimization request payload for one depot and its fleet."""

    depot_lat: float
    depot_lng: float
    vehicles: list[OptimizeVehicle]
    stops: list[OptimizeStop]
    distance_matrix: list[list[float]]
    max_route_duration: int = 480


class RouteStop(BaseModel):
    """Stop visit information for a generated route."""

    facility_index: int
    demand: float
    arrival_min: int
    sequence: int


class Route(BaseModel):
    """Route output for a single vehicle in an optimization solution."""

    vehicle_index: int
    stops: list[RouteStop]
    distance_km: float
    cost: float


class OptimizeResponse(BaseModel):
    """Optimizer output summary including aggregate KPIs."""

    status: str
    routes: list[Route]
    total_distance_km: float
    total_cost: float
    vehicles_used: int
    solver_time_ms: float
    baseline_cost: float
    cost_reduction_pct: float


class HealthResponse(BaseModel):
    """Generic health-check response shared by service endpoints."""

    status: str
    device: dict[str, object]
