"""Celery task definitions for FuelSense core app."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false

from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from time import perf_counter
from typing import TYPE_CHECKING, Any, ParamSpec, TypedDict, TypeVar, cast

import httpx
from celery import chord, shared_task
from django.db import transaction
from django.db.models import Max
from django.utils import timezone

from fuelsense.core.features import (
    build_anomaly_features,
    build_drift_data,
    build_lookback_matrix,
    extract_training_data,
)
from fuelsense.core.metrics import observe_planning_stage
from fuelsense.core.models import (
    AnomalyAlert,
    Delivery,
    DeliveryItem,
    Depot,
    DepotFacilityAssignment,
    Facility,
    Forecast,
    InventoryLog,
    ModelRegistry,
    PlanningCycle,
)
from fuelsense.core.reorder import filter_below_reorder, is_reliable_reorder_point
from fuelsense.core.routing import build_optimizer_request
from ml_pipeline.anomaly_training import AnomalyTrainer
from ml_pipeline.drift import DriftMonitor
from ml_pipeline.training import ForecastTrainer, TrainingDataset

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

RECOVERABLE_TASK_EXCEPTIONS = (RuntimeError, ValueError, TypeError, KeyError)
P = ParamSpec("P")
R = TypeVar("R")


def _shared_task(*args: object, **kwargs: object) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return a typed Celery task decorator for strict mypy compatibility."""
    decorator = cast("Any", shared_task)(*args, **kwargs)
    return cast("Callable[[Callable[P, R]], Callable[P, R]]", decorator)


def _default_parallel_workers() -> int:
    # Use host capacity by default and cap by work items at call sites.
    return max(os.cpu_count() or 1, 1)


def _build_remote_optimizer_client() -> httpx.Client | None:
    if os.environ.get("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "0") != "1":
        return None
    return httpx.Client(timeout=20.0)


def _delivery_idempotency_key(
    *,
    cycle: PlanningCycle | None,
    depot_id: int,
    vehicle_id: int,
    route: dict[str, Any],
) -> str:
    cycle_component = str(cycle.id) if cycle is not None else "adhoc"
    route_blob = json.dumps(route, sort_keys=True, separators=(",", ":"))
    payload = f"{cycle_component}|{depot_id}|{vehicle_id}|{route_blob}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _remote_required() -> bool:
    return os.environ.get("FUELSENSE_REQUIRE_REMOTE_SERVICES", "0") == "1"


def _handle_remote_unavailable(service_name: str, reason: str) -> None:
    if _remote_required():
        message = f"{service_name} unavailable ({reason}) and FUELSENSE_REQUIRE_REMOTE_SERVICES=1"
        raise RuntimeError(message)
    logger.warning("using fallback mode", extra={"service": service_name, "reason": reason})


def _fallback_optimizer_response(payload: dict[str, object]) -> dict[str, Any]:
    vehicles_raw = payload.get("vehicles", [])
    stops_raw = payload.get("stops", [])
    matrix_raw = payload.get("distance_matrix", [])
    vehicles = vehicles_raw if isinstance(vehicles_raw, list) else []
    stops = stops_raw if isinstance(stops_raw, list) else []
    distance_matrix = matrix_raw if isinstance(matrix_raw, list) else []
    if not vehicles or not distance_matrix:
        return {
            "status": "infeasible",
            "routes": [],
            "total_distance_km": 0.0,
            "total_cost": 0.0,
            "vehicles_used": 0,
            "solver_time_ms": 0.0,
            "baseline_cost": 0.0,
            "cost_reduction_pct": 0.0,
        }

    total_distance = 0.0
    total_cost = 0.0
    routes: list[dict[str, object]] = []
    min_cost_per_km = min(float(v.get("cost_per_km", 1.0)) for v in vehicles if isinstance(v, dict))
    baseline_cost = 0.0
    for stop in stops:
        if not isinstance(stop, dict):
            continue
        idx = int(stop.get("facility_index", 0))
        if idx <= 0:
            continue
        out_dist = float(distance_matrix[0][idx]) if idx < len(distance_matrix) else 0.0
        route_distance = out_dist * 2.0
        route_cost = route_distance * min_cost_per_km
        baseline_cost += route_cost
        total_distance += route_distance
        total_cost += route_cost
        routes.append(
            {
                "vehicle_index": 0,
                "stops": [
                    {
                        "facility_index": idx,
                        "demand": float(stop.get("demand", 0.0)),
                        "arrival_min": int(stop.get("time_window_start", 0)),
                        "sequence": 1,
                    },
                ],
                "distance_km": route_distance,
                "cost": route_cost,
            },
        )

    return {
        "status": "optimal" if routes else "infeasible",
        "routes": routes,
        "total_distance_km": total_distance,
        "total_cost": total_cost,
        "vehicles_used": 1 if routes else 0,
        "solver_time_ms": 0.0,
        "baseline_cost": baseline_cost,
        "cost_reduction_pct": 0.0,
    }


def _run_optimizer(payload: dict[str, object], optimizer_client: httpx.Client | None = None) -> dict[str, Any]:
    service_url = os.environ.get("OPTIMIZER_URL", "http://route-optimizer:8003").rstrip("/")
    endpoint = f"{service_url}/optimize"
    remote_enabled = os.environ.get("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "0") == "1"

    if not remote_enabled:
        _handle_remote_unavailable("optimizer", "remote_disabled")
        return _fallback_optimizer_response(payload)

    if optimizer_client is not None:
        try:
            response = optimizer_client.post(endpoint, json=payload)
            response.raise_for_status()
            body = response.json()
            return body if isinstance(body, dict) else _fallback_optimizer_response(payload)
        except httpx.HTTPError:
            _handle_remote_unavailable("optimizer", "request_failed")
            return _fallback_optimizer_response(payload)

    with httpx.Client(timeout=20.0) as client:
        try:
            response = client.post(endpoint, json=payload)
            response.raise_for_status()
            body = response.json()
            return body if isinstance(body, dict) else _fallback_optimizer_response(payload)
        except httpx.HTTPError:
            _handle_remote_unavailable("optimizer", "request_failed")
            return _fallback_optimizer_response(payload)


def _planned_arrival_for_minutes(arrival_min: int) -> datetime:
    start_of_day = timezone.localtime().replace(hour=0, minute=0, second=0, microsecond=0)
    return start_of_day + timedelta(minutes=max(arrival_min, 0))


@dataclass(frozen=True)
class _DeliveryMaterializationRequest:
    depot: Depot
    routes: list[dict[str, Any]]
    vehicles: list[Any]
    facility_index_map: dict[int, int]
    cycle: PlanningCycle | None
    solver_time_ms: float


def _materialize_delivery_routes(request: _DeliveryMaterializationRequest) -> int:
    depot = request.depot
    routes = request.routes
    vehicles = request.vehicles
    facility_index_map = request.facility_index_map
    cycle = request.cycle
    solver_time_ms = request.solver_time_ms
    created_count = 0
    item_rows: list[DeliveryItem] = []
    for route in routes:
        vehicle_idx = int(route.get("vehicle_index", 0))
        if vehicle_idx < 0 or vehicle_idx >= len(vehicles):
            continue
        vehicle = vehicles[vehicle_idx]
        key = _delivery_idempotency_key(
            cycle=cycle,
            depot_id=int(depot.id),
            vehicle_id=int(vehicle.id),
            route=route,
        )
        delivery, created = Delivery.objects.get_or_create(
            idempotency_key=key,
            defaults={
                "depot": depot,
                "vehicle": vehicle,
                "planned_date": timezone.localdate(),
                "status": Delivery.Status.PLANNED,
                "total_distance_km": float(route.get("distance_km", 0.0)),
                "total_cost": float(route.get("cost", 0.0)),
                "route_json": route,
                "solver_time_ms": solver_time_ms,
                "created_by_planning_cycle": cycle,
            },
        )
        if not created:
            continue
        created_count += 1
        for stop in list(route.get("stops", [])):
            if not isinstance(stop, dict):
                continue
            facility_index = int(stop.get("facility_index", -1))
            facility_id = facility_index_map.get(facility_index)
            if facility_id is None:
                continue
            item_rows.append(
                DeliveryItem(
                    delivery=delivery,
                    facility_id=facility_id,
                    quantity=float(stop.get("demand", 0.0)),
                    planned_arrival=_planned_arrival_for_minutes(int(stop.get("arrival_min", 0))),
                    sequence=int(stop.get("sequence", 1)),
                ),
            )

    if item_rows:
        DeliveryItem.objects.bulk_create(item_rows)
    return created_count


def _fallback_reorder_point_from_recent_consumption(facility: Facility) -> float:
    recent_consumption = list(
        InventoryLog.objects.filter(facility_id=facility.id)
        .order_by("-timestamp")
        .values_list("consumption", flat=True)[:7],
    )
    if recent_consumption:
        avg_daily_consumption = sum(max(float(value), 0.0) for value in recent_consumption) / len(recent_consumption)
        lead_time_days = 3.0
        safety_margin = 1.1
        heuristic_threshold = avg_daily_consumption * lead_time_days * safety_margin
        return max(float(facility.min_safe_inventory), heuristic_threshold)
    return float(facility.min_safe_inventory)


def _resolve_reorder_point_for_forecast(
    facility: Facility,
    forecast_data: list[dict[str, object]],
    model_version: str,
) -> tuple[float, str]:
    p90_values: list[float] = []
    for step in forecast_data[:3]:
        raw_p90 = step.get("p90", 0.0)
        p90 = float(raw_p90) if isinstance(raw_p90, int | float | str) else 0.0
        p90_values.append(p90)
    lead_time_p90 = sum(p90_values) if p90_values else 0.0
    model_reorder_point = lead_time_p90 * 1.1
    is_fallback_forecast = model_version == "fallback-local"

    if not is_fallback_forecast and model_reorder_point > 0.0:
        return model_reorder_point, "model_forecast"

    dynamic_reorder_point = facility.dynamic_reorder_point
    if is_reliable_reorder_point(dynamic_reorder_point) and dynamic_reorder_point is not None:
        return float(dynamic_reorder_point), "preserved_prior"

    return _fallback_reorder_point_from_recent_consumption(facility), "heuristic_fallback"


class _AnomalyFacilityPayload(TypedDict):
    facility_id: int
    actual_consumption: float
    predicted_consumption: float
    rolling_std: float
    features: dict[str, float]


def _anomaly_facility_payload(facility_id: int) -> _AnomalyFacilityPayload | None:
    features = build_anomaly_features(facility_id)
    if features is None:
        return None

    latest_log = (
        InventoryLog.objects.filter(facility_id=facility_id).order_by("-timestamp").values("consumption").first()
    )
    latest_forecast = (
        Forecast.objects.filter(facility_id=facility_id).order_by("-created_at").only("predictions_json").first()
    )
    if latest_log is None or latest_forecast is None:
        return None

    recent_logs = list(
        InventoryLog.objects.filter(facility_id=facility_id)
        .order_by("-timestamp")
        .values_list("consumption", flat=True)[:7],
    )
    actual = float(latest_log.get("consumption") or 0.0)
    preds = latest_forecast.predictions_json if isinstance(latest_forecast.predictions_json, list) else []
    first_pred = preds[0] if preds else {}
    predicted = float(first_pred.get("p50", 0.0)) if isinstance(first_pred, dict) else 0.0

    if recent_logs:
        mean = sum(float(x) for x in recent_logs) / len(recent_logs)
        variance = sum((float(x) - mean) ** 2 for x in recent_logs) / len(recent_logs)
        rolling_std = max(variance**0.5, 1e-6)
    else:
        rolling_std = 1e-6

    return {
        "facility_id": int(facility_id),
        "actual_consumption": actual,
        "predicted_consumption": predicted,
        "rolling_std": rolling_std,
        "features": features,
    }


def _anomaly_remote_result(
    endpoint: str,
    payload: _AnomalyFacilityPayload,
    *,
    remote_enabled: bool,
) -> dict[str, object]:
    if remote_enabled:
        with httpx.Client(timeout=10.0) as client:
            try:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                result = response.json()
                return result if isinstance(result, dict) else {"is_anomaly": False}
            except httpx.HTTPError:
                _handle_remote_unavailable("anomaly", "request_failed")
                return {"is_anomaly": False, "anomaly_type": None, "confidence": None}
    return {"is_anomaly": False, "anomaly_type": None, "confidence": None}


def _persist_anomaly_alert_if_needed(payload: _AnomalyFacilityPayload, result: dict[str, object]) -> bool:
    if not bool(result.get("is_anomaly")):
        return False

    anomaly_type = str(result.get("anomaly_type") or "UNKNOWN")
    if anomaly_type not in {
        AnomalyAlert.AnomalyType.LEAK,
        AnomalyAlert.AnomalyType.THEFT,
        AnomalyAlert.AnomalyType.EQUIPMENT_DEGRADATION,
        AnomalyAlert.AnomalyType.DEMAND_SHIFT,
        AnomalyAlert.AnomalyType.SENSOR_FAULT,
    }:
        anomaly_type = AnomalyAlert.AnomalyType.SENSOR_FAULT

    confidence_raw = result.get("confidence")
    confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float, str)) else 0.0

    AnomalyAlert.objects.create(
        facility_id=payload["facility_id"],
        timestamp=timezone.now(),
        anomaly_type=anomaly_type,
        score=confidence,
        actual_consumption=payload["actual_consumption"],
        predicted_consumption=payload["predicted_consumption"],
    )
    return True


@_shared_task(queue="default")
def daily_tick() -> dict[str, Any]:
    """Dispatch the daily ingestion, ML, and planning workflow."""
    facility_ids = list(Facility.objects.filter(is_active=True).values_list("id", flat=True))
    if os.environ.get("FUELSENSE_ENABLE_DAILY_TICK", "0") != "1":
        return {"facility_count": len(facility_ids), "status": "disabled"}

    chord_fn: Any = chord
    ingest_task: Any = ingest_facility_data
    complete_task: Any = ingestion_complete
    forecast_task: Any = run_batch_forecasts
    anomaly_task: Any = run_batch_anomaly_detection
    drift_task: Any = check_all_drift
    planning_task: Any = run_planning_cycle

    ingest_workflow = chord_fn([ingest_task.si(int(fid)) for fid in facility_ids], complete_task.si())
    pipeline = (
        ingest_workflow
        | forecast_task.si(facility_ids)
        | anomaly_task.si(facility_ids)
        | drift_task.si()
        | planning_task.si()
    )
    pipeline.delay()
    logger.info("daily_tick dispatched", extra={"facility_count": len(facility_ids)})
    return {"facility_count": len(facility_ids), "status": "dispatched"}


@_shared_task(queue="default")
def ingest_hourly() -> dict[str, Any]:
    """Emit hourly ingestion heartbeat metadata."""
    count = Facility.objects.filter(is_active=True).count()
    logger.info("ingest_hourly invoked", extra={"active_facilities": count})
    return {"active_facilities": count}


@_shared_task(queue="default")
def ingest_facility_data(facility_id: int) -> dict[str, Any]:
    """Append one synthetic inventory snapshot for a facility."""
    facility = Facility.objects.get(id=facility_id)
    now = timezone.now()
    last_log = facility.inventory_logs.order_by("-timestamp").first()
    baseline_inventory = float(last_log.inventory_level) if last_log else float(facility.current_inventory)
    new_inventory = max(baseline_inventory - 1.0, 0.0)
    InventoryLog.objects.create(
        facility=facility,
        timestamp=now,
        inventory_level=new_inventory,
        consumption=1.0,
        temperature=25.0,
        wind_speed=2.0,
        solar_irradiance=500.0,
    )
    logger.info("ingest_facility_data invoked", extra={"facility_id": facility_id})
    return {"facility_id": facility_id, "timestamp": now.isoformat()}


@_shared_task(queue="default")
def ingestion_complete() -> dict[str, Any]:
    """Report completion stats for recent ingestion activity."""
    recent_logs = InventoryLog.objects.filter(timestamp__gte=timezone.now() - timedelta(hours=1)).count()
    logger.info("ingestion_complete invoked", extra={"recent_logs": recent_logs})
    return {"recent_logs": recent_logs}


@_shared_task(queue="default")
def run_batch_forecasts(facility_ids: list[int] | None = None) -> dict[str, Any]:
    """Generate and persist forecasts while updating reorder thresholds."""
    if facility_ids is None:
        facility_ids = list(Facility.objects.filter(is_active=True).values_list("id", flat=True))

    lookback = build_lookback_matrix(facility_ids)
    if lookback.shape[0] == 0:
        return {"facility_count": 0, "created": 0}

    payload = {
        "requests": [
            {"facility_id": int(fid), "lookback": lookback[idx].tolist()} for idx, fid in enumerate(facility_ids)
        ],
    }

    service_url = os.environ.get("FORECASTER_URL", "http://demand-forecaster:8001").rstrip("/")
    endpoint = f"{service_url}/predict/batch"
    remote_enabled = os.environ.get("FUELSENSE_ENABLE_REMOTE_FORECAST", "0") == "1"

    if remote_enabled:
        with httpx.Client(timeout=30.0) as client:
            try:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                body = response.json()
            except httpx.HTTPError:
                _handle_remote_unavailable("forecaster", "request_failed")
                # Keep task resilient when the external forecaster service is not reachable.
                body = {
                    "responses": [
                        {
                            "facility_id": int(fid),
                            "model_version": "fallback-local",
                            "forecast": [{"day": i + 1, "p10": 0.0, "p50": 0.0, "p90": 0.0} for i in range(14)],
                        }
                        for fid in facility_ids
                    ],
                }
    else:
        _handle_remote_unavailable("forecaster", "remote_disabled")
        body = {
            "responses": [
                {
                    "facility_id": int(fid),
                    "model_version": "fallback-local",
                    "forecast": [{"day": i + 1, "p10": 0.0, "p50": 0.0, "p90": 0.0} for i in range(14)],
                }
                for fid in facility_ids
            ],
        }

    responses = list(body.get("responses", []))
    created = 0
    for item in responses:
        facility_id = int(item["facility_id"])
        forecast_data = list(item.get("forecast", []))
        facility = Facility.objects.only("id", "dynamic_reorder_point", "min_safe_inventory").get(id=facility_id)
        model_version = str(item.get("model_version", "unknown"))
        Forecast.objects.create(
            facility_id=facility_id,
            model_version=model_version,
            horizon_days=len(forecast_data),
            predictions_json=forecast_data,
            rmse=None,
        )
        created += 1

        reorder_point, reorder_source = _resolve_reorder_point_for_forecast(facility, forecast_data, model_version)
        Facility.objects.filter(id=facility_id).update(dynamic_reorder_point=reorder_point)
        logger.info(
            "reorder point updated",
            extra={
                "facility_id": facility_id,
                "model_version": model_version,
                "reorder_point": reorder_point,
                "reorder_source": reorder_source,
            },
        )

    logger.info(
        "run_batch_forecasts completed",
        extra={"facility_count": len(facility_ids), "created_count": created},
    )
    return {"facility_count": len(facility_ids), "created": created}


@_shared_task(queue="default")
def run_batch_anomaly_detection(facility_ids: list[int] | None = None) -> dict[str, Any]:
    """Run anomaly detection for facilities and persist resulting alerts."""
    if facility_ids is None:
        facility_ids = list(Facility.objects.filter(is_active=True).values_list("id", flat=True))

    service_url = os.environ.get("ANOMALY_URL", "http://anomaly-detector:8002").rstrip("/")
    endpoint = f"{service_url}/detect"
    remote_enabled = os.environ.get("FUELSENSE_ENABLE_REMOTE_ANOMALY", "0") == "1"
    if not remote_enabled:
        _handle_remote_unavailable("anomaly", "remote_disabled")

    anomaly_count = 0
    for facility_id in facility_ids:
        payload = _anomaly_facility_payload(int(facility_id))
        if payload is None:
            continue
        result = _anomaly_remote_result(endpoint, payload, remote_enabled=remote_enabled)
        if _persist_anomaly_alert_if_needed(payload, result):
            anomaly_count += 1

    logger.info(
        "run_batch_anomaly_detection completed",
        extra={"facility_count": len(facility_ids), "anomaly_count": anomaly_count},
    )
    return {"facility_count": len(facility_ids), "anomaly_count": anomaly_count}


@_shared_task(queue="training")
def check_all_drift() -> dict[str, Any]:
    """Evaluate model drift and enqueue retraining for drifting facilities."""
    drift_payload = build_drift_data()
    summary = DriftMonitor().check_all_facilities(drift_payload)
    retrain_task: Any = retrain_model
    for facility_id in summary["retrain_facility_ids"]:
        retrain_task.delay(int(facility_id), "DEMAND_FORECAST")

    logger.info(
        "check_all_drift completed",
        extra={
            "facilities_checked": summary["facilities_checked"],
            "retrain_facility_ids": summary["retrain_facility_ids"],
        },
    )
    summary["models"] = summary["facilities_checked"]
    return summary


@_shared_task(queue="training")
def retrain_model(facility_id: int | None, model_type: str) -> dict[str, Any]:
    """Train and potentially promote a model for a facility scope."""
    if os.environ.get("FUELSENSE_ENABLE_TRAINING_TASKS", "0") != "1":
        return {"facility_id": facility_id, "model_type": model_type, "status": "disabled"}

    demand_forecast_type = str(ModelRegistry.ModelType.DEMAND_FORECAST)
    anomaly_detector_type = str(ModelRegistry.ModelType.ANOMALY_DETECTOR)

    if model_type == demand_forecast_type:
        return _retrain_demand_forecast_model(facility_id)

    if model_type == anomaly_detector_type:
        return _retrain_anomaly_detector_model()

    logger.info("retrain_model skipped for unsupported model type", extra={"model_type": model_type})
    return {"facility_id": facility_id, "model_type": model_type, "status": "skipped"}


def _retrain_demand_forecast_model(facility_id: int | None) -> dict[str, Any]:
    model_type = ModelRegistry.ModelType.DEMAND_FORECAST

    if facility_id is None:
        return {
            "facility_id": facility_id,
            "model_type": model_type,
            "status": "skipped",
            "error": "facility_id_required",
        }

    dataset = extract_training_data(facility_id)
    trainer = ForecastTrainer()
    try:
        result = trainer.train_and_register(
            facility_id=facility_id,
            dataset=TrainingDataset(
                train_data=dataset["train_data"],
                train_targets=dataset["train_targets"],
                val_data=dataset["val_data"],
                val_targets=dataset["val_targets"],
                test_data=dataset["test_data"],
                test_targets=dataset["test_targets"],
            ),
        )
    except RECOVERABLE_TASK_EXCEPTIONS as exc:
        logger.warning("retrain_model failed", extra={"facility_id": facility_id, "error": str(exc)})
        return {
            "facility_id": facility_id,
            "model_type": model_type,
            "status": "failed",
            "error": str(exc),
        }

    current_active = (
        ModelRegistry.objects.filter(model_type=model_type, facility_id=facility_id, is_active=True)
        .order_by("-version")
        .first()
    )
    previous_rmse = (
        float(current_active.validation_rmse) if current_active and current_active.validation_rmse else float("inf")
    )
    candidate_validation_rmse = float(result.get("validation_rmse", result["test_rmse"]))
    improved = candidate_validation_rmse < previous_rmse

    if improved:
        ModelRegistry.objects.filter(model_type=model_type, facility_id=facility_id, is_active=True).update(
            is_active=False,
        )
        max_version = (
            ModelRegistry.objects.filter(model_type=model_type, facility_id=facility_id)
            .aggregate(v=Max("version"))
            .get("v")
            or 0
        )
        ModelRegistry.objects.create(
            model_type=model_type,
            facility_id=facility_id,
            mlflow_run_id=str(result["run_id"]),
            version=int(max_version) + 1,
            is_active=True,
            trained_at=timezone.now(),
            training_rmse=float(result.get("training_rmse", candidate_validation_rmse)),
            validation_rmse=candidate_validation_rmse,
            drift_ratio=1.0,
            last_drift_check=timezone.now(),
        )

    status = "promoted" if improved else "rejected"
    logger.info(
        "retrain_model completed",
        extra={"facility_id": facility_id, "model_type": model_type, "status": status, "rmse": result["test_rmse"]},
    )
    return {
        "facility_id": facility_id,
        "model_type": model_type,
        "status": status,
        "test_rmse": float(result["test_rmse"]),
        "run_id": str(result["run_id"]),
    }


def _retrain_anomaly_detector_model() -> dict[str, Any]:
    model_type = ModelRegistry.ModelType.ANOMALY_DETECTOR

    trainer = AnomalyTrainer()
    try:
        result = trainer.train_and_register()
    except RECOVERABLE_TASK_EXCEPTIONS as exc:
        logger.warning("anomaly retrain failed", extra={"error": str(exc)})
        return {
            "facility_id": None,
            "model_type": model_type,
            "status": "failed",
            "error": str(exc),
        }

    ModelRegistry.objects.filter(model_type=model_type, facility__isnull=True, is_active=True).update(is_active=False)
    max_version = (
        ModelRegistry.objects.filter(model_type=model_type, facility__isnull=True).aggregate(v=Max("version")).get("v")
        or 0
    )
    ModelRegistry.objects.create(
        model_type=model_type,
        facility=None,
        mlflow_run_id=str(result["run_id"]),
        version=int(max_version) + 1,
        is_active=True,
        trained_at=timezone.now(),
        training_rmse=None,
        validation_rmse=None,
        drift_ratio=1.0,
        last_drift_check=None,
    )

    logger.info(
        "anomaly retrain completed",
        extra={
            "model_type": model_type,
            "status": "promoted",
            "f1": float(result.get("f1", 0.0)),
            "classifier_accuracy": float(result.get("classifier_accuracy", 0.0)),
        },
    )
    return {
        "facility_id": None,
        "model_type": model_type,
        "status": "promoted",
        "run_id": str(result["run_id"]),
        "f1": float(result.get("f1", 0.0)),
        "classifier_accuracy": float(result.get("classifier_accuracy", 0.0)),
    }


@dataclass
class _PlanningTotals:
    deliveries_created: int = 0
    total_distance: float = 0.0
    total_cost: float = 0.0
    total_solver_time: float = 0.0
    total_baseline: float = 0.0

    def add_result(self, result: dict[str, Any], created_count: int) -> None:
        self.deliveries_created += created_count
        self.total_distance += float(result.get("total_distance_km", 0.0))
        self.total_cost += float(result.get("total_cost", 0.0))
        self.total_solver_time += float(result.get("solver_time_ms", 0.0))
        self.total_baseline += float(result.get("baseline_cost", 0.0))


@dataclass(frozen=True)
class _OptimizerExecutionConfig:
    parallel_enabled: bool
    max_workers: int
    failure_log_message: str
    cycle_id: int | None
    entity_key: str


def _mark_cycle_running(cycle_id: int) -> PlanningCycle | None:
    try:
        with transaction.atomic():
            cycle = PlanningCycle.objects.select_for_update().get(id=cycle_id)
            if cycle.status != PlanningCycle.ExecutionStatus.QUEUED:
                return None
            cycle.status = PlanningCycle.ExecutionStatus.RUNNING
            cycle.started_at = timezone.now()
            cycle.save(update_fields=["status", "started_at"])
            return cycle
    except PlanningCycle.DoesNotExist:
        return None


def _resolve_cycle_for_execution(
    cycle_id: int | None,
) -> tuple[PlanningCycle | None, dict[str, Any] | None]:
    if cycle_id is None:
        return None, None
    cycle = _mark_cycle_running(cycle_id)
    if cycle is not None:
        return cycle, None
    if not PlanningCycle.objects.filter(id=cycle_id).exists():
        return None, {"cycle_id": cycle_id, "error": "cycle_not_found"}
    current_status = PlanningCycle.objects.only("status").get(id=cycle_id).status
    return None, {"cycle_id": cycle_id, "status": current_status, "already_processed": True}


def _complete_cycle_with_no_queue(cycle: PlanningCycle) -> None:
    cycle.facilities_in_queue = 0
    cycle.deliveries_created = 0
    cycle.total_distance_km = 0.0
    cycle.total_cost = 0.0
    cycle.solver_time_ms = 0.0
    cycle.baseline_cost = 0.0
    cycle.cost_reduction_pct = 0.0
    cycle.status = PlanningCycle.ExecutionStatus.COMPLETED
    cycle.completed_at = timezone.now()
    cycle.save(
        update_fields=[
            "facilities_in_queue",
            "deliveries_created",
            "total_distance_km",
            "total_cost",
            "solver_time_ms",
            "baseline_cost",
            "cost_reduction_pct",
            "status",
            "completed_at",
        ],
    )


def _mark_cycle_failed(cycle: PlanningCycle, facilities_in_queue: int) -> None:
    cycle.facilities_in_queue = facilities_in_queue
    cycle.deliveries_created = 0
    cycle.total_distance_km = 0.0
    cycle.total_cost = 0.0
    cycle.solver_time_ms = 0.0
    cycle.baseline_cost = 0.0
    cycle.cost_reduction_pct = 0.0
    cycle.status = PlanningCycle.ExecutionStatus.FAILED
    cycle.completed_at = timezone.now()
    cycle.save(
        update_fields=[
            "facilities_in_queue",
            "deliveries_created",
            "total_distance_km",
            "total_cost",
            "solver_time_ms",
            "baseline_cost",
            "cost_reduction_pct",
            "status",
            "completed_at",
        ],
    )


def _build_depot_jobs(queued_facilities: list[Facility]) -> dict[int, dict[str, Any]]:
    assignments = (
        DepotFacilityAssignment.objects.filter(facility_id__in=[int(f.id) for f in queued_facilities])
        .select_related("depot", "facility")
        .order_by("depot_id", "facility_id")
    )
    facilities_by_depot: dict[int, list[Any]] = {}
    depot_map: dict[int, Any] = {}
    for assignment in assignments:
        depot_map[int(assignment.depot_id)] = assignment.depot
        facilities_by_depot.setdefault(int(assignment.depot_id), []).append(assignment.facility)

    depot_jobs: dict[int, dict[str, Any]] = {}
    for depot_id, facilities in facilities_by_depot.items():
        depot = depot_map[depot_id]
        payload, facility_index_map = build_optimizer_request(depot, facilities)
        depot_jobs[depot_id] = {
            "depot": depot,
            "facilities": facilities,
            "payload": payload,
            "facility_index_map": facility_index_map,
        }
    return depot_jobs


def _env_max_workers(env_var: str, item_count: int) -> int:
    max_workers_env = int(os.environ.get(env_var, str(_default_parallel_workers())))
    return max(1, min(max_workers_env, item_count or 1))


def _execute_optimizer_jobs(
    jobs: dict[int, dict[str, Any]],
    config: _OptimizerExecutionConfig,
) -> tuple[dict[int, dict[str, Any]], int]:
    results: dict[int, dict[str, Any]] = {}
    failed = 0
    optimizer_client = _build_remote_optimizer_client()
    try:
        if config.parallel_enabled and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                futures = {
                    executor.submit(_run_optimizer, job["payload"], optimizer_client): entity_id
                    for entity_id, job in jobs.items()
                }
                for future in as_completed(futures):
                    entity_id = futures[future]
                    try:
                        results[entity_id] = future.result()
                    except RECOVERABLE_TASK_EXCEPTIONS:
                        failed += 1
                        logger.exception(
                            config.failure_log_message,
                            extra={config.entity_key: entity_id, "cycle_id": config.cycle_id},
                        )
        else:
            for entity_id, job in jobs.items():
                try:
                    results[entity_id] = _run_optimizer(job["payload"], optimizer_client=optimizer_client)
                except RECOVERABLE_TASK_EXCEPTIONS:
                    failed += 1
                    logger.exception(
                        config.failure_log_message,
                        extra={config.entity_key: entity_id, "cycle_id": config.cycle_id},
                    )
    finally:
        if optimizer_client is not None and hasattr(optimizer_client, "close"):
            optimizer_client.close()
    return results, failed


def _persist_depot_results(
    *,
    cycle: PlanningCycle | None,
    cycle_id: int | None,
    depot_jobs: dict[int, dict[str, Any]],
    depot_results: dict[int, dict[str, Any]],
) -> tuple[_PlanningTotals, int, int]:
    totals = _PlanningTotals()
    planning_cycles = 0
    failed_depots = 0

    for depot_id, result in depot_results.items():
        job = depot_jobs[depot_id]
        depot = job["depot"]
        facilities = job["facilities"]
        facility_index_map = job["facility_index_map"]
        routes = [r for r in list(result.get("routes", [])) if isinstance(r, dict)]
        try:
            with transaction.atomic():
                cycle_for_delivery = cycle
                if cycle_for_delivery is None:
                    now = timezone.now()
                    cycle_for_delivery = PlanningCycle.objects.create(
                        trigger_type=PlanningCycle.TriggerType.SCHEDULED,
                        status=PlanningCycle.ExecutionStatus.COMPLETED,
                        started_at=now,
                        completed_at=now,
                        facilities_in_queue=len(facilities),
                        deliveries_created=0,
                        total_distance_km=float(result.get("total_distance_km", 0.0)),
                        total_cost=float(result.get("total_cost", 0.0)),
                        solver_time_ms=float(result.get("solver_time_ms", 0.0)),
                        baseline_cost=float(result.get("baseline_cost", 0.0)),
                        cost_reduction_pct=float(result.get("cost_reduction_pct", 0.0)),
                    )
                vehicles = list(depot.vehicles.filter(is_available=True).order_by("id"))
                created_for_depot = _materialize_delivery_routes(
                    _DeliveryMaterializationRequest(
                        depot=depot,
                        routes=routes,
                        vehicles=vehicles,
                        facility_index_map=facility_index_map,
                        cycle=cycle_for_delivery,
                        solver_time_ms=float(result.get("solver_time_ms", 0.0)),
                    ),
                )
                if cycle is None:
                    cycle_for_delivery.deliveries_created = created_for_depot
                    cycle_for_delivery.save(update_fields=["deliveries_created"])
                    planning_cycles += 1
                totals.add_result(result, created_for_depot)
        except RECOVERABLE_TASK_EXCEPTIONS:
            failed_depots += 1
            logger.exception(
                "persisting planning results failed",
                extra={"depot_id": depot_id, "cycle_id": cycle_id},
            )
    return totals, planning_cycles, failed_depots


def _update_cycle_with_planning_totals(
    *,
    cycle: PlanningCycle,
    queued_count: int,
    failed_depots: int,
    totals: _PlanningTotals,
) -> None:
    reduction = 0.0
    if totals.total_baseline > 0.0:
        reduction = max((totals.total_baseline - totals.total_cost) / totals.total_baseline * 100.0, 0.0)
    cycle.facilities_in_queue = queued_count
    cycle.deliveries_created = totals.deliveries_created
    cycle.total_distance_km = totals.total_distance
    cycle.total_cost = totals.total_cost
    cycle.solver_time_ms = totals.total_solver_time
    cycle.baseline_cost = totals.total_baseline
    cycle.cost_reduction_pct = reduction
    cycle.status = (
        PlanningCycle.ExecutionStatus.FAILED
        if failed_depots > 0 and totals.deliveries_created == 0
        else PlanningCycle.ExecutionStatus.COMPLETED
    )
    cycle.completed_at = timezone.now()
    cycle.save(
        update_fields=[
            "facilities_in_queue",
            "deliveries_created",
            "total_distance_km",
            "total_cost",
            "solver_time_ms",
            "baseline_cost",
            "cost_reduction_pct",
            "status",
            "completed_at",
        ],
    )


def _build_emergency_jobs(
    facility_ids: list[int],
) -> tuple[dict[int, dict[str, Any]], int]:
    assignments = (
        DepotFacilityAssignment.objects.filter(facility_id__in=facility_ids)
        .select_related("depot", "facility")
        .order_by("facility_id")
    )
    assignment_map: dict[int, DepotFacilityAssignment] = {int(a.facility_id): a for a in assignments}
    missing_facility_ids = [int(fid) for fid in facility_ids if int(fid) not in assignment_map]

    facility_jobs: dict[int, dict[str, Any]] = {}
    for facility_id, assignment in assignment_map.items():
        payload, facility_index_map = build_optimizer_request(assignment.depot, [assignment.facility])
        facility_jobs[facility_id] = {
            "assignment": assignment,
            "payload": payload,
            "facility_index_map": facility_index_map,
        }
    return facility_jobs, len(missing_facility_ids)


def _materialize_one_emergency_result(
    *,
    cycle: PlanningCycle,
    job: dict[str, Any],
    result: dict[str, Any],
) -> int:
    assignment = job["assignment"]
    routes = [r for r in list(result.get("routes", [])) if isinstance(r, dict)]
    vehicles = list(assignment.depot.vehicles.filter(is_available=True).order_by("id"))
    return _materialize_delivery_routes(
        _DeliveryMaterializationRequest(
            depot=assignment.depot,
            routes=routes,
            vehicles=vehicles,
            facility_index_map=job["facility_index_map"],
            cycle=cycle,
            solver_time_ms=float(result.get("solver_time_ms", 0.0)),
        ),
    )


def _update_emergency_cycle(
    *,
    cycle_id: int,
    facility_count: int,
    failed_facilities: int,
    totals: _PlanningTotals,
) -> None:
    reduction = 0.0
    if totals.total_baseline > 0.0:
        reduction = max((totals.total_baseline - totals.total_cost) / totals.total_baseline * 100.0, 0.0)
    status = (
        PlanningCycle.ExecutionStatus.FAILED
        if failed_facilities > 0 and totals.deliveries_created == 0
        else PlanningCycle.ExecutionStatus.COMPLETED
    )
    PlanningCycle.objects.filter(id=cycle_id).update(
        facilities_in_queue=facility_count,
        deliveries_created=totals.deliveries_created,
        total_distance_km=totals.total_distance,
        total_cost=totals.total_cost,
        solver_time_ms=totals.total_solver_time,
        baseline_cost=totals.total_baseline,
        cost_reduction_pct=reduction,
        status=status,
        completed_at=timezone.now(),
    )


def _emergency_fail_and_return(
    cycle_id: int,
    facility_count: int,
    failed_facilities: int,
    error: str,
) -> dict[str, Any]:
    PlanningCycle.objects.filter(id=cycle_id).update(
        facilities_in_queue=facility_count,
        deliveries_created=0,
        total_distance_km=0.0,
        total_cost=0.0,
        solver_time_ms=0.0,
        baseline_cost=0.0,
        cost_reduction_pct=0.0,
        status=PlanningCycle.ExecutionStatus.FAILED,
        completed_at=timezone.now(),
    )
    return {
        "cycle_id": cycle_id,
        "facilities_in_queue": facility_count,
        "deliveries_created": 0,
        "failed_facilities": failed_facilities,
        "partial_success": False,
        "status": PlanningCycle.ExecutionStatus.FAILED,
        "error": error,
    }


@_shared_task(queue="planning")
def run_planning_cycle(cycle_id: int | None = None) -> dict[str, Any]:
    """Execute scheduled or manual planning and persist delivery outputs."""
    cycle, early_response = _resolve_cycle_for_execution(cycle_id)
    if early_response is not None:
        return early_response

    trigger_type = cycle.trigger_type if cycle else PlanningCycle.TriggerType.SCHEDULED
    try:
        queue_start = perf_counter()
        queued_facilities = list(filter_below_reorder(Facility.objects.filter(is_active=True)).order_by("id"))
        observe_planning_stage("queue_selection", perf_counter() - queue_start, trigger_type)
        if not queued_facilities:
            if cycle is not None:
                _complete_cycle_with_no_queue(cycle)
            return {"queued": 0, "planning_cycles": 0, "deliveries_created": 0}

        request_build_start = perf_counter()
        depot_jobs = _build_depot_jobs(queued_facilities)
        observe_planning_stage("request_build", perf_counter() - request_build_start, trigger_type)

        optimize_start = perf_counter()
        parallel_enabled = os.environ.get("FUELSENSE_PLANNING_PARALLEL_DEPOTS", "1") == "1"
        max_workers = _env_max_workers("FUELSENSE_PLANNING_PARALLEL_WORKERS", len(depot_jobs))
        depot_results, failed_depots = _execute_optimizer_jobs(
            depot_jobs,
            _OptimizerExecutionConfig(
                parallel_enabled=parallel_enabled,
                max_workers=max_workers,
                failure_log_message="optimizer execution failed",
                cycle_id=cycle_id,
                entity_key="depot_id",
            ),
        )
        observe_planning_stage("optimizer_call", perf_counter() - optimize_start, trigger_type)

        strict_mode = os.environ.get("FUELSENSE_PLANNING_STRICT_DEPOT_SUCCESS", "1") == "1"
        if strict_mode and failed_depots > 0:
            if cycle is not None:
                _mark_cycle_failed(cycle, len(queued_facilities))
            return {
                "queued": len(queued_facilities),
                "planning_cycles": 0,
                "deliveries_created": 0,
                "failed_depots": failed_depots,
                "partial_success": False,
                "status": PlanningCycle.ExecutionStatus.FAILED,
                "error": "strict_mode_depot_failure",
            }

        persist_start = perf_counter()
        totals, planning_cycles, persist_failures = _persist_depot_results(
            cycle=cycle,
            cycle_id=cycle_id,
            depot_jobs=depot_jobs,
            depot_results=depot_results,
        )
        failed_depots += persist_failures
        observe_planning_stage("persist", perf_counter() - persist_start, trigger_type)

        if cycle is not None:
            _update_cycle_with_planning_totals(
                cycle=cycle,
                queued_count=len(queued_facilities),
                failed_depots=failed_depots,
                totals=totals,
            )
            planning_cycles = 1

        logger.info(
            "run_planning_cycle completed",
            extra={
                "queued": len(queued_facilities),
                "planning_cycles": planning_cycles,
                "deliveries_created": totals.deliveries_created,
                "failed_depots": failed_depots,
                "cycle_id": cycle_id,
            },
        )
        return {
            "queued": len(queued_facilities),
            "planning_cycles": planning_cycles,
            "deliveries_created": totals.deliveries_created,
            "failed_depots": failed_depots,
            "partial_success": failed_depots > 0 and totals.deliveries_created > 0,
        }
    except RECOVERABLE_TASK_EXCEPTIONS as exc:
        if cycle is not None:
            cycle.status = PlanningCycle.ExecutionStatus.FAILED
            cycle.completed_at = timezone.now()
            cycle.save(update_fields=["status", "completed_at"])
        return {
            "cycle_id": cycle_id,
            "error": str(exc),
            "status": PlanningCycle.ExecutionStatus.FAILED if cycle is not None else "failed",
        }


@_shared_task(queue="planning")
def run_emergency_planning_cycle(cycle_id: int, facility_ids: list[int]) -> dict[str, Any]:
    """Execute emergency planning for provided facilities under one cycle."""
    cycle, early_response = _resolve_cycle_for_execution(cycle_id)
    if early_response is not None:
        return early_response
    if cycle is None:
        return {"cycle_id": cycle_id, "error": "cycle_resolution_failed"}

    trigger_type = PlanningCycle.TriggerType.EMERGENCY
    totals = _PlanningTotals()
    failed_facilities = 0
    strict_mode = os.environ.get("FUELSENSE_EMERGENCY_STRICT_FACILITY_SUCCESS", "1") == "1"

    try:
        request_build_start = perf_counter()
        facility_jobs, missing_count = _build_emergency_jobs(facility_ids)
        failed_facilities += missing_count
        observe_planning_stage("request_build", perf_counter() - request_build_start, trigger_type)

        optimizer_stage_start = perf_counter()
        parallel_enabled = os.environ.get("FUELSENSE_EMERGENCY_PARALLEL", "1") == "1"
        max_workers = _env_max_workers("FUELSENSE_EMERGENCY_PARALLEL_WORKERS", len(facility_jobs))
        facility_results, optimizer_failures = _execute_optimizer_jobs(
            facility_jobs,
            _OptimizerExecutionConfig(
                parallel_enabled=parallel_enabled,
                max_workers=max_workers,
                failure_log_message="emergency optimizer execution failed",
                cycle_id=cycle_id,
                entity_key="facility_id",
            ),
        )
        failed_facilities += optimizer_failures
        observe_planning_stage("optimizer_call", perf_counter() - optimizer_stage_start, trigger_type)

        if strict_mode and failed_facilities > 0:
            return _emergency_fail_and_return(
                cycle_id,
                len(facility_ids),
                failed_facilities,
                "strict_mode_facility_failure",
            )

        persist_start = perf_counter()
        if strict_mode:
            try:
                with transaction.atomic():
                    for facility_id, result in facility_results.items():
                        created_count = _materialize_one_emergency_result(
                            cycle=cycle,
                            job=facility_jobs[facility_id],
                            result=result,
                        )
                        totals.add_result(result, created_count)
            except RECOVERABLE_TASK_EXCEPTIONS:
                logger.exception("emergency persistence failed", extra={"cycle_id": cycle_id})
                return _emergency_fail_and_return(
                    cycle_id,
                    len(facility_ids),
                    max(failed_facilities, 1),
                    "strict_mode_persist_failure",
                )
        else:
            for facility_id, result in facility_results.items():
                try:
                    with transaction.atomic():
                        created_count = _materialize_one_emergency_result(
                            cycle=cycle,
                            job=facility_jobs[facility_id],
                            result=result,
                        )
                        totals.add_result(result, created_count)
                except RECOVERABLE_TASK_EXCEPTIONS:
                    failed_facilities += 1
                    logger.exception(
                        "emergency persistence failed",
                        extra={"facility_id": facility_id, "cycle_id": cycle_id},
                    )
        observe_planning_stage("persist", perf_counter() - persist_start, trigger_type)

        _update_emergency_cycle(
            cycle_id=cycle_id,
            facility_count=len(facility_ids),
            failed_facilities=failed_facilities,
            totals=totals,
        )
        return {
            "cycle_id": cycle_id,
            "facilities_in_queue": len(facility_ids),
            "deliveries_created": totals.deliveries_created,
            "failed_facilities": failed_facilities,
            "partial_success": failed_facilities > 0 and totals.deliveries_created > 0,
        }
    except RECOVERABLE_TASK_EXCEPTIONS as exc:
        PlanningCycle.objects.filter(id=cycle_id).update(
            status=PlanningCycle.ExecutionStatus.FAILED,
            completed_at=timezone.now(),
        )
        return {"cycle_id": cycle_id, "error": str(exc), "status": PlanningCycle.ExecutionStatus.FAILED}


@_shared_task(queue="planning")
def trigger_emergency_delivery(
    facility_id: int,
    cycle_id: int | None = None,
    optimizer_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Plan and materialize an emergency delivery for a single facility."""
    assignment = (
        DepotFacilityAssignment.objects.filter(facility_id=facility_id).select_related("depot", "facility").first()
    )
    if assignment is None:
        return {"facility_id": facility_id, "exists": False}

    payload, facility_index_map = build_optimizer_request(assignment.depot, [assignment.facility])
    result = _run_optimizer(payload, optimizer_client=optimizer_client)
    routes = [r for r in list(result.get("routes", [])) if isinstance(r, dict)]

    if cycle_id is None:
        now = timezone.now()
        cycle: PlanningCycle | None = PlanningCycle.objects.create(
            trigger_type=PlanningCycle.TriggerType.EMERGENCY,
            status=PlanningCycle.ExecutionStatus.COMPLETED,
            started_at=now,
            completed_at=now,
            facilities_in_queue=1,
            deliveries_created=len(routes),
            total_distance_km=float(result.get("total_distance_km", 0.0)),
            total_cost=float(result.get("total_cost", 0.0)),
            solver_time_ms=float(result.get("solver_time_ms", 0.0)),
            baseline_cost=float(result.get("baseline_cost", 0.0)),
            cost_reduction_pct=float(result.get("cost_reduction_pct", 0.0)),
        )
    else:
        cycle = PlanningCycle.objects.filter(id=cycle_id).first()

    vehicles = list(assignment.depot.vehicles.filter(is_available=True).order_by("id"))
    delivery_count = _materialize_delivery_routes(
        _DeliveryMaterializationRequest(
            depot=assignment.depot,
            routes=routes,
            vehicles=vehicles,
            facility_index_map=facility_index_map,
            cycle=cycle,
            solver_time_ms=float(result.get("solver_time_ms", 0.0)),
        ),
    )

    logger.info(
        "trigger_emergency_delivery completed",
        extra={"facility_id": facility_id, "deliveries_created": delivery_count, "cycle_id": cycle_id},
    )
    return {
        "facility_id": facility_id,
        "exists": True,
        "deliveries_created": delivery_count,
        "total_distance_km": float(result.get("total_distance_km", 0.0)),
        "total_cost": float(result.get("total_cost", 0.0)),
        "solver_time_ms": float(result.get("solver_time_ms", 0.0)),
        "baseline_cost": float(result.get("baseline_cost", 0.0)),
    }
