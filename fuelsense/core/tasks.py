"""Celery task definitions for FuelSense core app."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false

from __future__ import annotations

import logging
import os
from datetime import timedelta
from datetime import datetime
from typing import Any

import httpx
from celery import chord, shared_task
from django.db.models import F, Max
from django.utils import timezone

from fuelsense.core.features import (
    build_anomaly_features,
    build_drift_data,
    build_lookback_matrix,
    extract_training_data,
)
from fuelsense.core.models import (
    AnomalyAlert,
    Delivery,
    DeliveryItem,
    DepotFacilityAssignment,
    Facility,
    Forecast,
    InventoryLog,
    ModelRegistry,
    PlanningCycle,
)
from fuelsense.core.routing import build_optimizer_request
from ml_pipeline.drift import DriftMonitor

logger = logging.getLogger(__name__)


def _remote_required() -> bool:
    return os.environ.get("FUELSENSE_REQUIRE_REMOTE_SERVICES", "0") == "1"


def _handle_remote_unavailable(service_name: str, reason: str) -> None:
    if _remote_required():
        raise RuntimeError(f"{service_name} unavailable ({reason}) and FUELSENSE_REQUIRE_REMOTE_SERVICES=1")
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
                    }
                ],
                "distance_km": route_distance,
                "cost": route_cost,
            }
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


def _run_optimizer(payload: dict[str, object]) -> dict[str, Any]:
    service_url = os.environ.get("OPTIMIZER_URL", "http://route-optimizer:8003").rstrip("/")
    endpoint = f"{service_url}/optimize"
    remote_enabled = os.environ.get("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "0") == "1"

    if not remote_enabled:
        _handle_remote_unavailable("optimizer", "remote_disabled")
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


@shared_task(queue="default")
def daily_tick() -> dict[str, Any]:
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


@shared_task(queue="default")
def ingest_hourly() -> dict[str, Any]:
    count = Facility.objects.filter(is_active=True).count()
    logger.info("ingest_hourly invoked", extra={"active_facilities": count})
    return {"active_facilities": count}


@shared_task(queue="default")
def ingest_facility_data(facility_id: int) -> dict[str, Any]:
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


@shared_task(queue="default")
def ingestion_complete() -> dict[str, Any]:
    recent_logs = InventoryLog.objects.filter(timestamp__gte=timezone.now() - timedelta(hours=1)).count()
    logger.info("ingestion_complete invoked", extra={"recent_logs": recent_logs})
    return {"recent_logs": recent_logs}


@shared_task(queue="default")
def run_batch_forecasts(facility_ids: list[int] | None = None) -> dict[str, Any]:
    if facility_ids is None:
        facility_ids = list(Facility.objects.filter(is_active=True).values_list("id", flat=True))

    lookback = build_lookback_matrix(facility_ids)
    if lookback.shape[0] == 0:
        return {"facility_count": 0, "created": 0}

    payload = {
        "requests": [
            {"facility_id": int(fid), "lookback": lookback[idx].tolist()} for idx, fid in enumerate(facility_ids)
        ]
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
                    ]
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
            ]
        }

    responses = list(body.get("responses", []))
    created = 0
    for item in responses:
        facility_id = int(item["facility_id"])
        forecast_data = list(item.get("forecast", []))
        Forecast.objects.create(
            facility_id=facility_id,
            model_version=str(item.get("model_version", "unknown")),
            horizon_days=len(forecast_data),
            predictions_json=forecast_data,
            rmse=None,
        )
        created += 1

        p90_values = [float(step.get("p90", 0.0)) for step in forecast_data[:3]]
        lead_time_p90 = sum(p90_values) if p90_values else 0.0
        reorder_point = lead_time_p90 * 1.1
        Facility.objects.filter(id=facility_id).update(dynamic_reorder_point=reorder_point)

    logger.info(
        "run_batch_forecasts completed",
        extra={"facility_count": len(facility_ids), "created_count": created},
    )
    return {"facility_count": len(facility_ids), "created": created}


@shared_task(queue="default")
def run_batch_anomaly_detection(facility_ids: list[int] | None = None) -> dict[str, Any]:
    if facility_ids is None:
        facility_ids = list(Facility.objects.filter(is_active=True).values_list("id", flat=True))

    service_url = os.environ.get("ANOMALY_URL", "http://anomaly-detector:8002").rstrip("/")
    endpoint = f"{service_url}/detect"
    remote_enabled = os.environ.get("FUELSENSE_ENABLE_REMOTE_ANOMALY", "0") == "1"
    if not remote_enabled:
        _handle_remote_unavailable("anomaly", "remote_disabled")

    anomaly_count = 0
    for facility_id in facility_ids:
        features = build_anomaly_features(int(facility_id))
        if features is None:
            continue

        latest_log = (
            InventoryLog.objects.filter(facility_id=facility_id).order_by("-timestamp").values("consumption").first()
        )
        recent_logs = list(
            InventoryLog.objects.filter(facility_id=facility_id)
            .order_by("-timestamp")
            .values_list("consumption", flat=True)[:7]
        )
        latest_forecast = (
            Forecast.objects.filter(facility_id=facility_id).order_by("-created_at").only("predictions_json").first()
        )
        if latest_log is None or latest_forecast is None:
            continue

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

        payload = {
            "facility_id": int(facility_id),
            "actual_consumption": actual,
            "predicted_consumption": predicted,
            "rolling_std": rolling_std,
            "features": features,
        }

        if remote_enabled:
            with httpx.Client(timeout=10.0) as client:
                try:
                    response = client.post(endpoint, json=payload)
                    response.raise_for_status()
                    result = response.json()
                except httpx.HTTPError:
                    _handle_remote_unavailable("anomaly", "request_failed")
                    result = {"is_anomaly": False, "anomaly_type": None, "confidence": None}
        else:
            result = {"is_anomaly": False, "anomaly_type": None, "confidence": None}

        if bool(result.get("is_anomaly")):
            anomaly_type = str(result.get("anomaly_type") or "UNKNOWN")
            if anomaly_type not in {
                AnomalyAlert.AnomalyType.LEAK,
                AnomalyAlert.AnomalyType.THEFT,
                AnomalyAlert.AnomalyType.EQUIPMENT_DEGRADATION,
                AnomalyAlert.AnomalyType.DEMAND_SHIFT,
                AnomalyAlert.AnomalyType.SENSOR_FAULT,
            }:
                anomaly_type = AnomalyAlert.AnomalyType.SENSOR_FAULT

            AnomalyAlert.objects.create(
                facility_id=int(facility_id),
                timestamp=timezone.now(),
                anomaly_type=anomaly_type,
                score=float(result.get("confidence") or 0.0),
                actual_consumption=actual,
                predicted_consumption=predicted,
            )
            anomaly_count += 1

    logger.info(
        "run_batch_anomaly_detection completed",
        extra={"facility_count": len(facility_ids), "anomaly_count": anomaly_count},
    )
    return {"facility_count": len(facility_ids), "anomaly_count": anomaly_count}


@shared_task(queue="training")
def check_all_drift() -> dict[str, Any]:
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


@shared_task(queue="training")
def retrain_model(facility_id: int | None, model_type: str) -> dict[str, Any]:
    if os.environ.get("FUELSENSE_ENABLE_TRAINING_TASKS", "0") != "1":
        return {"facility_id": facility_id, "model_type": model_type, "status": "disabled"}

    if model_type != "DEMAND_FORECAST":
        logger.info("retrain_model skipped for unsupported model type", extra={"model_type": model_type})
        return {"facility_id": facility_id, "model_type": model_type, "status": "skipped"}

    from ml_pipeline.training import ForecastTrainer

    dataset = extract_training_data(facility_id)
    trainer = ForecastTrainer()
    try:
        result = trainer.train_and_register(
            facility_id=facility_id,
            train_data=dataset["train_data"],
            train_targets=dataset["train_targets"],
            val_data=dataset["val_data"],
            val_targets=dataset["val_targets"],
            test_data=dataset["test_data"],
            test_targets=dataset["test_targets"],
        )
    except Exception as exc:
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
            is_active=False
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


@shared_task(queue="planning")
def run_planning_cycle() -> dict[str, Any]:
    queued_facilities = list(
        Facility.objects.filter(is_active=True, current_inventory__lte=F("dynamic_reorder_point")).order_by("id")
    )
    if not queued_facilities:
        return {"queued": 0, "planning_cycles": 0, "deliveries_created": 0}

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

    planning_cycles = 0
    deliveries_created = 0
    for depot_id, facilities in facilities_by_depot.items():
        depot = depot_map[depot_id]
        payload, facility_index_map = build_optimizer_request(depot, facilities)
        result = _run_optimizer(payload)
        routes = [r for r in list(result.get("routes", [])) if isinstance(r, dict)]

        cycle = PlanningCycle.objects.create(
            trigger_type=PlanningCycle.TriggerType.SCHEDULED,
            facilities_in_queue=len(facilities),
            deliveries_created=len(routes),
            total_distance_km=float(result.get("total_distance_km", 0.0)),
            total_cost=float(result.get("total_cost", 0.0)),
            solver_time_ms=float(result.get("solver_time_ms", 0.0)),
            baseline_cost=float(result.get("baseline_cost", 0.0)),
            cost_reduction_pct=float(result.get("cost_reduction_pct", 0.0)),
        )
        planning_cycles += 1

        vehicles = list(depot.vehicles.filter(is_available=True).order_by("id"))
        for route in routes:
            vehicle_idx = int(route.get("vehicle_index", 0))
            if vehicle_idx < 0 or vehicle_idx >= len(vehicles):
                continue
            delivery = Delivery.objects.create(
                depot=depot,
                vehicle=vehicles[vehicle_idx],
                planned_date=timezone.localdate(),
                status=Delivery.Status.PLANNED,
                total_distance_km=float(route.get("distance_km", 0.0)),
                total_cost=float(route.get("cost", 0.0)),
                route_json=route,
                solver_time_ms=float(result.get("solver_time_ms", 0.0)),
                created_by_planning_cycle=cycle,
            )
            deliveries_created += 1

            for stop in list(route.get("stops", [])):
                if not isinstance(stop, dict):
                    continue
                facility_index = int(stop.get("facility_index", -1))
                facility_id = facility_index_map.get(facility_index)
                if facility_id is None:
                    continue
                DeliveryItem.objects.create(
                    delivery=delivery,
                    facility_id=facility_id,
                    quantity=float(stop.get("demand", 0.0)),
                    planned_arrival=_planned_arrival_for_minutes(int(stop.get("arrival_min", 0))),
                    sequence=int(stop.get("sequence", 1)),
                )

    logger.info(
        "run_planning_cycle completed",
        extra={
            "queued": len(queued_facilities),
            "planning_cycles": planning_cycles,
            "deliveries_created": deliveries_created,
        },
    )
    return {
        "queued": len(queued_facilities),
        "planning_cycles": planning_cycles,
        "deliveries_created": deliveries_created,
    }


@shared_task(queue="planning")
def trigger_emergency_delivery(facility_id: int) -> dict[str, Any]:
    assignment = (
        DepotFacilityAssignment.objects.filter(facility_id=facility_id).select_related("depot", "facility").first()
    )
    if assignment is None:
        return {"facility_id": facility_id, "exists": False}

    payload, facility_index_map = build_optimizer_request(assignment.depot, [assignment.facility])
    result = _run_optimizer(payload)
    routes = [r for r in list(result.get("routes", [])) if isinstance(r, dict)]

    cycle = PlanningCycle.objects.create(
        trigger_type=PlanningCycle.TriggerType.EMERGENCY,
        facilities_in_queue=1,
        deliveries_created=len(routes),
        total_distance_km=float(result.get("total_distance_km", 0.0)),
        total_cost=float(result.get("total_cost", 0.0)),
        solver_time_ms=float(result.get("solver_time_ms", 0.0)),
        baseline_cost=float(result.get("baseline_cost", 0.0)),
        cost_reduction_pct=float(result.get("cost_reduction_pct", 0.0)),
    )

    vehicles = list(assignment.depot.vehicles.filter(is_available=True).order_by("id"))
    delivery_count = 0
    for route in routes:
        vehicle_idx = int(route.get("vehicle_index", 0))
        if vehicle_idx < 0 or vehicle_idx >= len(vehicles):
            continue
        delivery = Delivery.objects.create(
            depot=assignment.depot,
            vehicle=vehicles[vehicle_idx],
            planned_date=timezone.localdate(),
            status=Delivery.Status.PLANNED,
            total_distance_km=float(route.get("distance_km", 0.0)),
            total_cost=float(route.get("cost", 0.0)),
            route_json=route,
            solver_time_ms=float(result.get("solver_time_ms", 0.0)),
            created_by_planning_cycle=cycle,
        )
        delivery_count += 1
        for stop in list(route.get("stops", [])):
            if not isinstance(stop, dict):
                continue
            facility_index = int(stop.get("facility_index", -1))
            mapped_facility_id = facility_index_map.get(facility_index)
            if mapped_facility_id is None:
                continue
            DeliveryItem.objects.create(
                delivery=delivery,
                facility_id=mapped_facility_id,
                quantity=float(stop.get("demand", 0.0)),
                planned_arrival=_planned_arrival_for_minutes(int(stop.get("arrival_min", 0))),
                sequence=int(stop.get("sequence", 1)),
            )

    logger.info(
        "trigger_emergency_delivery completed",
        extra={"facility_id": facility_id, "deliveries_created": delivery_count},
    )
    return {"facility_id": facility_id, "exists": True, "deliveries_created": delivery_count}
