"""Celery task definitions for FuelSense core app."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Any

import httpx
from celery import shared_task
from django.db.models import F, Max
from django.utils import timezone

from fuelsense.core.features import build_anomaly_features, build_drift_data, build_lookback_matrix, extract_training_data
from fuelsense.core.models import AnomalyAlert, Facility, Forecast, InventoryLog, ModelRegistry
from ml_pipeline.drift import DriftMonitor
from ml_pipeline.training import ForecastTrainer

logger = logging.getLogger(__name__)


@shared_task(queue="default")
def daily_tick() -> dict[str, Any]:
    facility_ids = list(Facility.objects.filter(is_active=True).values_list("id", flat=True))
    logger.info("daily_tick invoked", extra={"facility_count": len(facility_ids)})
    return {"facility_count": len(facility_ids)}


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

    anomaly_count = 0
    for facility_id in facility_ids:
        features = build_anomaly_features(int(facility_id))
        if features is None:
            continue

        latest_log = (
            InventoryLog.objects.filter(facility_id=facility_id)
            .order_by("-timestamp")
            .values("consumption")
            .first()
        )
        recent_logs = list(
            InventoryLog.objects.filter(facility_id=facility_id)
            .order_by("-timestamp")
            .values_list("consumption", flat=True)[:7]
        )
        latest_forecast = Forecast.objects.filter(facility_id=facility_id).order_by("-created_at").only("predictions_json").first()
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
    previous_rmse = float(current_active.validation_rmse) if current_active and current_active.validation_rmse else float("inf")
    improved = float(result["test_rmse"]) < previous_rmse

    if improved:
        ModelRegistry.objects.filter(model_type=model_type, facility_id=facility_id, is_active=True).update(is_active=False)
        max_version = (
            ModelRegistry.objects.filter(model_type=model_type, facility_id=facility_id).aggregate(v=Max("version")).get("v")
            or 0
        )
        ModelRegistry.objects.create(
            model_type=model_type,
            facility_id=facility_id,
            mlflow_run_id=str(result["run_id"]),
            version=int(max_version) + 1,
            is_active=True,
            trained_at=timezone.now(),
            training_rmse=float(result["test_rmse"]),
            validation_rmse=float(result["test_rmse"]),
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
    queued = Facility.objects.filter(current_inventory__lte=F("dynamic_reorder_point"))
    count = queued.count()
    logger.info("run_planning_cycle invoked", extra={"queued": count})
    return {"queued": count}


@shared_task(queue="planning")
def trigger_emergency_delivery(facility_id: int) -> dict[str, Any]:
    exists = Facility.objects.filter(id=facility_id).exists()
    logger.info("trigger_emergency_delivery invoked", extra={"facility_id": facility_id, "exists": exists})
    return {"facility_id": facility_id, "exists": exists}
