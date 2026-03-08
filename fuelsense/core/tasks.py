"""Celery task definitions for FuelSense core app."""

from __future__ import annotations

import logging
from datetime import timedelta

from celery import shared_task
from django.db.models import F
from django.utils import timezone

from fuelsense.core.models import Facility, InventoryLog, ModelRegistry

logger = logging.getLogger(__name__)


@shared_task(queue="default")
def daily_tick() -> dict:
	facility_ids = list(Facility.objects.filter(is_active=True).values_list("id", flat=True))
	logger.info("daily_tick invoked", extra={"facility_count": len(facility_ids)})
	return {"facility_count": len(facility_ids)}


@shared_task(queue="default")
def ingest_hourly() -> dict:
	count = Facility.objects.filter(is_active=True).count()
	logger.info("ingest_hourly invoked", extra={"active_facilities": count})
	return {"active_facilities": count}


@shared_task(queue="default")
def ingest_facility_data(facility_id: int) -> dict:
	facility = Facility.objects.get(id=facility_id)
	now = timezone.now()
	last_log = facility.inventory_logs.order_by("-timestamp").first()
	baseline_inventory = last_log.inventory_level if last_log else facility.current_inventory
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
def ingestion_complete() -> dict:
	recent_logs = InventoryLog.objects.filter(timestamp__gte=timezone.now() - timedelta(hours=1)).count()
	logger.info("ingestion_complete invoked", extra={"recent_logs": recent_logs})
	return {"recent_logs": recent_logs}


@shared_task(queue="default")
def run_batch_forecasts(facility_ids: list[int] | None = None) -> dict:
	if facility_ids is None:
		facility_ids = list(Facility.objects.filter(is_active=True).values_list("id", flat=True))
	logger.info("run_batch_forecasts invoked", extra={"facility_count": len(facility_ids)})
	return {"facility_count": len(facility_ids)}


@shared_task(queue="default")
def run_batch_anomaly_detection(facility_ids: list[int] | None = None) -> dict:
	if facility_ids is None:
		facility_ids = list(Facility.objects.filter(is_active=True).values_list("id", flat=True))
	logger.info("run_batch_anomaly_detection invoked", extra={"facility_count": len(facility_ids)})
	return {"facility_count": len(facility_ids)}


@shared_task(queue="training")
def check_all_drift() -> dict:
	models_count = ModelRegistry.objects.count()
	logger.info("check_all_drift invoked", extra={"models": models_count})
	return {"models": models_count}


@shared_task(queue="training")
def retrain_model(facility_id: int | None, model_type: str) -> dict:
	logger.info("retrain_model invoked", extra={"facility_id": facility_id, "model_type": model_type})
	return {"facility_id": facility_id, "model_type": model_type}


@shared_task(queue="planning")
def run_planning_cycle() -> dict:
	queued = Facility.objects.filter(current_inventory__lte=F("dynamic_reorder_point"))
	count = queued.count()
	logger.info("run_planning_cycle invoked", extra={"queued": count})
	return {"queued": count}


@shared_task(queue="planning")
def trigger_emergency_delivery(facility_id: int) -> dict:
	exists = Facility.objects.filter(id=facility_id).exists()
	logger.info("trigger_emergency_delivery invoked", extra={"facility_id": facility_id, "exists": exists})
	return {"facility_id": facility_id, "exists": exists}
