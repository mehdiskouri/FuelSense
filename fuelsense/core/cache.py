"""Redis cache helpers for frequently accessed operational data."""

from __future__ import annotations

from django.core.cache import cache
from django.db.models import Avg
from django.db.models import F

from fuelsense.core.models import Facility


def get_or_set_facility_inventory(facility_id: int) -> float:
    key = f"cache:facility:{facility_id}:latest_inventory"
    value = cache.get(key)
    if value is not None:
        return float(value)

    facility = Facility.objects.only("current_inventory").get(id=facility_id)
    value = float(facility.current_inventory)
    cache.set(key, value, timeout=3600)
    return value


def get_or_set_reorder_status(facility_id: int) -> str:
    key = f"cache:facility:{facility_id}:reorder_status"
    value = cache.get(key)
    if value is not None:
        return str(value)

    facility = Facility.objects.get(id=facility_id)
    value = facility.reorder_status
    cache.set(key, value, timeout=900)
    return value


def get_or_set_dashboard_kpis() -> dict:
    key = "cache:dashboard:kpis"
    cached = cache.get(key)
    if cached is not None:
        return dict(cached)

    from fuelsense.core.models import AnomalyAlert, Delivery, Forecast, InventoryLog

    payload = {
        "avg_delivery_cost_last_30d": float(
            Delivery.objects.exclude(total_cost__isnull=True).aggregate(v=Avg("total_cost"))["v"] or 0.0
        ),
        "forecast_accuracy_mape": float(Forecast.objects.exclude(rmse__isnull=True).aggregate(v=Avg("rmse"))["v"] or 0.0),
        "anomaly_detection_rate": float(AnomalyAlert.objects.count() / max(InventoryLog.objects.count(), 1)),
        "unacknowledged_anomalies_count": int(AnomalyAlert.objects.filter(is_acknowledged=False).count()),
        "facilities_below_reorder": int(Facility.objects.filter(current_inventory__lte=F("dynamic_reorder_point")).count()),
        "deliveries_in_transit": int(Delivery.objects.filter(status=Delivery.Status.IN_TRANSIT).count()),
    }
    cache.set(key, payload, timeout=300)
    return payload


def invalidate_facility_cache(facility_id: int) -> None:
    cache.delete_many(
        [
            f"cache:facility:{facility_id}:latest_inventory",
            f"cache:facility:{facility_id}:reorder_status",
        ]
    )
