"""Redis cache helpers for frequently accessed operational data."""

from __future__ import annotations

from decimal import Decimal
from typing import TypedDict, cast

from django.core.cache import cache
from django.db.models import Avg
from django.db.models import F

from fuelsense.core.models import Facility


class DashboardKpis(TypedDict):
    avg_delivery_cost_last_30d: float
    forecast_accuracy_mape: float
    anomaly_detection_rate: float
    unacknowledged_anomalies_count: int
    facilities_below_reorder: int
    deliveries_in_transit: int


def _to_float(value: float | int | Decimal | None) -> float:
    if value is None:
        return 0.0
    return float(value)


def _is_dashboard_kpis(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    typed_payload = cast(dict[str, object], payload)
    expected_keys = {
        "avg_delivery_cost_last_30d",
        "forecast_accuracy_mape",
        "anomaly_detection_rate",
        "unacknowledged_anomalies_count",
        "facilities_below_reorder",
        "deliveries_in_transit",
    }
    payload_keys = set(typed_payload.keys())
    return expected_keys.issubset(payload_keys)


def get_or_set_facility_inventory(facility_id: int) -> float:
    key = f"cache:facility:{facility_id}:latest_inventory"
    value = cache.get(key)
    if value is not None:
        return float(value)

    facility = Facility.objects.only("current_inventory").get(id=facility_id)
    raw_inventory = cast(float | int | Decimal | None, getattr(facility, "current_inventory", None))
    value = _to_float(raw_inventory)
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


def get_or_set_dashboard_kpis() -> DashboardKpis:
    key = "cache:dashboard:kpis"
    cached = cache.get(key)
    if _is_dashboard_kpis(cached):
        return DashboardKpis(
            avg_delivery_cost_last_30d=_to_float(cached.get("avg_delivery_cost_last_30d")),
            forecast_accuracy_mape=_to_float(cached.get("forecast_accuracy_mape")),
            anomaly_detection_rate=_to_float(cached.get("anomaly_detection_rate")),
            unacknowledged_anomalies_count=int(cached.get("unacknowledged_anomalies_count", 0)),
            facilities_below_reorder=int(cached.get("facilities_below_reorder", 0)),
            deliveries_in_transit=int(cached.get("deliveries_in_transit", 0)),
        )

    from fuelsense.core.models import AnomalyAlert, Delivery, Forecast, InventoryLog

    payload: DashboardKpis = {
        "avg_delivery_cost_last_30d": _to_float(
            Delivery.objects.exclude(total_cost__isnull=True).aggregate(v=Avg("total_cost")).get("v")
        ),
        "forecast_accuracy_mape": _to_float(
            Forecast.objects.exclude(rmse__isnull=True).aggregate(v=Avg("rmse")).get("v")
        ),
        "anomaly_detection_rate": float(AnomalyAlert.objects.count() / max(InventoryLog.objects.count(), 1)),
        "unacknowledged_anomalies_count": int(AnomalyAlert.objects.filter(is_acknowledged=False).count()),
        "facilities_below_reorder": int(
            Facility.objects.filter(current_inventory__lte=F("dynamic_reorder_point")).count()
        ),
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
