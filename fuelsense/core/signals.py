"""Signal handlers for cache invalidation and metric refreshes."""

from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false
from django.core.cache import cache
from django.db.models.signals import post_save
from django.dispatch import receiver

from fuelsense.core.cache import invalidate_facility_cache
from fuelsense.core.models import AnomalyAlert, Facility, Forecast, InventoryLog


@receiver(post_save, sender=InventoryLog)
def on_inventory_log_saved(_sender: object, instance: InventoryLog, **_kwargs: object) -> None:
    """Invalidate per-facility cache entries when inventory logs are saved."""
    invalidate_facility_cache(instance.facility_id)


@receiver(post_save, sender=Forecast)
def on_forecast_saved(_sender: object, instance: Forecast, **_kwargs: object) -> None:
    """Invalidate forecast and dashboard caches when new forecasts are persisted."""
    cache.delete(f"cache:facility:{instance.facility_id}:latest_forecast")
    cache.delete("cache:dashboard:kpis")


@receiver(post_save, sender=AnomalyAlert)
def on_anomaly_alert_saved(_sender: object, _instance: AnomalyAlert, **_kwargs: object) -> None:
    """Invalidate anomaly and dashboard caches on alert writes."""
    cache.delete("cache:anomaly_alerts:today")
    cache.delete("cache:dashboard:kpis")


@receiver(post_save, sender=Facility)
def on_facility_saved(_sender: object, instance: Facility, **_kwargs: object) -> None:
    """Invalidate facility/dashboard caches when facility attributes change."""
    invalidate_facility_cache(instance.id)
    cache.delete("cache:dashboard:kpis")
