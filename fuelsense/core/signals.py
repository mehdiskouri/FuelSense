"""Signal handlers for cache invalidation and metric refreshes."""

from __future__ import annotations

from django.core.cache import cache
from django.db.models.signals import post_save
from django.dispatch import receiver

from fuelsense.core.cache import invalidate_facility_cache
from fuelsense.core.models import AnomalyAlert, Forecast, InventoryLog


@receiver(post_save, sender=InventoryLog)
def on_inventory_log_saved(sender, instance: InventoryLog, **kwargs) -> None:
    invalidate_facility_cache(instance.facility_id)


@receiver(post_save, sender=Forecast)
def on_forecast_saved(sender, instance: Forecast, **kwargs) -> None:
    cache.delete(f"cache:facility:{instance.facility_id}:latest_forecast")
    cache.delete("cache:dashboard:kpis")


@receiver(post_save, sender=AnomalyAlert)
def on_anomaly_alert_saved(sender, instance: AnomalyAlert, **kwargs) -> None:
    cache.delete("cache:anomaly_alerts:today")
    cache.delete("cache:dashboard:kpis")
