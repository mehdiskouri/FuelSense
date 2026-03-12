"""Prometheus business metrics for FuelSense."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

active_anomaly_alerts = Gauge("active_anomaly_alerts", "Unacknowledged anomaly alerts")
facilities_below_reorder = Gauge("facilities_below_reorder", "Facilities below effective reorder point")
forecast_rmse = Gauge("forecast_rmse", "Forecast RMSE")
# Backward-compatible legacy metric name. It carries RMSE values, not MAPE.
forecast_mape_pct = Gauge("forecast_mape_pct", "Deprecated: Forecast RMSE")

celery_task_duration_seconds = Histogram(
    "celery_task_duration_seconds",
    "Celery task duration",
    labelnames=("task_name", "queue"),
)
celery_task_failures_total = Counter(
    "celery_task_failures_total",
    "Celery task failures",
    labelnames=("task_name", "queue"),
)
celery_task_success_total = Counter(
    "celery_task_success_total",
    "Celery task successes",
    labelnames=("task_name", "queue"),
)
retraining_triggered_total = Counter(
    "retraining_triggered_total",
    "Model retraining triggers",
    labelnames=("facility_id", "model_type"),
)


def refresh_business_gauges() -> None:
    from django.db.models import Avg

    from fuelsense.core.models import AnomalyAlert, Facility, Forecast
    from fuelsense.core.reorder import filter_below_reorder

    active_anomaly_alerts.set(AnomalyAlert.objects.filter(is_acknowledged=False).count())
    facilities_below_reorder.set(filter_below_reorder(Facility.objects.all()).count())
    value = float(Forecast.objects.exclude(rmse__isnull=True).aggregate(v=Avg("rmse"))["v"] or 0.0)
    forecast_rmse.set(value)
    forecast_mape_pct.set(value)
