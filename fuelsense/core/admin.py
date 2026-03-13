"""Django admin registrations for FuelSense core models."""

from __future__ import annotations

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportMissingTypeArgument=false
import base64
import logging
import math
from datetime import datetime, timedelta
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl
from celery import current_app
from django.contrib import admin, messages
from django.db.models import Avg, Count
from django.shortcuts import render
from django.urls import path
from django.utils import timezone

from fuelsense.core.models import (
    AnomalyAlert,
    Delivery,
    DeliveryItem,
    Depot,
    DepotFacilityAssignment,
    Facility,
    Forecast,
    FuelType,
    InventoryLog,
    ModelRegistry,
    PlanningCycle,
    Vehicle,
)
from fuelsense.core.reorder import filter_below_reorder

if TYPE_CHECKING:
    from collections.abc import Callable

    from django.db.models import QuerySet
    from django.http import HttpRequest, HttpResponse
    from django.urls.resolvers import URLPattern, URLResolver

    InventoryLogInlineBase = admin.TabularInline[InventoryLog, Facility]
    DeliveryItemInlineBase = admin.TabularInline[DeliveryItem, Delivery]
    FacilityAdminBase = admin.ModelAdmin[Facility]
    DeliveryAdminBase = admin.ModelAdmin[Delivery]
    PlanningCycleAdminBase = admin.ModelAdmin[PlanningCycle]
    ModelRegistryAdminBase = admin.ModelAdmin[ModelRegistry]
    FuelTypeAdminBase = admin.ModelAdmin[FuelType]
    DepotAdminBase = admin.ModelAdmin[Depot]
    DepotFacilityAssignmentAdminBase = admin.ModelAdmin[DepotFacilityAssignment]
    VehicleAdminBase = admin.ModelAdmin[Vehicle]
    InventoryLogAdminBase = admin.ModelAdmin[InventoryLog]
    DeliveryItemAdminBase = admin.ModelAdmin[DeliveryItem]
    ForecastAdminBase = admin.ModelAdmin[Forecast]
    AnomalyAlertAdminBase = admin.ModelAdmin[AnomalyAlert]
else:
    InventoryLogInlineBase = admin.TabularInline
    DeliveryItemInlineBase = admin.TabularInline
    FacilityAdminBase = admin.ModelAdmin
    DeliveryAdminBase = admin.ModelAdmin
    PlanningCycleAdminBase = admin.ModelAdmin
    ModelRegistryAdminBase = admin.ModelAdmin
    FuelTypeAdminBase = admin.ModelAdmin
    DepotAdminBase = admin.ModelAdmin
    DepotFacilityAssignmentAdminBase = admin.ModelAdmin
    VehicleAdminBase = admin.ModelAdmin
    InventoryLogAdminBase = admin.ModelAdmin
    DeliveryItemAdminBase = admin.ModelAdmin
    ForecastAdminBase = admin.ModelAdmin
    AnomalyAlertAdminBase = admin.ModelAdmin

mpl.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
FORECAST_STALE_DAYS_THRESHOLD = 7


def _figure_to_base64() -> str:
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_forecast_p50_points(forecast: Forecast) -> tuple[list[datetime], list[float], int]:
    if not isinstance(forecast.predictions_json, list):
        return [], [], 0

    timestamps: list[datetime] = []
    values: list[float] = []
    skipped_points = 0

    for idx, raw in enumerate(forecast.predictions_json, start=1):
        if not isinstance(raw, dict):
            skipped_points += 1
            continue

        raw_day = raw.get("day", idx)
        try:
            day_offset = int(raw_day)
        except (TypeError, ValueError):
            day_offset = idx
        if day_offset < 1:
            day_offset = idx

        raw_p50 = raw.get("p50")
        if isinstance(raw_p50, (int, float, str)):
            try:
                p50_value = float(raw_p50)
            except ValueError:
                skipped_points += 1
                continue
        else:
            skipped_points += 1
            continue
        if not math.isfinite(p50_value):
            skipped_points += 1
            continue

        timestamps.append(forecast.created_at + timedelta(days=day_offset))
        values.append(p50_value)

    expected_horizon = int(getattr(forecast, "horizon_days", 0) or 0)
    if expected_horizon > 0 and len(values) != expected_horizon:
        logger.warning(
            "forecast horizon mismatch",
            extra={
                "forecast_id": forecast.id,
                "expected_horizon": expected_horizon,
                "valid_points": len(values),
                "skipped_points": skipped_points,
            },
        )

    return timestamps, values, skipped_points


class InventoryLogInline(InventoryLogInlineBase):
    """Inline inventory log preview for facility admin pages."""

    model = InventoryLog
    extra = 0
    max_num = 10
    ordering = ("-timestamp",)


class DeliveryItemInline(DeliveryItemInlineBase):
    """Inline delivery items editor for delivery admin pages."""

    model = DeliveryItem
    extra = 0


@admin.register(Facility)
class FacilityAdmin(FacilityAdminBase):
    """Admin configuration for facilities with forecast and inventory charts."""

    list_display = (
        "name",
        "facility_type",
        "fuel_type",
        "current_inventory",
        "dynamic_reorder_point",
        "reorder_status",
        "is_active",
    )
    list_filter = ("fuel_type", "facility_type", "is_active")
    inlines = (InventoryLogInline,)
    actions = ("trigger_emergency_delivery",)
    change_form_template = "admin/core/facility_change_form.html"

    @admin.action(description="Trigger Emergency Delivery")
    def trigger_emergency_delivery(self, request: HttpRequest, queryset: QuerySet[Facility]) -> None:
        """Queue emergency delivery tasks for selected facilities."""
        for facility in queryset:
            current_app.send_task("fuelsense.core.tasks.trigger_emergency_delivery", args=[facility.id])
        self.message_user(request, f"Queued emergency delivery for {queryset.count()} facilities.", messages.SUCCESS)

    def change_view(
        self,
        request: HttpRequest,
        object_id: str,
        form_url: str = "",
        extra_context: dict[str, object] | None = None,
    ) -> HttpResponse:
        """Render facility admin page with inventory/forecast overlays and alert context."""
        extra_context = extra_context or {}
        facility = Facility.objects.filter(pk=object_id).first()
        if facility:
            latest_forecast = facility.forecasts.order_by("-created_at").first()
            extra_context["forecast_model_version"] = None
            extra_context["forecast_is_fallback"] = False
            extra_context["forecast_is_stale"] = False
            extra_context["forecast_stale_days"] = 0
            extra_context["forecast_skipped_points"] = 0

            forecast_timestamps: list[datetime] = []
            forecast_values: list[float] = []
            if latest_forecast:
                forecast_timestamps, forecast_values, skipped_points = _extract_forecast_p50_points(latest_forecast)
                forecast_age_days = max((timezone.now() - latest_forecast.created_at).days, 0)
                extra_context["forecast_model_version"] = latest_forecast.model_version
                extra_context["forecast_is_fallback"] = latest_forecast.model_version == "fallback-local"
                extra_context["forecast_is_stale"] = forecast_age_days > FORECAST_STALE_DAYS_THRESHOLD
                extra_context["forecast_stale_days"] = forecast_age_days
                extra_context["forecast_skipped_points"] = skipped_points

            logs_qs = facility.inventory_logs.order_by("-timestamp")[:30]
            logs = list(reversed(logs_qs))
            if logs:
                plt.figure(figsize=(8, 3))
                x = [log.timestamp for log in logs]
                plt.plot(cast("Any", x), [log.consumption for log in logs], label="Consumption")
                plt.plot(cast("Any", x), [log.inventory_level for log in logs], label="Inventory")
                if forecast_timestamps and forecast_values:
                    plt.plot(cast("Any", forecast_timestamps), forecast_values, "o-", label="Forecast p50")
                plt.legend()
                extra_context["facility_chart_b64"] = _figure_to_base64()
            extra_context["recent_alerts"] = facility.alerts.order_by("-timestamp")[:10]
        return super().change_view(request, object_id, form_url, extra_context)


@admin.register(Delivery)
class DeliveryAdmin(DeliveryAdminBase):
    """Admin configuration for deliveries with route map rendering."""

    list_display = ("id", "depot", "vehicle", "planned_date", "status", "total_cost", "total_distance_km")
    list_filter = ("status", "planned_date")
    inlines = (DeliveryItemInline,)
    change_form_template = "admin/core/delivery_change_form.html"

    def change_view(
        self,
        request: HttpRequest,
        object_id: str,
        form_url: str = "",
        extra_context: dict[str, object] | None = None,
    ) -> HttpResponse:
        """Render delivery admin page with route map and cost breakdown."""
        extra_context = extra_context or {}
        delivery = (
            Delivery.objects.select_related("depot").prefetch_related("items__facility").filter(pk=object_id).first()
        )
        if delivery and delivery.items.exists():
            points_lat = [delivery.depot.latitude]
            points_lng = [delivery.depot.longitude]
            for item in delivery.items.order_by("sequence"):
                points_lat.append(item.facility.latitude)
                points_lng.append(item.facility.longitude)
            points_lat.append(delivery.depot.latitude)
            points_lng.append(delivery.depot.longitude)

            plt.figure(figsize=(6, 4))
            plt.plot(points_lng, points_lat, marker="o")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title(f"Delivery Route #{delivery.id}")
            extra_context["route_map_b64"] = _figure_to_base64()
            extra_context["cost_breakdown"] = {
                "total_cost": delivery.total_cost or 0.0,
                "solver_time_ms": delivery.solver_time_ms or 0.0,
            }
        return super().change_view(request, object_id, form_url, extra_context)


@admin.register(PlanningCycle)
class PlanningCycleAdmin(PlanningCycleAdminBase):
    """Read-only admin representation for planning cycle execution records."""

    list_display = (
        "triggered_at",
        "trigger_type",
        "status",
        "started_at",
        "completed_at",
        "facilities_in_queue",
        "deliveries_created",
        "total_cost",
        "baseline_cost",
        "cost_reduction_pct",
    )
    readonly_fields = tuple(field.name for field in PlanningCycle._meta.fields)  # noqa: SLF001


@admin.register(ModelRegistry)
class ModelRegistryAdmin(ModelRegistryAdminBase):
    """Admin configuration for model registry lifecycle operations."""

    list_display = ("model_type", "facility", "version", "is_active", "trained_at", "drift_ratio")
    list_filter = ("model_type", "is_active")
    actions = ("promote_to_active", "rollback_to_previous")

    @admin.action(description="Promote to Active")
    def promote_to_active(self, request: HttpRequest, queryset: QuerySet[ModelRegistry]) -> None:
        """Promote selected models as active and demote siblings for same scope."""
        for model in queryset:
            ModelRegistry.objects.filter(model_type=model.model_type, facility=model.facility).exclude(
                id=model.id,
            ).update(is_active=False)
            model.is_active = True
            model.save(update_fields=["is_active"])
        self.message_user(request, f"Promoted {queryset.count()} model(s).", messages.SUCCESS)

    @admin.action(description="Rollback to Previous")
    def rollback_to_previous(self, request: HttpRequest, queryset: QuerySet[ModelRegistry]) -> None:
        """Rollback selected models to the highest available previous version."""
        rolled = 0
        for model in queryset:
            prev = (
                ModelRegistry.objects.filter(
                    model_type=model.model_type,
                    facility=model.facility,
                    version__lt=model.version,
                )
                .order_by("-version")
                .first()
            )
            if prev:
                ModelRegistry.objects.filter(model_type=model.model_type, facility=model.facility).update(
                    is_active=False,
                )
                prev.is_active = True
                prev.save(update_fields=["is_active"])
                rolled += 1
        self.message_user(request, f"Rolled back {rolled} model(s).", messages.SUCCESS)


@admin.register(FuelType)
class FuelTypeAdmin(FuelTypeAdminBase):
    """Admin registration for fuel type records."""

    list_display = ()


@admin.register(Depot)
class DepotAdmin(DepotAdminBase):
    """Admin registration for depots."""

    list_display = ()


@admin.register(DepotFacilityAssignment)
class DepotFacilityAssignmentAdmin(DepotFacilityAssignmentAdminBase):
    """Admin registration for depot-facility assignments."""

    list_display = ()


@admin.register(Vehicle)
class VehicleAdmin(VehicleAdminBase):
    """Admin registration for vehicle records."""

    list_display = ()


@admin.register(InventoryLog)
class InventoryLogAdmin(InventoryLogAdminBase):
    """Admin registration for inventory logs."""

    list_display = ()


@admin.register(DeliveryItem)
class DeliveryItemAdmin(DeliveryItemAdminBase):
    """Admin registration for delivery items."""

    list_display = ()


@admin.register(Forecast)
class ForecastAdmin(ForecastAdminBase):
    """Admin registration for forecast records."""

    list_display = ()


@admin.register(AnomalyAlert)
class AnomalyAlertAdmin(AnomalyAlertAdminBase):
    """Admin registration for anomaly alerts."""

    list_display = ()


def admin_dashboard(request: HttpRequest) -> HttpResponse:
    """Render operational admin dashboard with KPI snapshots and activity lists."""
    below_reorder = filter_below_reorder(Facility.objects.all())
    in_transit = Delivery.objects.filter(status=Delivery.Status.IN_TRANSIT)
    today_alerts = AnomalyAlert.objects.filter(timestamp__date=timezone.now().date()).order_by("-timestamp")
    drift = ModelRegistry.objects.select_related("facility").order_by("-trained_at")[:50]
    kpis = {
        "avg_delivery_cost": Delivery.objects.aggregate(v=Avg("total_cost"))["v"] or 0.0,
        "forecast_accuracy": Forecast.objects.aggregate(v=Avg("rmse"))["v"] or 0.0,
        "anomaly_detection_rate": (AnomalyAlert.objects.count() / max(InventoryLog.objects.count(), 1)) * 100,
        "route_cost_reduction": PlanningCycle.objects.filter(status=PlanningCycle.ExecutionStatus.COMPLETED)
        .exclude(cost_reduction_pct__isnull=True)
        .aggregate(v=Avg("cost_reduction_pct"))["v"]
        or 0.0,
        "in_transit_count": in_transit.count(),
        "below_reorder_count": below_reorder.count(),
        "today_alert_count": today_alerts.count(),
        "active_model_count": ModelRegistry.objects.filter(is_active=True).aggregate(v=Count("id"))["v"] or 0,
    }
    return render(
        request,
        "admin/core/dashboard.html",
        {
            "below_reorder": below_reorder,
            "in_transit": in_transit,
            "today_alerts": today_alerts,
            "drift": drift,
            "kpis": kpis,
        },
    )


def get_admin_urls(urls: Callable[[], list[URLResolver | URLPattern]]) -> Callable[[], list[URLResolver | URLPattern]]:
    """Inject FuelSense dashboard route into Django admin URL set."""

    def get_urls() -> list[URLResolver | URLPattern]:
        custom_urls = [
            path("fuelsense/dashboard/", admin.site.admin_view(admin_dashboard), name="fuelsense-dashboard"),
        ]
        return custom_urls + urls()

    return get_urls


cast(Any, admin.site).get_urls = get_admin_urls(admin.site.get_urls)  # noqa: TC006
