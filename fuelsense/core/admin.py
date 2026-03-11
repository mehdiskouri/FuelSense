"""Django admin registrations for FuelSense core models."""

from __future__ import annotations

import base64
from io import BytesIO

import matplotlib
from django.contrib import admin, messages
from django.db.models import Avg, Count, F
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _figure_to_base64() -> str:
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class InventoryLogInline(admin.TabularInline):
    model = InventoryLog
    extra = 0
    max_num = 10
    ordering = ("-timestamp",)


class DeliveryItemInline(admin.TabularInline):
    model = DeliveryItem
    extra = 0


@admin.register(Facility)
class FacilityAdmin(admin.ModelAdmin):
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
    inlines = [InventoryLogInline]
    actions = ["trigger_emergency_delivery"]
    change_form_template = "admin/core/facility_change_form.html"

    @admin.action(description="Trigger Emergency Delivery")
    def trigger_emergency_delivery(self, request, queryset):
        from celery import current_app

        for facility in queryset:
            current_app.send_task("fuelsense.core.tasks.trigger_emergency_delivery", args=[facility.id])
        self.message_user(request, f"Queued emergency delivery for {queryset.count()} facilities.", messages.SUCCESS)

    def change_view(self, request, object_id, form_url="", extra_context=None):
        extra_context = extra_context or {}
        facility = Facility.objects.filter(pk=object_id).first()
        if facility:
            logs = facility.inventory_logs.order_by("-timestamp")[:30]
            logs = list(reversed(logs))
            if logs:
                plt.figure(figsize=(8, 3))
                x = [log.timestamp for log in logs]
                plt.plot(x, [log.consumption for log in logs], label="Consumption")
                plt.plot(x, [log.inventory_level for log in logs], label="Inventory")
                latest_forecast = facility.forecasts.order_by("-created_at").first()
                if latest_forecast and isinstance(latest_forecast.predictions_json, list):
                    forecast_values = [
                        row.get("p50") for row in latest_forecast.predictions_json if isinstance(row, dict)
                    ]
                    if forecast_values:
                        plt.plot(
                            [x[-1] for _ in forecast_values],
                            forecast_values,
                            "o",
                            label="Forecast p50",
                        )
                plt.legend()
                extra_context["facility_chart_b64"] = _figure_to_base64()
            extra_context["recent_alerts"] = facility.alerts.order_by("-timestamp")[:10]
        return super().change_view(request, object_id, form_url, extra_context)


@admin.register(Delivery)
class DeliveryAdmin(admin.ModelAdmin):
    list_display = ("id", "depot", "vehicle", "planned_date", "status", "total_cost", "total_distance_km")
    list_filter = ("status", "planned_date")
    inlines = [DeliveryItemInline]
    change_form_template = "admin/core/delivery_change_form.html"

    def change_view(self, request, object_id, form_url="", extra_context=None):
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
class PlanningCycleAdmin(admin.ModelAdmin):
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
    readonly_fields = [field.name for field in PlanningCycle._meta.fields]


@admin.register(ModelRegistry)
class ModelRegistryAdmin(admin.ModelAdmin):
    list_display = ("model_type", "facility", "version", "is_active", "trained_at", "drift_ratio")
    list_filter = ("model_type", "is_active")
    actions = ["promote_to_active", "rollback_to_previous"]

    @admin.action(description="Promote to Active")
    def promote_to_active(self, request, queryset):
        for model in queryset:
            ModelRegistry.objects.filter(model_type=model.model_type, facility=model.facility).exclude(
                id=model.id
            ).update(is_active=False)
            model.is_active = True
            model.save(update_fields=["is_active"])
        self.message_user(request, f"Promoted {queryset.count()} model(s).", messages.SUCCESS)

    @admin.action(description="Rollback to Previous")
    def rollback_to_previous(self, request, queryset):
        rolled = 0
        for model in queryset:
            prev = (
                ModelRegistry.objects.filter(
                    model_type=model.model_type, facility=model.facility, version__lt=model.version
                )
                .order_by("-version")
                .first()
            )
            if prev:
                ModelRegistry.objects.filter(model_type=model.model_type, facility=model.facility).update(
                    is_active=False
                )
                prev.is_active = True
                prev.save(update_fields=["is_active"])
                rolled += 1
        self.message_user(request, f"Rolled back {rolled} model(s).", messages.SUCCESS)


@admin.register(FuelType)
class FuelTypeAdmin(admin.ModelAdmin):
    pass


@admin.register(Depot)
class DepotAdmin(admin.ModelAdmin):
    pass


@admin.register(DepotFacilityAssignment)
class DepotFacilityAssignmentAdmin(admin.ModelAdmin):
    pass


@admin.register(Vehicle)
class VehicleAdmin(admin.ModelAdmin):
    pass


@admin.register(InventoryLog)
class InventoryLogAdmin(admin.ModelAdmin):
    pass


@admin.register(DeliveryItem)
class DeliveryItemAdmin(admin.ModelAdmin):
    pass


@admin.register(Forecast)
class ForecastAdmin(admin.ModelAdmin):
    pass


@admin.register(AnomalyAlert)
class AnomalyAlertAdmin(admin.ModelAdmin):
    pass


def admin_dashboard(request):
    below_reorder = Facility.objects.filter(current_inventory__lte=F("dynamic_reorder_point"))
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


def get_admin_urls(urls):
    def get_urls():
        custom_urls = [
            path("fuelsense/dashboard/", admin.site.admin_view(admin_dashboard), name="fuelsense-dashboard"),
        ]
        return custom_urls + urls()

    return get_urls


admin.site.get_urls = get_admin_urls(admin.site.get_urls)
