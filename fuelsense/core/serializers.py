"""DRF serializers for FuelSense core app."""

from __future__ import annotations

from django.db.models import QuerySet
from rest_framework import serializers

from fuelsense.core.models import (
    AnomalyAlert,
    Delivery,
    DeliveryItem,
    Facility,
    Forecast,
    FuelType,
    InventoryLog,
    ModelRegistry,
    PlanningCycle,
    Vehicle,
)


class FuelTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = FuelType
        fields = "__all__"


class VehicleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Vehicle
        fields = ["id", "registration", "capacity", "cost_per_km", "is_available"]


class InventoryLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = InventoryLog
        fields = [
            "id",
            "timestamp",
            "inventory_level",
            "consumption",
            "temperature",
            "wind_speed",
            "solar_irradiance",
        ]


class ForecastSerializer(serializers.ModelSerializer):
    class Meta:
        model = Forecast
        fields = ["id", "model_version", "created_at", "horizon_days", "predictions_json", "rmse"]


class AnomalyAlertSerializer(serializers.ModelSerializer):
    acknowledged_by = serializers.SerializerMethodField()

    class Meta:
        model = AnomalyAlert
        fields = [
            "id",
            "facility",
            "timestamp",
            "anomaly_type",
            "score",
            "actual_consumption",
            "predicted_consumption",
            "is_acknowledged",
            "acknowledged_by",
            "notes",
        ]

    def get_acknowledged_by(self, obj: AnomalyAlert) -> str | None:
        return obj.acknowledged_by.username if obj.acknowledged_by else None


class AnomalyAlertAcknowledgeSerializer(serializers.Serializer):
    alert_id = serializers.IntegerField()
    notes = serializers.CharField(required=False, allow_blank=True)


class FacilityListSerializer(serializers.ModelSerializer):
    fuel_type = FuelTypeSerializer(read_only=True)
    reorder_status = serializers.CharField(read_only=True)

    class Meta:
        model = Facility
        fields = [
            "id",
            "name",
            "facility_type",
            "fuel_type",
            "latitude",
            "longitude",
            "current_inventory",
            "dynamic_reorder_point",
            "reorder_status",
            "is_active",
        ]


class FacilityDetailSerializer(FacilityListSerializer):
    latest_forecast = serializers.SerializerMethodField()
    latest_inventory_log = serializers.SerializerMethodField()

    class Meta(FacilityListSerializer.Meta):
        fields = FacilityListSerializer.Meta.fields + [
            "storage_capacity",
            "min_safe_inventory",
            "delivery_window_start",
            "delivery_window_end",
            "delivery_days",
            "latest_forecast",
            "latest_inventory_log",
        ]

    def get_latest_forecast(self, obj: Facility) -> dict | None:
        latest = obj.forecasts.order_by("-created_at").first()
        return ForecastSerializer(latest).data if latest else None

    def get_latest_inventory_log(self, obj: Facility) -> dict | None:
        latest = obj.inventory_logs.order_by("-timestamp").first()
        return InventoryLogSerializer(latest).data if latest else None


class DeliveryItemSerializer(serializers.ModelSerializer):
    facility = FacilityListSerializer(read_only=True)

    class Meta:
        model = DeliveryItem
        fields = ["id", "facility", "quantity", "planned_arrival", "actual_arrival", "sequence"]


class DeliveryListSerializer(serializers.ModelSerializer):
    vehicle = VehicleSerializer(read_only=True)

    class Meta:
        model = Delivery
        fields = ["id", "depot", "vehicle", "planned_date", "status", "total_distance_km", "total_cost"]


class DeliveryDetailSerializer(DeliveryListSerializer):
    items = DeliveryItemSerializer(many=True, read_only=True)

    class Meta(DeliveryListSerializer.Meta):
        fields = DeliveryListSerializer.Meta.fields + ["route_json", "solver_time_ms", "items"]


class DeliveryStatusUpdateSerializer(serializers.Serializer):
    status = serializers.ChoiceField(choices=Delivery.Status.choices)
    notes = serializers.CharField(required=False, allow_blank=True)


class PlanningCycleSerializer(serializers.ModelSerializer):
    class Meta:
        model = PlanningCycle
        fields = "__all__"


class PlanningTriggerSerializer(serializers.Serializer):
    trigger_type = serializers.ChoiceField(choices=["MANUAL", "EMERGENCY"])
    facility_ids = serializers.ListField(child=serializers.IntegerField(), required=False)


class ModelRegistrySerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelRegistry
        fields = "__all__"


class DashboardKPISerializer(serializers.Serializer):
    avg_delivery_cost_last_30d = serializers.FloatField()
    forecast_accuracy_mape = serializers.FloatField()
    anomaly_detection_rate = serializers.FloatField()
    unacknowledged_anomalies_count = serializers.IntegerField()
    facilities_below_reorder = serializers.IntegerField()
    deliveries_in_transit = serializers.IntegerField()


class DriftHeatmapSerializer(serializers.Serializer):
    facility_id = serializers.IntegerField(allow_null=True)
    facility_name = serializers.CharField(allow_null=True)
    drift_ratio = serializers.FloatField()
    model_version = serializers.IntegerField()


def latest_forecasts(qs: QuerySet[Forecast]) -> QuerySet[Forecast]:
    return qs.order_by("-created_at")
