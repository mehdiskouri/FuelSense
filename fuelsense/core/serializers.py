"""DRF serializers for FuelSense core app."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from django.db.models import QuerySet


class FuelTypeSerializer(serializers.ModelSerializer):
    """Serializer for fuel type records."""

    class Meta:
        """DRF metadata for FuelType serialization."""

        model = FuelType
        fields = "__all__"


class VehicleSerializer(serializers.ModelSerializer):
    """Serializer for vehicle records."""

    class Meta:
        """DRF metadata for Vehicle serialization."""

        model = Vehicle
        fields = ("id", "registration", "capacity", "cost_per_km", "is_available")


class InventoryLogSerializer(serializers.ModelSerializer):
    """Serializer for inventory log snapshots."""

    class Meta:
        """DRF metadata for InventoryLog serialization."""

        model = InventoryLog
        fields = (
            "id",
            "timestamp",
            "inventory_level",
            "consumption",
            "temperature",
            "wind_speed",
            "solar_irradiance",
        )


class ForecastSerializer(serializers.ModelSerializer):
    """Serializer for persisted forecasts."""

    class Meta:
        """DRF metadata for Forecast serialization."""

        model = Forecast
        fields = ("id", "model_version", "created_at", "horizon_days", "predictions_json", "rmse")


class AnomalyAlertSerializer(serializers.ModelSerializer):
    """Serializer for anomaly alert records."""

    acknowledged_by = serializers.SerializerMethodField()

    class Meta:
        """DRF metadata for AnomalyAlert serialization."""

        model = AnomalyAlert
        fields = (
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
        )

    def get_acknowledged_by(self, obj: AnomalyAlert) -> str | None:
        """Return username for acknowledging user when present."""
        return obj.acknowledged_by.username if obj.acknowledged_by else None


class AnomalyAlertAcknowledgeSerializer(serializers.Serializer):
    """Payload serializer for alert acknowledgement requests."""

    alert_id = serializers.IntegerField()
    notes = serializers.CharField(required=False, allow_blank=True)


class FacilityListSerializer(serializers.ModelSerializer):
    """Serializer for facility list responses."""

    fuel_type = FuelTypeSerializer(read_only=True)
    reorder_status = serializers.CharField(read_only=True)

    class Meta:
        """DRF metadata for list-level Facility serialization."""

        model = Facility
        fields = (
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
        )


class FacilityDetailSerializer(FacilityListSerializer):
    """Serializer for detailed facility responses."""

    latest_forecast = serializers.SerializerMethodField()
    latest_inventory_log = serializers.SerializerMethodField()

    class Meta(FacilityListSerializer.Meta):
        """DRF metadata for detail-level Facility serialization."""

        fields = (
            *FacilityListSerializer.Meta.fields,
            *(
                "storage_capacity",
                "min_safe_inventory",
                "delivery_window_start",
                "delivery_window_end",
                "delivery_days",
                "latest_forecast",
                "latest_inventory_log",
            ),
        )

    def get_latest_forecast(self, obj: Facility) -> dict | None:
        """Return the latest forecast payload when available."""
        latest = obj.forecasts.order_by("-created_at").first()
        return ForecastSerializer(latest).data if latest else None

    def get_latest_inventory_log(self, obj: Facility) -> dict | None:
        """Return the latest inventory observation when available."""
        latest = obj.inventory_logs.order_by("-timestamp").first()
        return InventoryLogSerializer(latest).data if latest else None


class DeliveryItemSerializer(serializers.ModelSerializer):
    """Serializer for delivery stop items."""

    facility = FacilityListSerializer(read_only=True)

    class Meta:
        """DRF metadata for DeliveryItem serialization."""

        model = DeliveryItem
        fields = ("id", "facility", "quantity", "planned_arrival", "actual_arrival", "sequence")


class DeliveryListSerializer(serializers.ModelSerializer):
    """Serializer for delivery list responses."""

    vehicle = VehicleSerializer(read_only=True)

    class Meta:
        """DRF metadata for list-level Delivery serialization."""

        model = Delivery
        fields = ("id", "depot", "vehicle", "planned_date", "status", "total_distance_km", "total_cost")


class DeliveryDetailSerializer(DeliveryListSerializer):
    """Serializer for detailed delivery responses."""

    items = DeliveryItemSerializer(many=True, read_only=True)

    class Meta(DeliveryListSerializer.Meta):
        """DRF metadata for detail-level Delivery serialization."""

        fields = (*DeliveryListSerializer.Meta.fields, *("route_json", "solver_time_ms", "items"))


class DeliveryStatusUpdateSerializer(serializers.Serializer):
    """Payload serializer for delivery status transition requests."""

    status = serializers.ChoiceField(choices=Delivery.Status.choices)
    notes = serializers.CharField(required=False, allow_blank=True)


class PlanningCycleSerializer(serializers.ModelSerializer):
    """Serializer for planning cycle history records."""

    class Meta:
        """DRF metadata for PlanningCycle serialization."""

        model = PlanningCycle
        fields = "__all__"


class PlanningTriggerSerializer(serializers.Serializer):
    """Payload serializer for manual and emergency planning triggers."""

    trigger_type = serializers.ChoiceField(choices=["MANUAL", "EMERGENCY"])
    facility_ids = serializers.ListField(child=serializers.IntegerField(), required=False, allow_null=True)

    def validate(self, attrs: dict[str, object]) -> dict[str, object]:
        """Enforce emergency trigger payload requirements."""
        trigger_type = attrs.get("trigger_type")
        facility_ids = attrs.get("facility_ids")
        if trigger_type == "EMERGENCY" and not facility_ids:
            raise serializers.ValidationError(
                {"facility_ids": "Provide at least one facility id for EMERGENCY trigger."},
            )
        return attrs


class ModelRegistrySerializer(serializers.ModelSerializer):
    """Serializer for model registry entries."""

    class Meta:
        """DRF metadata for ModelRegistry serialization."""

        model = ModelRegistry
        fields = "__all__"


class DashboardKPISerializer(serializers.Serializer):
    """Serializer for dashboard KPI payloads."""

    avg_delivery_cost_last_30d = serializers.FloatField()
    forecast_accuracy_rmse = serializers.FloatField()
    anomaly_detection_rate = serializers.FloatField()
    unacknowledged_anomalies_count = serializers.IntegerField()
    facilities_below_reorder = serializers.IntegerField()
    deliveries_in_transit = serializers.IntegerField()


class DriftHeatmapSerializer(serializers.Serializer):
    """Serializer for drift heatmap response payloads."""

    facility_id = serializers.IntegerField(allow_null=True)
    facility_name = serializers.CharField(allow_null=True)
    drift_ratio = serializers.FloatField()
    model_version = serializers.IntegerField()


def latest_forecasts(qs: QuerySet[Forecast]) -> QuerySet[Forecast]:
    """Return forecasts ordered from newest to oldest."""
    return qs.order_by("-created_at")
