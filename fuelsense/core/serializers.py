"""DRF serializers for FuelSense core app."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict

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


class AnomalyAlertAcknowledgeData(TypedDict, total=False):
    """Validated payload for acknowledging an anomaly alert."""

    alert_id: int
    notes: str


class DeliveryStatusUpdateData(TypedDict, total=False):
    """Validated payload for transitioning delivery status."""

    status: str
    notes: str


class PlanningTriggerData(TypedDict, total=False):
    """Validated payload for manual or emergency planning trigger."""

    trigger_type: Literal["MANUAL", "EMERGENCY"]
    facility_ids: list[int] | None


class DashboardKPIData(TypedDict):
    """Dashboard KPI response payload schema."""

    avg_delivery_cost_last_30d: float
    forecast_accuracy_rmse: float
    anomaly_detection_rate: float
    unacknowledged_anomalies_count: int
    facilities_below_reorder: int
    deliveries_in_transit: int


class DriftHeatmapPoint(TypedDict):
    """Per-model drift heatmap response payload schema."""

    facility_id: int | None
    facility_name: str | None
    drift_ratio: float
    model_version: int


class FuelTypeSerializer(serializers.ModelSerializer[FuelType]):
    """Serializer for fuel type records."""

    class Meta:
        """DRF metadata for FuelType serialization."""

        model = FuelType
        fields = "__all__"


class VehicleSerializer(serializers.ModelSerializer[Vehicle]):
    """Serializer for vehicle records."""

    class Meta:
        """DRF metadata for Vehicle serialization."""

        model = Vehicle
        fields = ("id", "registration", "capacity", "cost_per_km", "is_available")


class InventoryLogSerializer(serializers.ModelSerializer[InventoryLog]):
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


class ForecastSerializer(serializers.ModelSerializer[Forecast]):
    """Serializer for persisted forecasts."""

    class Meta:
        """DRF metadata for Forecast serialization."""

        model = Forecast
        fields = ("id", "model_version", "created_at", "horizon_days", "predictions_json", "rmse")


class AnomalyAlertSerializer(serializers.ModelSerializer[AnomalyAlert]):
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


class AnomalyAlertAcknowledgeSerializer(serializers.Serializer[AnomalyAlertAcknowledgeData]):
    """Payload serializer for alert acknowledgement requests."""

    alert_id = serializers.IntegerField()
    notes = serializers.CharField(required=False, allow_blank=True)


class FacilityListSerializer(serializers.ModelSerializer[Facility]):
    """Serializer for facility list responses."""

    fuel_type = FuelTypeSerializer(read_only=True)
    reorder_status = serializers.CharField(read_only=True)

    class Meta:
        """DRF metadata for list-level Facility serialization."""

        model = Facility
        fields: ClassVar[tuple[str, ...]] = (
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

        fields: ClassVar[tuple[str, ...]] = (
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

    def get_latest_forecast(self, obj: Facility) -> dict[str, object] | None:
        """Return the latest forecast payload when available."""
        latest = obj.forecasts.order_by("-created_at").first()
        return ForecastSerializer(latest).data if latest else None

    def get_latest_inventory_log(self, obj: Facility) -> dict[str, object] | None:
        """Return the latest inventory observation when available."""
        latest = obj.inventory_logs.order_by("-timestamp").first()
        return InventoryLogSerializer(latest).data if latest else None


class DeliveryItemSerializer(serializers.ModelSerializer[DeliveryItem]):
    """Serializer for delivery stop items."""

    facility = FacilityListSerializer(read_only=True)

    class Meta:
        """DRF metadata for DeliveryItem serialization."""

        model = DeliveryItem
        fields = ("id", "facility", "quantity", "planned_arrival", "actual_arrival", "sequence")


class DeliveryListSerializer(serializers.ModelSerializer[Delivery]):
    """Serializer for delivery list responses."""

    vehicle = VehicleSerializer(read_only=True)

    class Meta:
        """DRF metadata for list-level Delivery serialization."""

        model = Delivery
        fields: ClassVar[tuple[str, ...]] = (
            "id",
            "depot",
            "vehicle",
            "planned_date",
            "status",
            "total_distance_km",
            "total_cost",
        )


class DeliveryDetailSerializer(DeliveryListSerializer):
    """Serializer for detailed delivery responses."""

    items = DeliveryItemSerializer(many=True, read_only=True)

    class Meta(DeliveryListSerializer.Meta):
        """DRF metadata for detail-level Delivery serialization."""

        fields: ClassVar[tuple[str, ...]] = (
            *DeliveryListSerializer.Meta.fields,
            "route_json",
            "solver_time_ms",
            "items",
        )


class DeliveryStatusUpdateSerializer(serializers.Serializer[DeliveryStatusUpdateData]):
    """Payload serializer for delivery status transition requests."""

    status = serializers.ChoiceField(choices=Delivery.Status.choices)
    notes = serializers.CharField(required=False, allow_blank=True)


class PlanningCycleSerializer(serializers.ModelSerializer[PlanningCycle]):
    """Serializer for planning cycle history records."""

    class Meta:
        """DRF metadata for PlanningCycle serialization."""

        model = PlanningCycle
        fields = "__all__"


class PlanningTriggerSerializer(serializers.Serializer[PlanningTriggerData]):
    """Payload serializer for manual and emergency planning triggers."""

    trigger_type = serializers.ChoiceField(choices=["MANUAL", "EMERGENCY"])
    facility_ids = serializers.ListField(child=serializers.IntegerField(), required=False, allow_null=True)

    def validate(self, attrs: PlanningTriggerData) -> PlanningTriggerData:
        """Enforce emergency trigger payload requirements."""
        trigger_type = attrs.get("trigger_type")
        facility_ids = attrs.get("facility_ids")
        if trigger_type == "EMERGENCY" and not facility_ids:
            raise serializers.ValidationError(
                {"facility_ids": "Provide at least one facility id for EMERGENCY trigger."},
            )
        return attrs


class ModelRegistrySerializer(serializers.ModelSerializer[ModelRegistry]):
    """Serializer for model registry entries."""

    class Meta:
        """DRF metadata for ModelRegistry serialization."""

        model = ModelRegistry
        fields = "__all__"


class DashboardKPISerializer(serializers.Serializer[DashboardKPIData]):
    """Serializer for dashboard KPI payloads."""

    avg_delivery_cost_last_30d = serializers.FloatField()
    forecast_accuracy_rmse = serializers.FloatField()
    anomaly_detection_rate = serializers.FloatField()
    unacknowledged_anomalies_count = serializers.IntegerField()
    facilities_below_reorder = serializers.IntegerField()
    deliveries_in_transit = serializers.IntegerField()


class DriftHeatmapSerializer(serializers.Serializer[DriftHeatmapPoint]):
    """Serializer for drift heatmap response payloads."""

    facility_id = serializers.IntegerField(allow_null=True)
    facility_name = serializers.CharField(allow_null=True)
    drift_ratio = serializers.FloatField()
    model_version = serializers.IntegerField()


def latest_forecasts(qs: QuerySet[Forecast]) -> QuerySet[Forecast]:
    """Return forecasts ordered from newest to oldest."""
    return qs.order_by("-created_at")
