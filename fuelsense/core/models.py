"""Core domain models for FuelSense."""

from __future__ import annotations

from django.conf import settings
from django.db import models

from fuelsense.core.reorder import get_effective_reorder_point


class FuelType(models.Model):
    """Catalog entry for a fuel commodity."""

    name = models.CharField(max_length=50)
    unit = models.CharField(max_length=20)
    density_kg_per_unit = models.FloatField()
    hazmat_class = models.CharField(max_length=10, blank=True)

    def __str__(self) -> str:
        """Return the display name for admin and logs."""
        return self.name


class Facility(models.Model):
    """Operational site that stores and consumes fuel."""

    class FacilityType(models.TextChoices):
        """Supported facility categories."""

        POWER_PLANT = "POWER_PLANT", "Power Plant"
        INDUSTRIAL = "INDUSTRIAL", "Industrial"
        STORAGE = "STORAGE", "Storage"

    name = models.CharField(max_length=200)
    facility_type = models.CharField(max_length=20, choices=FacilityType.choices)
    fuel_type = models.ForeignKey(FuelType, on_delete=models.PROTECT)
    latitude = models.FloatField()
    longitude = models.FloatField()
    storage_capacity = models.FloatField()
    current_inventory = models.FloatField()
    min_safe_inventory = models.FloatField()
    dynamic_reorder_point = models.FloatField(null=True)
    delivery_window_start = models.TimeField()
    delivery_window_end = models.TimeField()
    delivery_days = models.JSONField(default=list)
    is_active = models.BooleanField(default=True)

    class Meta:
        """Model indexes for common facility filters."""

        indexes = (
            models.Index(fields=["fuel_type", "is_active"]),
            models.Index(fields=["current_inventory"]),
        )

    def __str__(self) -> str:
        """Return facility name for display contexts."""
        return self.name

    @property
    def effective_reorder_point(self) -> float:
        """Return dynamic threshold with min-safe fallback semantics."""
        return get_effective_reorder_point(self.dynamic_reorder_point, self.min_safe_inventory)

    @property
    def reorder_status(self) -> str:
        """Compute qualitative inventory risk against reorder thresholds."""
        reorder_point = self.effective_reorder_point
        if self.current_inventory <= self.min_safe_inventory:
            return "CRITICAL"
        if self.current_inventory <= reorder_point:
            return "WARNING"
        return "OK"


class Depot(models.Model):
    """Supply origin servicing one or more facilities."""

    name = models.CharField(max_length=200)
    latitude = models.FloatField()
    longitude = models.FloatField()
    fuel_type = models.ForeignKey(FuelType, on_delete=models.PROTECT)
    fuel_inventory = models.FloatField()
    facilities: models.ManyToManyField[Facility, DepotFacilityAssignment] = models.ManyToManyField(
        Facility,
        through="DepotFacilityAssignment",
    )

    def __str__(self) -> str:
        """Return depot name for display contexts."""
        return self.name


class DepotFacilityAssignment(models.Model):
    """Many-to-many bridge linking depots to facilities."""

    depot = models.ForeignKey(Depot, on_delete=models.CASCADE)
    facility = models.ForeignKey(Facility, on_delete=models.CASCADE)

    class Meta:
        """Enforce unique assignment per depot/facility pair."""

        unique_together = (("depot", "facility"),)

    def __str__(self) -> str:
        """Return a readable depot-to-facility mapping string."""
        return f"{self.depot} -> {self.facility}"


class Vehicle(models.Model):
    """Transport unit available for planning and delivery."""

    depot = models.ForeignKey(Depot, on_delete=models.CASCADE, related_name="vehicles")
    registration = models.CharField(max_length=50, unique=True)
    capacity = models.FloatField()
    cost_per_km = models.FloatField()
    is_available = models.BooleanField(default=True)

    def __str__(self) -> str:
        """Return vehicle registration identifier."""
        return self.registration


class InventoryLog(models.Model):
    """Time-series inventory and weather snapshot for a facility."""

    facility = models.ForeignKey(Facility, on_delete=models.CASCADE, related_name="inventory_logs")
    timestamp = models.DateTimeField()
    inventory_level = models.FloatField()
    consumption = models.FloatField()
    temperature = models.FloatField(null=True)
    wind_speed = models.FloatField(null=True)
    solar_irradiance = models.FloatField(null=True)

    class Meta:
        """Optimize facility time-series access and uniqueness."""

        indexes = (models.Index(fields=["facility", "-timestamp"]),)
        unique_together = (("facility", "timestamp"),)

    def __str__(self) -> str:
        """Return compact facility/timestamp label."""
        return f"{self.facility.name} @ {self.timestamp.isoformat()}"


class PlanningCycle(models.Model):
    """Planning execution lifecycle and aggregate optimization metrics."""

    class TriggerType(models.TextChoices):
        """Planning invocation source."""

        SCHEDULED = "SCHEDULED", "Scheduled"
        MANUAL = "MANUAL", "Manual"
        EMERGENCY = "EMERGENCY", "Emergency"

    class ExecutionStatus(models.TextChoices):
        """Lifecycle state for planning execution."""

        QUEUED = "QUEUED", "Queued"
        RUNNING = "RUNNING", "Running"
        COMPLETED = "COMPLETED", "Completed"
        FAILED = "FAILED", "Failed"

    triggered_at = models.DateTimeField(auto_now_add=True)
    trigger_type = models.CharField(max_length=20, choices=TriggerType.choices)
    status = models.CharField(
        max_length=20,
        choices=ExecutionStatus.choices,
        default=ExecutionStatus.COMPLETED,
    )
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    facilities_in_queue = models.IntegerField()
    deliveries_created = models.IntegerField()
    total_distance_km = models.FloatField()
    total_cost = models.FloatField()
    solver_time_ms = models.FloatField()
    baseline_cost = models.FloatField(null=True)
    cost_reduction_pct = models.FloatField(null=True)

    class Meta:
        """Indexes for planning history and lifecycle queries."""

        indexes = (
            models.Index(fields=["status", "trigger_type"]),
            models.Index(fields=["-triggered_at", "status"]),
        )

    def __str__(self) -> str:
        """Return trigger/status label for operator visibility."""
        return f"PlanningCycle {self.id} ({self.trigger_type}/{self.status})"


class Delivery(models.Model):
    """Route plan assigned to one vehicle for one planning date."""

    class Status(models.TextChoices):
        """Delivery execution states."""

        PLANNED = "PLANNED", "Planned"
        IN_TRANSIT = "IN_TRANSIT", "In Transit"
        DELIVERED = "DELIVERED", "Delivered"
        FAILED = "FAILED", "Failed"

    depot = models.ForeignKey(Depot, on_delete=models.CASCADE)
    vehicle = models.ForeignKey(Vehicle, on_delete=models.CASCADE)
    planned_date = models.DateField()
    status = models.CharField(max_length=20, choices=Status.choices)
    total_distance_km = models.FloatField(null=True)
    total_cost = models.FloatField(null=True)
    route_json = models.JSONField(null=True)
    solver_time_ms = models.FloatField(null=True)
    idempotency_key = models.CharField(max_length=64, null=True, blank=True, unique=True, db_index=True)
    created_by_planning_cycle = models.ForeignKey(PlanningCycle, null=True, on_delete=models.SET_NULL)

    def __str__(self) -> str:
        """Return compact delivery id and status."""
        return f"Delivery {self.id} ({self.status})"


class DeliveryItem(models.Model):
    """Single stop within a delivery route."""

    delivery = models.ForeignKey(Delivery, on_delete=models.CASCADE, related_name="items")
    facility = models.ForeignKey(Facility, on_delete=models.CASCADE)
    quantity = models.FloatField()
    planned_arrival = models.DateTimeField()
    actual_arrival = models.DateTimeField(null=True)
    sequence = models.PositiveIntegerField()

    def __str__(self) -> str:
        """Return delivery and stop sequence label."""
        return f"Delivery {self.delivery_id} stop {self.sequence}"


class Forecast(models.Model):
    """Persisted demand forecast artifact for a facility."""

    facility = models.ForeignKey(Facility, on_delete=models.CASCADE, related_name="forecasts")
    model_version = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    horizon_days = models.IntegerField(default=14)
    predictions_json = models.JSONField()
    rmse = models.FloatField(null=True)

    def __str__(self) -> str:
        """Return compact forecast identity string."""
        return f"Forecast {self.facility_id} v{self.model_version}"


class AnomalyAlert(models.Model):
    """Detected anomaly event requiring operational review."""

    class AnomalyType(models.TextChoices):
        """Anomaly categories emitted by the detector."""

        LEAK = "LEAK", "Leak"
        THEFT = "THEFT", "Theft"
        EQUIPMENT_DEGRADATION = "EQUIPMENT_DEGRADATION", "Equipment Degradation"
        DEMAND_SHIFT = "DEMAND_SHIFT", "Demand Shift"
        SENSOR_FAULT = "SENSOR_FAULT", "Sensor Fault"

    facility = models.ForeignKey(Facility, on_delete=models.CASCADE, related_name="alerts")
    timestamp = models.DateTimeField()
    anomaly_type = models.CharField(max_length=30, choices=AnomalyType.choices)
    score = models.FloatField()
    actual_consumption = models.FloatField()
    predicted_consumption = models.FloatField()
    is_acknowledged = models.BooleanField(default=False)
    acknowledged_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    notes = models.TextField(blank=True)

    def __str__(self) -> str:
        """Return anomaly type and facility identifier."""
        return f"{self.anomaly_type} @ facility {self.facility_id}"


class ModelRegistry(models.Model):
    """Version registry for demand and anomaly models."""

    class ModelType(models.TextChoices):
        """Supported model families in registry."""

        DEMAND_FORECAST = "DEMAND_FORECAST", "Demand Forecast"
        ANOMALY_DETECTOR = "ANOMALY_DETECTOR", "Anomaly Detector"

    model_type = models.CharField(max_length=30, choices=ModelType.choices)
    facility = models.ForeignKey(Facility, null=True, on_delete=models.CASCADE)
    mlflow_run_id = models.CharField(max_length=100)
    version = models.IntegerField()
    is_active = models.BooleanField(default=False)
    trained_at = models.DateTimeField()
    training_rmse = models.FloatField(null=True)
    validation_rmse = models.FloatField(null=True)
    drift_ratio = models.FloatField(default=1.0)
    last_drift_check = models.DateTimeField(null=True)

    def __str__(self) -> str:
        """Return model family/facility/version identifier."""
        facility_id = self.facility_id if self.facility_id is not None else "global"
        return f"{self.model_type}:{facility_id}:v{self.version}"
