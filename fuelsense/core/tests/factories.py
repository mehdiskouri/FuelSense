from __future__ import annotations

# pyright: reportIncompatibleVariableOverride=false
import datetime as dt
from typing import Any

from django.contrib.auth import get_user_model
from django.utils import timezone
from factory.declarations import LazyFunction, Sequence, SubFactory
from factory.django import DjangoModelFactory

from fuelsense.core import models


def _user_seq(n: int) -> str:
    return f"user{n}"


def _user_email_seq(n: int) -> str:
    return f"user{n}@example.com"


def _fuel_name_seq(n: int) -> str:
    return f"Fuel{n}"


def _facility_name_seq(n: int) -> str:
    return f"Facility {n}"


def _depot_name_seq(n: int) -> str:
    return f"Depot {n}"


def _registration_seq(n: int) -> str:
    return f"REG-{n:04d}"


def _run_id_seq(n: int) -> str:
    return f"run-{n}"


def _today_date() -> dt.date:
    return timezone.now().date()


class UserFactory(DjangoModelFactory[Any]):
    class Meta:
        model = get_user_model()

    username = Sequence(_user_seq)
    email = Sequence(_user_email_seq)


class FuelTypeFactory(DjangoModelFactory[models.FuelType]):
    class Meta:
        model = models.FuelType

    name = Sequence(_fuel_name_seq)
    unit = "m3"
    density_kg_per_unit = 850.0
    hazmat_class = "3"


class FacilityFactory(DjangoModelFactory[models.Facility]):
    class Meta:
        model = models.Facility

    name = Sequence(_facility_name_seq)
    facility_type = models.Facility.FacilityType.POWER_PLANT
    fuel_type = SubFactory(FuelTypeFactory)
    latitude = 24.7
    longitude = 46.6
    storage_capacity = 1000.0
    current_inventory = 700.0
    min_safe_inventory = 200.0
    dynamic_reorder_point = 300.0
    delivery_window_start = dt.time(hour=6)
    delivery_window_end = dt.time(hour=14)
    delivery_days = [0, 1, 2, 3, 4]
    is_active = True


class DepotFactory(DjangoModelFactory[models.Depot]):
    class Meta:
        model = models.Depot

    name = Sequence(_depot_name_seq)
    latitude = 24.7
    longitude = 46.7
    fuel_type = SubFactory(FuelTypeFactory)
    fuel_inventory = 50000.0


class DepotFacilityAssignmentFactory(DjangoModelFactory[models.DepotFacilityAssignment]):
    class Meta:
        model = models.DepotFacilityAssignment

    depot = SubFactory(DepotFactory)
    facility = SubFactory(FacilityFactory)


class VehicleFactory(DjangoModelFactory[models.Vehicle]):
    class Meta:
        model = models.Vehicle

    depot = SubFactory(DepotFactory)
    registration = Sequence(_registration_seq)
    capacity = 10000.0
    cost_per_km = 2.5
    is_available = True


class PlanningCycleFactory(DjangoModelFactory[models.PlanningCycle]):
    class Meta:
        model = models.PlanningCycle

    trigger_type = models.PlanningCycle.TriggerType.MANUAL
    facilities_in_queue = 2
    deliveries_created = 1
    total_distance_km = 200.0
    total_cost = 1000.0
    solver_time_ms = 300.0
    baseline_cost = 1200.0
    cost_reduction_pct = 16.7


class DeliveryFactory(DjangoModelFactory[models.Delivery]):
    class Meta:
        model = models.Delivery

    depot = SubFactory(DepotFactory)
    vehicle = SubFactory(VehicleFactory)
    planned_date = LazyFunction(_today_date)
    status = models.Delivery.Status.PLANNED
    total_distance_km = 100.0
    total_cost = 500.0
    route_json = []
    solver_time_ms = 100.0
    created_by_planning_cycle = SubFactory(PlanningCycleFactory)


class DeliveryItemFactory(DjangoModelFactory[models.DeliveryItem]):
    class Meta:
        model = models.DeliveryItem

    delivery = SubFactory(DeliveryFactory)
    facility = SubFactory(FacilityFactory)
    quantity = 120.0
    planned_arrival = LazyFunction(timezone.now)
    actual_arrival = None
    sequence = 1


class InventoryLogFactory(DjangoModelFactory[models.InventoryLog]):
    class Meta:
        model = models.InventoryLog

    facility = SubFactory(FacilityFactory)
    timestamp = LazyFunction(timezone.now)
    inventory_level = 500.0
    consumption = 20.0
    temperature = 28.0
    wind_speed = 3.0
    solar_irradiance = 600.0


class ForecastFactory(DjangoModelFactory[models.Forecast]):
    class Meta:
        model = models.Forecast

    facility = SubFactory(FacilityFactory)
    model_version = "v1"
    horizon_days = 14
    predictions_json = [{"day": 1, "p10": 10, "p50": 12, "p90": 15}]
    rmse = 1.1


class AnomalyAlertFactory(DjangoModelFactory[models.AnomalyAlert]):
    class Meta:
        model = models.AnomalyAlert

    facility = SubFactory(FacilityFactory)
    timestamp = LazyFunction(timezone.now)
    anomaly_type = models.AnomalyAlert.AnomalyType.LEAK
    score = 3.5
    actual_consumption = 33.0
    predicted_consumption = 20.0
    is_acknowledged = False
    acknowledged_by = None
    notes = ""


class ModelRegistryFactory(DjangoModelFactory[models.ModelRegistry]):
    class Meta:
        model = models.ModelRegistry

    model_type = models.ModelRegistry.ModelType.DEMAND_FORECAST
    facility = SubFactory(FacilityFactory)
    mlflow_run_id = Sequence(_run_id_seq)
    version = 1
    is_active = False
    trained_at = LazyFunction(timezone.now)
    training_rmse = 1.5
    validation_rmse = 1.3
    drift_ratio = 1.0
    last_drift_check = LazyFunction(timezone.now)
