from __future__ import annotations

import datetime as dt

import factory
from django.contrib.auth import get_user_model
from django.utils import timezone

from fuelsense.core import models


class UserFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = get_user_model()

    username = factory.Sequence(lambda n: f"user{n}")
    email = factory.LazyAttribute(lambda o: f"{o.username}@example.com")


class FuelTypeFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.FuelType

    name = factory.Sequence(lambda n: f"Fuel{n}")
    unit = "m3"
    density_kg_per_unit = 850.0
    hazmat_class = "3"


class FacilityFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.Facility

    name = factory.Sequence(lambda n: f"Facility {n}")
    facility_type = models.Facility.FacilityType.POWER_PLANT
    fuel_type = factory.SubFactory(FuelTypeFactory)
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


class DepotFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.Depot

    name = factory.Sequence(lambda n: f"Depot {n}")
    latitude = 24.7
    longitude = 46.7
    fuel_type = factory.SubFactory(FuelTypeFactory)
    fuel_inventory = 50000.0


class DepotFacilityAssignmentFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.DepotFacilityAssignment

    depot = factory.SubFactory(DepotFactory)
    facility = factory.SubFactory(FacilityFactory)


class VehicleFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.Vehicle

    depot = factory.SubFactory(DepotFactory)
    registration = factory.Sequence(lambda n: f"REG-{n:04d}")
    capacity = 10000.0
    cost_per_km = 2.5
    is_available = True


class PlanningCycleFactory(factory.django.DjangoModelFactory):
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


class DeliveryFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.Delivery

    depot = factory.SubFactory(DepotFactory)
    vehicle = factory.SubFactory(VehicleFactory)
    planned_date = factory.LazyFunction(lambda: timezone.now().date())
    status = models.Delivery.Status.PLANNED
    total_distance_km = 100.0
    total_cost = 500.0
    route_json = []
    solver_time_ms = 100.0
    created_by_planning_cycle = factory.SubFactory(PlanningCycleFactory)


class DeliveryItemFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.DeliveryItem

    delivery = factory.SubFactory(DeliveryFactory)
    facility = factory.SubFactory(FacilityFactory)
    quantity = 120.0
    planned_arrival = factory.LazyFunction(timezone.now)
    actual_arrival = None
    sequence = 1


class InventoryLogFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.InventoryLog

    facility = factory.SubFactory(FacilityFactory)
    timestamp = factory.LazyFunction(timezone.now)
    inventory_level = 500.0
    consumption = 20.0
    temperature = 28.0
    wind_speed = 3.0
    solar_irradiance = 600.0


class ForecastFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.Forecast

    facility = factory.SubFactory(FacilityFactory)
    model_version = "v1"
    horizon_days = 14
    predictions_json = [{"day": 1, "p10": 10, "p50": 12, "p90": 15}]
    rmse = 1.1


class AnomalyAlertFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.AnomalyAlert

    facility = factory.SubFactory(FacilityFactory)
    timestamp = factory.LazyFunction(timezone.now)
    anomaly_type = models.AnomalyAlert.AnomalyType.LEAK
    score = 3.5
    actual_consumption = 33.0
    predicted_consumption = 20.0
    is_acknowledged = False
    acknowledged_by = None
    notes = ""


class ModelRegistryFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = models.ModelRegistry

    model_type = models.ModelRegistry.ModelType.DEMAND_FORECAST
    facility = factory.SubFactory(FacilityFactory)
    mlflow_run_id = factory.Sequence(lambda n: f"run-{n}")
    version = 1
    is_active = False
    trained_at = factory.LazyFunction(timezone.now)
    training_rmse = 1.5
    validation_rmse = 1.3
    drift_ratio = 1.0
    last_drift_check = factory.LazyFunction(timezone.now)
