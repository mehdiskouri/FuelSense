from __future__ import annotations

import datetime as dt

import pytest
from django.db import IntegrityError
from django.utils import timezone

from fuelsense.core.models import Facility, InventoryLog
from fuelsense.core.tests.factories import (
    AnomalyAlertFactory,
    DeliveryFactory,
    DepotFactory,
    FacilityFactory,
    ForecastFactory,
    FuelTypeFactory,
    InventoryLogFactory,
    ModelRegistryFactory,
    PlanningCycleFactory,
    VehicleFactory,
)


@pytest.mark.django_db
def test_model_creation_smoke():
    FuelTypeFactory()
    facility = FacilityFactory()
    DepotFactory(fuel_type=facility.fuel_type)
    VehicleFactory()
    InventoryLogFactory(facility=facility)
    DeliveryFactory()
    ForecastFactory(facility=facility)
    AnomalyAlertFactory(facility=facility)
    PlanningCycleFactory()
    ModelRegistryFactory(facility=facility)


@pytest.mark.django_db
def test_reorder_status_property():
    facility = FacilityFactory(current_inventory=500, dynamic_reorder_point=300, min_safe_inventory=200)
    assert facility.reorder_status == "OK"

    facility.current_inventory = 250
    assert facility.reorder_status == "WARNING"

    facility.current_inventory = 150
    assert facility.reorder_status == "CRITICAL"


@pytest.mark.django_db
def test_inventory_log_unique_constraint():
    facility = FacilityFactory()
    ts = timezone.now().replace(microsecond=0)
    InventoryLog.objects.create(
        facility=facility,
        timestamp=ts,
        inventory_level=100,
        consumption=10,
        temperature=20,
        wind_speed=2,
        solar_irradiance=500,
    )
    with pytest.raises(IntegrityError):
        InventoryLog.objects.create(
            facility=facility,
            timestamp=ts,
            inventory_level=95,
            consumption=8,
            temperature=19,
            wind_speed=2,
            solar_irradiance=400,
        )


@pytest.mark.django_db
def test_facility_indexes_declared():
    index_fields = [tuple(idx.fields) for idx in Facility._meta.indexes]
    assert ("fuel_type", "is_active") in index_fields
    assert ("current_inventory",) in index_fields
