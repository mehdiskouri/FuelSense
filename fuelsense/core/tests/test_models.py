"""Model-level tests for constraints, properties, indexes, and string representations."""

from __future__ import annotations

import pytest
from django.db import IntegrityError
from django.utils import timezone

from fuelsense.core.models import Facility, InventoryLog
from fuelsense.core.tests.factories import (
    AnomalyAlertFactory,
    DeliveryFactory,
    DepotFacilityAssignmentFactory,
    DepotFactory,
    FacilityFactory,
    ForecastFactory,
    FuelTypeFactory,
    InventoryLogFactory,
    ModelRegistryFactory,
    PlanningCycleFactory,
    VehicleFactory,
)

MIN_SAFE_POINT = 200.0
ELEVATED_POINT = 280.0


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


@pytest.mark.django_db
def test_model_creation_smoke() -> None:
    """Factory smoke test should create core model records successfully."""
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
def test_reorder_status_property() -> None:
    """Reorder status should move across OK, WARNING, and CRITICAL thresholds."""
    facility = FacilityFactory(current_inventory=500, dynamic_reorder_point=300, min_safe_inventory=200)
    _check(facility.reorder_status == "OK")

    facility.current_inventory = 250
    _check(facility.reorder_status == "WARNING")

    facility.current_inventory = 150
    _check(facility.reorder_status == "CRITICAL")


@pytest.mark.django_db
def test_effective_reorder_point_falls_back_for_null_or_non_positive_dynamic() -> None:
    """Effective reorder point should fallback to min-safe until a positive dynamic value exists."""
    facility = FacilityFactory(current_inventory=250, dynamic_reorder_point=None, min_safe_inventory=200)
    _check(facility.effective_reorder_point == MIN_SAFE_POINT)
    _check(facility.reorder_status == "OK")

    facility.dynamic_reorder_point = 0.0
    facility.current_inventory = 180
    _check(facility.effective_reorder_point == MIN_SAFE_POINT)
    _check(facility.reorder_status == "CRITICAL")

    facility.dynamic_reorder_point = ELEVATED_POINT
    facility.current_inventory = 250
    _check(facility.effective_reorder_point == ELEVATED_POINT)
    _check(facility.reorder_status == "WARNING")


@pytest.mark.django_db
def test_inventory_log_unique_constraint() -> None:
    """Inventory log uniqueness should enforce one row per facility timestamp."""
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
def test_facility_indexes_declared() -> None:
    """Facility model should include expected indexes for query-heavy columns."""
    meta_attr = "_meta"
    model_meta = getattr(Facility, meta_attr)
    index_fields = [tuple(idx.fields) for idx in model_meta.indexes]
    _check(("fuel_type", "is_active") in index_fields)
    _check(("current_inventory",) in index_fields)


@pytest.mark.django_db
def test_model_string_representations_cover_expected_formats() -> None:
    """String representations should match the canonical user-facing formats."""
    assignment = DepotFacilityAssignmentFactory()
    forecast = ForecastFactory(facility=assignment.facility, model_version="v-test")
    alert = AnomalyAlertFactory(facility=assignment.facility)

    _check(str(assignment) == f"{assignment.depot} -> {assignment.facility}")
    _check(str(forecast) == f"Forecast {forecast.facility_id} vv-test")
    _check(str(alert) == f"{alert.anomaly_type} @ facility {alert.facility_id}")
