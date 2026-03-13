"""Tests for reorder filtering and cache invalidation semantics."""

from __future__ import annotations

import pytest
from django.core.cache import cache

from fuelsense.core.cache import get_or_set_reorder_status
from fuelsense.core.models import Facility
from fuelsense.core.reorder import filter_below_reorder
from fuelsense.core.tests.factories import FacilityFactory


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@pytest.mark.django_db
def test_filter_below_reorder_matches_model_reorder_status_semantics() -> None:
    """Queryset filtering should match Python-level reorder status semantics."""
    critical_null = FacilityFactory(dynamic_reorder_point=None, min_safe_inventory=200.0, current_inventory=180.0)
    ok_zero = FacilityFactory(dynamic_reorder_point=0.0, min_safe_inventory=200.0, current_inventory=220.0)
    warning_dynamic = FacilityFactory(dynamic_reorder_point=250.0, min_safe_inventory=200.0, current_inventory=230.0)
    ok_dynamic = FacilityFactory(dynamic_reorder_point=150.0, min_safe_inventory=100.0, current_inventory=160.0)

    queryset_ids = set(filter_below_reorder(Facility.objects.all()).values_list("id", flat=True))
    python_semantic_ids = {
        facility.id
        for facility in [critical_null, ok_zero, warning_dynamic, ok_dynamic]
        if facility.reorder_status in {"WARNING", "CRITICAL"}
    }

    _check(queryset_ids == python_semantic_ids, "Queryset ids should match Python reorder semantics")


@pytest.mark.django_db
def test_facility_save_invalidates_reorder_cache_after_min_safe_change() -> None:
    """Saving safety threshold changes should invalidate reorder cache entries."""
    facility = FacilityFactory(dynamic_reorder_point=None, min_safe_inventory=200.0, current_inventory=250.0)

    # Warm cache as OK with the original threshold.
    _check(get_or_set_reorder_status(facility.id) == "OK", "Initial status should be OK")
    cache_key = f"cache:facility:{facility.id}:reorder_status"
    _check(cache.get(cache_key) == "OK", "Reorder cache should be warmed with OK")

    # Lower inventory safety threshold should force CRITICAL after cache invalidation.
    facility.min_safe_inventory = 260.0
    facility.save(update_fields=["min_safe_inventory"])

    _check(cache.get(cache_key) is None, "Reorder cache should be invalidated on threshold change")
    _check(get_or_set_reorder_status(facility.id) == "CRITICAL", "Status should recompute to CRITICAL")


@pytest.mark.django_db
def test_facility_save_invalidates_dashboard_kpi_cache() -> None:
    """Saving reorder-related fields should invalidate dashboard KPI cache."""
    facility = FacilityFactory(dynamic_reorder_point=300.0, min_safe_inventory=200.0, current_inventory=310.0)

    payload = {
        "avg_delivery_cost_last_30d": 1.0,
        "forecast_accuracy_rmse": 2.0,
        "anomaly_detection_rate": 3.0,
        "unacknowledged_anomalies_count": 4,
        "facilities_below_reorder": 5,
        "deliveries_in_transit": 6,
    }
    cache.set("cache:dashboard:kpis", payload, timeout=60)
    _check(cache.get("cache:dashboard:kpis") is not None, "Dashboard KPI cache should be primed")

    facility.dynamic_reorder_point = 1000.0
    facility.save(update_fields=["dynamic_reorder_point"])

    _check(cache.get("cache:dashboard:kpis") is None, "Dashboard KPI cache should be invalidated")
