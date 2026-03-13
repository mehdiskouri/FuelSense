"""Extra phase-1 coverage tests for admin, cache, health, and infra imports."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, cast

import pytest
from django.core.cache import cache
from django.test import Client

from fuelsense.core import metrics
from fuelsense.core.cache import (
    _is_dashboard_kpis,
    _to_float,
    get_or_set_dashboard_kpis,
    get_or_set_facility_inventory,
    get_or_set_reorder_status,
    invalidate_facility_cache,
)
from fuelsense.core.tests.factories import (
    DeliveryFactory,
    DeliveryItemFactory,
    ForecastFactory,
    InventoryLogFactory,
    ModelRegistryFactory,
)

if TYPE_CHECKING:
    from fuelsense.core.models import Facility

HTTP_OK = 200
THREE_AS_FLOAT = 3.0
CACHED_INVENTORY_VALUE = 123.4
IN_TRANSIT_COUNT = 6


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


@pytest.mark.django_db
def test_cache_helpers_and_metrics_refresh(sample_facilities: list[object]) -> None:
    """Cache helpers and gauge refresh should execute on seeded facility data."""
    f = cast("Facility", sample_facilities[0])
    facility_id = cast("int", f.pk)
    InventoryLogFactory(facility=f)
    ForecastFactory(facility=f)
    DeliveryFactory()
    val = get_or_set_facility_inventory(facility_id)
    _check(isinstance(val, float))
    status = get_or_set_reorder_status(facility_id)
    _check(status in {"OK", "WARNING", "CRITICAL"})
    payload = get_or_set_dashboard_kpis()
    _check("avg_delivery_cost_last_30d" in payload)
    metrics.refresh_business_gauges()


@pytest.mark.django_db
def test_admin_actions_and_change_views(admin_client: Client) -> None:
    """Admin change pages for facility and delivery should be reachable."""
    facility = ModelRegistryFactory().facility
    InventoryLogFactory(facility=facility)
    ForecastFactory(facility=facility)
    fac_resp = admin_client.get(f"/admin/core/facility/{facility.id}/change/")
    _check(fac_resp.status_code == HTTP_OK)

    d = DeliveryFactory()
    DeliveryItemFactory(delivery=d, facility=facility)
    del_resp = admin_client.get(f"/admin/core/delivery/{d.id}/change/")
    _check(del_resp.status_code == HTTP_OK)


@pytest.mark.django_db
def test_model_registry_admin_actions(admin_client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    """Model registry admin actions should execute and return successful responses."""
    m1 = ModelRegistryFactory(version=1)
    m2 = ModelRegistryFactory(facility=m1.facility, model_type=m1.model_type, version=2)
    facility = m1.facility

    promote_resp = admin_client.post(
        "/admin/core/modelregistry/",
        {
            "action": "promote_to_active",
            "_selected_action": [str(m1.id), str(m2.id)],
        },
        follow=True,
    )
    _check(promote_resp.status_code == HTTP_OK)

    rollback_resp = admin_client.post(
        "/admin/core/modelregistry/",
        {
            "action": "rollback_to_previous",
            "_selected_action": [str(m2.id)],
        },
        follow=True,
    )
    _check(rollback_resp.status_code == HTTP_OK)

    def _noop_send_task(*args: object, **kwargs: object) -> None:
        _ = args, kwargs

    monkeypatch.setattr("celery.current_app.send_task", _noop_send_task, raising=False)

    emergency_resp = admin_client.post(
        "/admin/core/facility/",
        {
            "action": "trigger_emergency_delivery",
            "_selected_action": [str(facility.id)],
        },
        follow=True,
    )
    _check(emergency_resp.status_code == HTTP_OK)


@pytest.mark.django_db
def test_health_and_ready_endpoints() -> None:
    """Health and readiness endpoints should both return success."""
    client = Client()
    _check(client.get("/healthz").status_code == HTTP_OK)
    _check(client.get("/readyz").status_code == HTTP_OK)


@pytest.mark.django_db
def test_cache_helper_branches(sample_facilities: list[object]) -> None:
    """Type guards and cached helper branches should behave predictably."""
    facility = cast("Facility", sample_facilities[0])
    facility_id = cast("int", facility.pk)

    # Direct helper branch coverage.
    _check(_to_float(None) == 0.0)
    _check(_to_float(3) == THREE_AS_FLOAT)
    _check(_is_dashboard_kpis("not-a-dict") is False)
    _check(_is_dashboard_kpis({"avg_delivery_cost_last_30d": 1}) is False)

    # Cached-path coverage for facility helpers.
    cache.set(f"cache:facility:{facility_id}:latest_inventory", CACHED_INVENTORY_VALUE, timeout=10)
    _check(get_or_set_facility_inventory(facility_id) == CACHED_INVENTORY_VALUE)

    cache.set(f"cache:facility:{facility_id}:reorder_status", "OK", timeout=10)
    _check(get_or_set_reorder_status(facility_id) == "OK")

    # Cached dashboard KPI payload branch coverage.
    kpi_payload = {
        "avg_delivery_cost_last_30d": 1.0,
        "forecast_accuracy_rmse": 2.0,
        "anomaly_detection_rate": 3.0,
        "unacknowledged_anomalies_count": 4,
        "facilities_below_reorder": 5,
        "deliveries_in_transit": 6,
    }
    _check(_is_dashboard_kpis(kpi_payload) is True)
    cache.set("cache:dashboard:kpis", kpi_payload, timeout=10)
    cached_kpis = get_or_set_dashboard_kpis()
    _check(cached_kpis["deliveries_in_transit"] == IN_TRANSIT_COUNT)

    invalidate_facility_cache(facility_id)
    _check(cache.get(f"cache:facility:{facility_id}:latest_inventory") is None)
    _check(cache.get(f"cache:facility:{facility_id}:reorder_status") is None)


def test_import_infra_modules() -> None:
    """Importing deployment modules should not raise import-time exceptions."""
    _check(importlib.import_module("fuelsense.asgi") is not None)
    _check(importlib.import_module("fuelsense.celery") is not None)
    _check(importlib.import_module("fuelsense.settings.production") is not None)
    _check(importlib.import_module("fuelsense.wsgi") is not None)
    _check(importlib.import_module("fuelsense.urls") is not None)
