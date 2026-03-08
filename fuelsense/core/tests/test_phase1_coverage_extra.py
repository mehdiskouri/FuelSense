from __future__ import annotations

import pytest
from django.test import Client

from fuelsense.core import metrics
from fuelsense.core.cache import get_or_set_dashboard_kpis, get_or_set_facility_inventory, get_or_set_reorder_status
from fuelsense.core.tests.factories import (
    DeliveryItemFactory,
    DeliveryFactory,
    ForecastFactory,
    InventoryLogFactory,
    ModelRegistryFactory,
)


@pytest.mark.django_db
def test_cache_helpers_and_metrics_refresh(sample_facilities):
    f = sample_facilities[0]
    InventoryLogFactory(facility=f)
    ForecastFactory(facility=f)
    DeliveryFactory()
    val = get_or_set_facility_inventory(f.id)
    assert isinstance(val, float)
    status = get_or_set_reorder_status(f.id)
    assert status in {"OK", "WARNING", "CRITICAL"}
    payload = get_or_set_dashboard_kpis()
    assert "avg_delivery_cost_last_30d" in payload
    metrics.refresh_business_gauges()


@pytest.mark.django_db
def test_admin_actions_and_change_views(admin_client):
    facility = ModelRegistryFactory().facility
    InventoryLogFactory(facility=facility)
    ForecastFactory(facility=facility)
    fac_resp = admin_client.get(f"/admin/core/facility/{facility.id}/change/")
    assert fac_resp.status_code == 200

    d = DeliveryFactory()
    DeliveryItemFactory(delivery=d, facility=facility)
    del_resp = admin_client.get(f"/admin/core/delivery/{d.id}/change/")
    assert del_resp.status_code == 200

@pytest.mark.django_db
def test_model_registry_admin_actions(admin_client, monkeypatch):
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
    assert promote_resp.status_code == 200

    rollback_resp = admin_client.post(
        "/admin/core/modelregistry/",
        {
            "action": "rollback_to_previous",
            "_selected_action": [str(m2.id)],
        },
        follow=True,
    )
    assert rollback_resp.status_code == 200

    monkeypatch.setattr("celery.current_app.send_task", lambda *args, **kwargs: None, raising=False)

    emergency_resp = admin_client.post(
        "/admin/core/facility/",
        {
            "action": "trigger_emergency_delivery",
            "_selected_action": [str(facility.id)],
        },
        follow=True,
    )
    assert emergency_resp.status_code == 200


@pytest.mark.django_db
def test_health_and_ready_endpoints():
    client = Client()
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200


def test_import_infra_modules():
    import fuelsense.asgi  # noqa: F401
    import fuelsense.celery  # noqa: F401
    import fuelsense.settings.production  # noqa: F401
    import fuelsense.wsgi  # noqa: F401
    import fuelsense.urls  # noqa: F401
