from __future__ import annotations

import pytest
from django.urls import reverse

from fuelsense.core.models import AnomalyAlert, Delivery, ModelRegistry
from fuelsense.core.tests.factories import (
    AnomalyAlertFactory,
    DeliveryFactory,
    FacilityFactory,
    ForecastFactory,
    ModelRegistryFactory,
    PlanningCycleFactory,
)


@pytest.mark.django_db
def test_facility_list_and_detail(api_client):
    facility = FacilityFactory()
    res = api_client.get("/api/v1/facilities/")
    assert res.status_code == 200
    assert len(res.data["results"]) >= 1

    detail = api_client.get(f"/api/v1/facilities/{facility.id}/")
    assert detail.status_code == 200
    assert detail.data["id"] == facility.id


@pytest.mark.django_db
def test_facility_inventory_forecasts_alerts_and_ack(api_client):
    facility = FacilityFactory()
    ForecastFactory(facility=facility)
    alert = AnomalyAlertFactory(facility=facility)

    inv = api_client.get(f"/api/v1/facilities/{facility.id}/inventory/?days=90")
    assert inv.status_code == 200

    fc = api_client.get(f"/api/v1/facilities/{facility.id}/forecasts/")
    assert fc.status_code == 200

    alerts = api_client.get(f"/api/v1/facilities/{facility.id}/alerts/")
    assert alerts.status_code == 200

    ack = api_client.post(
        f"/api/v1/facilities/{facility.id}/acknowledge/",
        {"alert_id": alert.id, "notes": "ack"},
        format="json",
    )
    assert ack.status_code == 200
    alert.refresh_from_db()
    assert alert.is_acknowledged is True


@pytest.mark.django_db
def test_delivery_endpoints_and_transition(api_client):
    delivery = DeliveryFactory(status=Delivery.Status.PLANNED)
    lst = api_client.get("/api/v1/deliveries/")
    assert lst.status_code == 200

    detail = api_client.get(f"/api/v1/deliveries/{delivery.id}/")
    assert detail.status_code == 200

    update = api_client.post(
        f"/api/v1/deliveries/{delivery.id}/update-status/",
        {"status": Delivery.Status.IN_TRANSIT},
        format="json",
    )
    assert update.status_code == 200


@pytest.mark.django_db
def test_planning_model_dashboard_endpoints(api_client, monkeypatch):
    ModelRegistryFactory()
    PlanningCycleFactory()

    called = {"tasks": []}

    def fake_send_task(name, args=None, kwargs=None):
        called["tasks"].append(name)

    monkeypatch.setattr("fuelsense.core.views.current_app.send_task", fake_send_task)

    trig = api_client.post("/api/v1/planning/trigger/", {"trigger_type": "MANUAL"}, format="json")
    assert trig.status_code == 202
    assert called["tasks"]

    hist = api_client.get("/api/v1/planning/history/")
    assert hist.status_code == 200

    model = ModelRegistry.objects.first()
    promote = api_client.post(f"/api/v1/models/{model.id}/promote/")
    assert promote.status_code == 200

    retrain = api_client.post(f"/api/v1/models/{model.id}/retrain/")
    assert retrain.status_code == 202

    kpis = api_client.get("/api/v1/dashboard/kpis/")
    assert kpis.status_code == 200

    drift = api_client.get("/api/v1/dashboard/drift-heatmap/")
    assert drift.status_code == 200


@pytest.mark.django_db
def test_auth_required():
    from rest_framework.test import APIClient

    client = APIClient()
    res = client.get("/api/v1/facilities/")
    assert res.status_code in {401, 403}
