"""API-level tests for core DRF endpoints and task triggers."""

from __future__ import annotations

# pyright: reportMissingTypeStubs=false
from typing import Protocol, cast
from unittest.mock import patch

import pytest
from django.core.cache import cache
from django.test import Client

from fuelsense.core.models import Delivery, ModelRegistry, PlanningCycle
from fuelsense.core.tests.factories import (
    create_anomaly_alert,
    create_delivery,
    create_facility,
    create_forecast,
    create_model_registry,
    create_planning_cycle,
)

HTTP_OK = 200
HTTP_ACCEPTED = 202
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
FACILITY_ID_ONE = 1
FACILITY_ID_TWO = 2


class _ApiResponse(Protocol):
    status_code: int
    data: object


class _ApiClient(Protocol):
    def get(self, path: str, data: object | None = None, **extra: object) -> object: ...

    def post(
        self,
        path: str,
        data: object | None = None,
        request_format: str | None = None,
        **extra: object,
    ) -> object: ...


def _get(api_client: object, path: str) -> _ApiResponse:
    return cast("_ApiResponse", cast("_ApiClient", api_client).get(path))


def _post(api_client: object, path: str, payload: object) -> _ApiResponse:
    return cast("_ApiResponse", cast("_ApiClient", api_client).post(path, payload, format="json"))


def _data_dict(response: _ApiResponse) -> dict[str, object]:
    return cast("dict[str, object]", response.data)


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


@pytest.mark.django_db
def test_facility_list_and_detail(api_client: object) -> None:
    """Facility list/detail endpoints should return created facility records."""
    facility = create_facility()
    res = _get(api_client, "/api/v1/facilities/")
    _check(res.status_code == HTTP_OK)
    _check(len(cast("list[object]", _data_dict(res)["results"])) >= 1)

    detail = _get(api_client, f"/api/v1/facilities/{facility.id}/")
    _check(detail.status_code == HTTP_OK)
    _check(_data_dict(detail)["id"] == facility.id)


@pytest.mark.django_db
def test_facility_inventory_forecasts_alerts_and_ack(api_client: object) -> None:
    """Facility inventory, forecasts, alerts, and acknowledge actions should succeed."""
    facility = create_facility()
    create_forecast(facility=facility)
    alert = create_anomaly_alert(facility=facility)

    inv = _get(api_client, f"/api/v1/facilities/{facility.id}/inventory/?days=90")
    _check(inv.status_code == HTTP_OK)

    fc = _get(api_client, f"/api/v1/facilities/{facility.id}/forecasts/")
    _check(fc.status_code == HTTP_OK)

    alerts = _get(api_client, f"/api/v1/facilities/{facility.id}/alerts/")
    _check(alerts.status_code == HTTP_OK)

    ack = _post(api_client, f"/api/v1/facilities/{facility.id}/acknowledge/", {"alert_id": alert.id, "notes": "ack"})
    _check(ack.status_code == HTTP_OK)
    alert.refresh_from_db()
    _check(alert.is_acknowledged is True)


@pytest.mark.django_db
def test_facility_forecasts_404_and_acknowledge_not_found(api_client: object) -> None:
    """Forecast and acknowledge endpoints should return 404 for missing records."""
    facility = create_facility()
    fc = _get(api_client, f"/api/v1/facilities/{facility.id}/forecasts/")
    _check(fc.status_code == HTTP_NOT_FOUND)

    ack = _post(api_client, f"/api/v1/facilities/{facility.id}/acknowledge/", {"alert_id": 999999, "notes": "missing"})
    _check(ack.status_code == HTTP_NOT_FOUND)


@pytest.mark.django_db
def test_facility_alert_filters(api_client: object) -> None:
    """Alert filters should return only matching anomaly-type/ack-state records."""
    facility = create_facility()
    create_anomaly_alert(facility=facility, anomaly_type="LEAK", is_acknowledged=True)
    create_anomaly_alert(facility=facility, anomaly_type="THEFT", is_acknowledged=False)

    res = _get(api_client, f"/api/v1/facilities/{facility.id}/alerts/?anomaly_type=LEAK&is_acknowledged=true")
    _check(res.status_code == HTTP_OK)
    _check(_data_dict(res)["count"] == 1)


@pytest.mark.django_db
def test_delivery_endpoints_and_transition(api_client: object) -> None:
    """Delivery list/detail and valid status transition endpoints should succeed."""
    delivery = create_delivery(status=Delivery.Status.PLANNED)
    lst = _get(api_client, "/api/v1/deliveries/")
    _check(lst.status_code == HTTP_OK)

    detail = _get(api_client, f"/api/v1/deliveries/{delivery.id}/")
    _check(detail.status_code == HTTP_OK)

    update = _post(
        api_client,
        f"/api/v1/deliveries/{delivery.id}/update-status/",
        {"status": Delivery.Status.IN_TRANSIT},
    )
    _check(update.status_code == HTTP_OK)


@pytest.mark.django_db
def test_delivery_invalid_transition_returns_400(api_client: object) -> None:
    """Invalid delivery status transitions should return HTTP 400."""
    delivery = create_delivery(status=Delivery.Status.DELIVERED)
    update = _post(
        api_client,
        f"/api/v1/deliveries/{delivery.id}/update-status/",
        {"status": Delivery.Status.IN_TRANSIT},
    )
    _check(update.status_code == HTTP_BAD_REQUEST)


@pytest.mark.django_db
def test_planning_model_dashboard_endpoints(api_client: object) -> None:
    """Planning trigger/history/model/dashboard endpoints should return expected statuses."""
    create_model_registry()
    create_planning_cycle()

    called: dict[str, list[str]] = {"tasks": []}

    def fake_send_task(name: str, args: list[object] | None = None, kwargs: dict[str, object] | None = None) -> None:
        _ = args
        called["tasks"].append(f"{name}:{kwargs}")

    with patch("fuelsense.core.views.current_app.send_task", side_effect=fake_send_task):
        trig = _post(api_client, "/api/v1/planning/trigger/", {"trigger_type": "MANUAL"})
    _check(trig.status_code == HTTP_ACCEPTED)
    _check(_data_dict(trig)["status"] == "QUEUED")
    _check(called["tasks"])
    _check("fuelsense.core.tasks.run_planning_cycle" in called["tasks"][0])
    _check("cycle_id" in called["tasks"][0])

    hist = _get(api_client, "/api/v1/planning/history/")
    _check(hist.status_code == HTTP_OK)

    model = ModelRegistry.objects.first()
    _check(model is not None)
    if model is None:
        msg = "Expected model registry entry"
        raise AssertionError(msg)
    model_pk = model.pk
    promote = _post(api_client, f"/api/v1/models/{model_pk}/promote/", {})
    _check(promote.status_code == HTTP_OK)

    retrain = _post(api_client, f"/api/v1/models/{model_pk}/retrain/", {})
    _check(retrain.status_code == HTTP_ACCEPTED)

    kpis = _get(api_client, "/api/v1/dashboard/kpis/")
    _check(kpis.status_code == HTTP_OK)

    kpis_cached = _get(api_client, "/api/v1/dashboard/kpis/")
    _check(kpis_cached.status_code == HTTP_OK)

    drift = _get(api_client, "/api/v1/dashboard/drift-heatmap/")
    _check(drift.status_code == HTTP_OK)


@pytest.mark.django_db
def test_planning_trigger_emergency_dispatch(api_client: object) -> None:
    """Emergency trigger should enqueue emergency planning task."""
    called: dict[str, list[str]] = {"tasks": []}

    def fake_send_task(name: str, args: list[object] | None = None, kwargs: dict[str, object] | None = None) -> None:
        _ = args
        called["tasks"].append(f"{name}:{kwargs}")

    with patch("fuelsense.core.views.current_app.send_task", side_effect=fake_send_task):
        trig = _post(
            api_client,
            "/api/v1/planning/trigger/",
            {"trigger_type": "EMERGENCY", "facility_ids": [FACILITY_ID_ONE, FACILITY_ID_TWO]},
        )
    _check(trig.status_code == HTTP_ACCEPTED)
    _check("fuelsense.core.tasks.run_emergency_planning_cycle" in called["tasks"][0])


@pytest.mark.django_db
def test_planning_trigger_emergency_rejects_empty_facility_ids(
    api_client: object,
) -> None:
    """Emergency trigger should reject empty facility id lists."""
    called: dict[str, list[str]] = {"tasks": []}

    def fake_send_task(name: str, args: list[object] | None = None, kwargs: dict[str, object] | None = None) -> None:
        _ = args
        called["tasks"].append(f"{name}:{kwargs}")

    before_count = PlanningCycle.objects.count()

    with patch("fuelsense.core.views.current_app.send_task", side_effect=fake_send_task):
        trig = _post(api_client, "/api/v1/planning/trigger/", {"trigger_type": "EMERGENCY", "facility_ids": []})

    _check(trig.status_code == HTTP_BAD_REQUEST)
    _check("facility_ids" in _data_dict(trig))
    _check(PlanningCycle.objects.count() == before_count)
    _check(called["tasks"] == [])


@pytest.mark.django_db
def test_planning_trigger_emergency_rejects_missing_facility_ids(
    api_client: object,
) -> None:
    """Emergency trigger should reject missing facility id field."""
    called: dict[str, list[str]] = {"tasks": []}

    def fake_send_task(name: str, args: list[object] | None = None, kwargs: dict[str, object] | None = None) -> None:
        _ = args
        called["tasks"].append(f"{name}:{kwargs}")

    before_count = PlanningCycle.objects.count()

    with patch("fuelsense.core.views.current_app.send_task", side_effect=fake_send_task):
        trig = _post(api_client, "/api/v1/planning/trigger/", {"trigger_type": "EMERGENCY"})

    _check(trig.status_code == HTTP_BAD_REQUEST)
    _check("facility_ids" in _data_dict(trig))
    _check(PlanningCycle.objects.count() == before_count)
    _check(called["tasks"] == [])


@pytest.mark.django_db
def test_planning_trigger_emergency_rejects_null_facility_ids(
    api_client: object,
) -> None:
    """Emergency trigger should reject null facility id payloads."""
    called: dict[str, list[str]] = {"tasks": []}

    def fake_send_task(name: str, args: list[object] | None = None, kwargs: dict[str, object] | None = None) -> None:
        _ = args
        called["tasks"].append(f"{name}:{kwargs}")

    before_count = PlanningCycle.objects.count()

    with patch("fuelsense.core.views.current_app.send_task", side_effect=fake_send_task):
        trig = _post(api_client, "/api/v1/planning/trigger/", {"trigger_type": "EMERGENCY", "facility_ids": None})

    _check(trig.status_code == HTTP_BAD_REQUEST)
    _check("facility_ids" in _data_dict(trig))
    _check(PlanningCycle.objects.count() == before_count)
    _check(called["tasks"] == [])


@pytest.mark.django_db
def test_planning_history_without_pagination_branch(
    api_client: object,
) -> None:
    """History endpoint should return list branch when pagination is disabled."""
    create_planning_cycle()

    def _no_pagination(self: object, queryset: object) -> None:
        _ = self, queryset

    with patch("fuelsense.core.views.PlanningViewSet.paginate_queryset", side_effect=_no_pagination):
        res = _get(api_client, "/api/v1/planning/history/")
    _check(res.status_code == HTTP_OK)
    _check(isinstance(res.data, list))


@pytest.mark.django_db
def test_auth_required() -> None:
    """Anonymous requests should be rejected by auth-protected API endpoints."""
    client = Client()
    res = client.get("/api/v1/facilities/")
    _check(res.status_code in {401, 403})


@pytest.mark.django_db
def test_dashboard_kpis_counts_below_reorder_with_fallback_threshold(api_client: object) -> None:
    """KPI endpoint should count facilities below fallback reorder threshold."""
    create_facility(current_inventory=180.0, min_safe_inventory=220.0, dynamic_reorder_point=None)
    create_facility(current_inventory=260.0, min_safe_inventory=220.0, dynamic_reorder_point=None)
    cache.delete("cache:dashboard:kpis")

    response = _get(api_client, "/api/v1/dashboard/kpis/")
    _check(response.status_code == HTTP_OK)
    _check(_data_dict(response)["facilities_below_reorder"] == 1)
