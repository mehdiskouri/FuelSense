"""Admin UI tests for list/change pages and facility chart rendering behavior."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest
from django.utils import timezone

from fuelsense.core.tests.factories import (
    create_delivery,
    create_facility,
    create_forecast,
    create_inventory_log,
    create_model_registry,
)

if TYPE_CHECKING:
    from django.test import Client

HTTP_OK = 200
EXPECTED_PLOT_CALLS = 1
EXPECTED_HORIZON_POINTS = 3


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


@pytest.mark.django_db
def test_admin_list_pages_render(admin_client: Client) -> None:
    """Core admin list pages should render successfully for staff users."""
    urls = [
        "/admin/core/facility/",
        "/admin/core/delivery/",
        "/admin/core/planningcycle/",
        "/admin/core/modelregistry/",
    ]
    for url in urls:
        response = admin_client.get(url)
        _check(response.status_code == HTTP_OK)


@pytest.mark.django_db
def test_admin_custom_pages_render(admin_client: Client) -> None:
    """Custom facility/delivery change pages and dashboard should render."""
    facility = create_facility()
    delivery = create_delivery()
    create_model_registry()

    facility_page = admin_client.get(f"/admin/core/facility/{facility.id}/change/")
    _check(facility_page.status_code == HTTP_OK)

    delivery_page = admin_client.get(f"/admin/core/delivery/{delivery.id}/change/")
    _check(delivery_page.status_code == HTTP_OK)

    dashboard_page = admin_client.get("/admin/fuelsense/dashboard/")
    _check(dashboard_page.status_code == HTTP_OK)


@pytest.mark.django_db
def test_facility_chart_forecast_overlay_uses_future_horizon_timestamps(
    admin_client: Client,
) -> None:
    """Forecast overlay plot should use forecast-day offsets from forecast creation time."""
    facility = create_facility()
    base = timezone.now() - timedelta(days=3)
    create_inventory_log(facility=facility, timestamp=base, consumption=10.0, inventory_level=500.0)
    create_inventory_log(facility=facility, timestamp=base + timedelta(days=1), consumption=12.0, inventory_level=490.0)
    forecast = create_forecast(
        facility=facility,
        predictions_json=[
            {"day": 1, "p50": 100.0},
            {"day": 2, "p50": 110.0},
            {"day": 3, "p50": 120.0},
        ],
    )
    created_at = timezone.now() - timedelta(hours=1)
    forecast.created_at = created_at
    forecast.save(update_fields=["created_at"])

    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _capture_plot(*args: object, **kwargs: object) -> list[object]:
        captured.append((args, kwargs))
        return []

    def _noop_legend(*_args: object, **_kwargs: object) -> None:
        return None

    with (
        patch("fuelsense.core.admin.plt.plot", side_effect=_capture_plot),
        patch("fuelsense.core.admin.plt.legend", side_effect=_noop_legend),
    ):
        response = admin_client.get(f"/admin/core/facility/{facility.id}/change/")
    _check(response.status_code == HTTP_OK)

    forecast_plot_calls = [call for call in captured if call[1].get("label") == "Forecast p50"]
    _check(len(forecast_plot_calls) == EXPECTED_PLOT_CALLS)
    x_values = cast("list[object]", forecast_plot_calls[0][0][0])
    _check(len(x_values) == EXPECTED_HORIZON_POINTS)
    _check(x_values[0] == created_at + timedelta(days=1))
    _check(x_values[1] == created_at + timedelta(days=2))
    _check(x_values[2] == created_at + timedelta(days=3))
    _check(len(set(x_values)) == EXPECTED_HORIZON_POINTS)


@pytest.mark.django_db
def test_facility_chart_handles_malformed_forecast_points_and_shows_reliability_notes(admin_client: Client) -> None:
    """Facility admin view should show reliability warnings for malformed forecast points."""
    facility = create_facility()
    create_inventory_log(facility=facility)
    forecast = create_forecast(
        facility=facility,
        model_version="fallback-local",
        horizon_days=4,
        predictions_json=[
            {"day": 1, "p50": 100.0},
            {"day": "bad", "p50": 110.0},
            {"day": 3, "p50": "not-a-number"},
            "invalid-row",
        ],
    )
    old_created_at = timezone.now() - timedelta(days=10)
    forecast.created_at = old_created_at
    forecast.save(update_fields=["created_at"])

    response = admin_client.get(f"/admin/core/facility/{facility.id}/change/")
    _check(response.status_code == HTTP_OK)
    content = response.content.decode("utf-8")
    _check("Latest forecast model" in content)
    _check("fallback-local" in content)
    _check("Forecast source is fallback-local" in content)
    _check("Forecast may be stale" in content)
    _check("Skipped malformed forecast points: 2" in content)
