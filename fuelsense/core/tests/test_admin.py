from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from django.utils import timezone

from fuelsense.core.tests.factories import (
    DeliveryFactory,
    FacilityFactory,
    ForecastFactory,
    InventoryLogFactory,
    ModelRegistryFactory,
)


@pytest.mark.django_db
def test_admin_list_pages_render(admin_client: Any) -> None:
    urls = [
        "/admin/core/facility/",
        "/admin/core/delivery/",
        "/admin/core/planningcycle/",
        "/admin/core/modelregistry/",
    ]
    for url in urls:
        response = admin_client.get(url)
        assert response.status_code == 200


@pytest.mark.django_db
def test_admin_custom_pages_render(admin_client: Any) -> None:
    facility = FacilityFactory()
    delivery = DeliveryFactory()
    ModelRegistryFactory()

    facility_page = admin_client.get(f"/admin/core/facility/{facility.id}/change/")
    assert facility_page.status_code == 200

    delivery_page = admin_client.get(f"/admin/core/delivery/{delivery.id}/change/")
    assert delivery_page.status_code == 200

    dashboard_page = admin_client.get("/admin/fuelsense/dashboard/")
    assert dashboard_page.status_code == 200


@pytest.mark.django_db
def test_facility_chart_forecast_overlay_uses_future_horizon_timestamps(
    admin_client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facility = FacilityFactory()
    base = timezone.now() - timedelta(days=3)
    InventoryLogFactory(facility=facility, timestamp=base, consumption=10.0, inventory_level=500.0)
    InventoryLogFactory(facility=facility, timestamp=base + timedelta(days=1), consumption=12.0, inventory_level=490.0)
    forecast = ForecastFactory(
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

    monkeypatch.setattr("fuelsense.core.admin.plt.plot", _capture_plot)
    monkeypatch.setattr("fuelsense.core.admin.plt.legend", lambda *args, **kwargs: None)

    response = admin_client.get(f"/admin/core/facility/{facility.id}/change/")
    assert response.status_code == 200

    forecast_plot_calls = [call for call in captured if call[1].get("label") == "Forecast p50"]
    assert len(forecast_plot_calls) == 1
    x_values = list(forecast_plot_calls[0][0][0])
    assert len(x_values) == 3
    assert x_values[0] == created_at + timedelta(days=1)
    assert x_values[1] == created_at + timedelta(days=2)
    assert x_values[2] == created_at + timedelta(days=3)
    assert len(set(x_values)) == 3


@pytest.mark.django_db
def test_facility_chart_handles_malformed_forecast_points_and_shows_reliability_notes(admin_client: Any) -> None:
    facility = FacilityFactory()
    InventoryLogFactory(facility=facility)
    forecast = ForecastFactory(
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
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "Latest forecast model" in content
    assert "fallback-local" in content
    assert "Forecast source is fallback-local" in content
    assert "Forecast may be stale" in content
    assert "Skipped malformed forecast points: 2" in content
