from __future__ import annotations

from typing import Any

import pytest

from fuelsense.core.tests.factories import DeliveryFactory, FacilityFactory, ModelRegistryFactory


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
