from __future__ import annotations

import datetime as dt

import pytest

from fuelsense.core.routing import _haversine, build_optimizer_request
from fuelsense.core.tests.factories import DepotFactory, DepotFacilityAssignmentFactory, FacilityFactory, VehicleFactory


@pytest.mark.django_db
def test_haversine_known_value() -> None:
    # Approx Riyadh to Dammam great-circle distance.
    km = _haversine(24.7136, 46.6753, 26.4207, 50.0888)
    assert abs(km - 394.0) < 5.0


@pytest.mark.django_db
def test_build_optimizer_request_schema_and_matrix_properties() -> None:
    depot = DepotFactory(latitude=24.7, longitude=46.7)
    VehicleFactory(depot=depot, capacity=1000.0, cost_per_km=2.1, is_available=True)

    facilities = []
    for idx in range(3):
        facility = FacilityFactory(
            latitude=24.7 + idx * 0.05,
            longitude=46.7 + idx * 0.05,
            current_inventory=200.0,
            dynamic_reorder_point=350.0,
            delivery_window_start=dt.time(8, 0),
            delivery_window_end=dt.time(14, 0),
        )
        DepotFacilityAssignmentFactory(depot=depot, facility=facility)
        facilities.append(facility)

    payload, facility_index_map = build_optimizer_request(depot, facilities)

    assert {"depot_lat", "depot_lng", "vehicles", "stops", "distance_matrix", "max_route_duration"}.issubset(payload.keys())
    matrix = payload["distance_matrix"]
    assert isinstance(matrix, list)
    assert len(matrix) == 4

    for i in range(len(matrix)):
        assert abs(float(matrix[i][i])) < 1e-9
        for j in range(len(matrix)):
            assert abs(float(matrix[i][j]) - float(matrix[j][i])) < 1e-9

    stops = payload["stops"]
    for stop in stops:
        assert float(stop["demand"]) >= 0.0

    assert facility_index_map == {1: facilities[0].id, 2: facilities[1].id, 3: facilities[2].id}
