"""Routing tests for request schema, matrix behavior, and cache branches."""

from __future__ import annotations

import datetime as dt

import pytest

from fuelsense.core.routing import _haversine, _time_to_minutes, build_optimizer_request
from fuelsense.core.tests.factories import DepotFacilityAssignmentFactory, DepotFactory, FacilityFactory, VehicleFactory

HAVERSINE_TOLERANCE_KM = 5.0
DEFAULT_FALLBACK_MINUTES = 123
EXPECTED_MATRIX_SIZE = 4
FLOAT_EPSILON = 1e-9
EXPECTED_RECOMPUTE_COUNT_WITHOUT_CACHE = 2
EXPECTED_MATRIX_LARGE_SIZE = 81
EXPECTED_MAPPING_LARGE_SIZE = 80


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


@pytest.mark.django_db
def test_haversine_known_value() -> None:
    """Haversine helper should return close Riyadh-to-Dammam distance."""
    # Approx Riyadh to Dammam great-circle distance.
    km = _haversine(24.7136, 46.6753, 26.4207, 50.0888)
    _check(abs(km - 394.0) < HAVERSINE_TOLERANCE_KM)


@pytest.mark.django_db
def test_time_to_minutes_uses_default_for_none() -> None:
    """Time-to-minutes helper should return provided default when input is None."""
    _check(_time_to_minutes(None, DEFAULT_FALLBACK_MINUTES) == DEFAULT_FALLBACK_MINUTES)


@pytest.mark.django_db
def test_build_optimizer_request_schema_and_matrix_properties() -> None:
    """Routing payload should include required schema keys and symmetric distance matrix."""
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

    required_keys = {
        "depot_lat",
        "depot_lng",
        "vehicles",
        "stops",
        "distance_matrix",
        "max_route_duration",
    }
    _check(required_keys.issubset(payload.keys()))
    matrix = payload["distance_matrix"]
    _check(isinstance(matrix, list))
    _check(len(matrix) == EXPECTED_MATRIX_SIZE)

    for i in range(len(matrix)):
        _check(abs(float(matrix[i][i])) < FLOAT_EPSILON)
        for j in range(len(matrix)):
            _check(abs(float(matrix[i][j]) - float(matrix[j][i])) < FLOAT_EPSILON)

    stops = payload["stops"]
    for stop in stops:
        _check(float(stop["demand"]) >= 0.0)

    _check(facility_index_map == {1: facilities[0].id, 2: facilities[1].id, 3: facilities[2].id})


@pytest.mark.django_db
def test_build_optimizer_request_uses_min_safe_fallback_for_demand() -> None:
    """Demand should fall back to min-safe delta when dynamic reorder point is absent."""
    depot = DepotFactory(latitude=24.7, longitude=46.7)
    VehicleFactory(depot=depot, capacity=1000.0, cost_per_km=2.1, is_available=True)
    facility = FacilityFactory(current_inventory=150.0, min_safe_inventory=220.0, dynamic_reorder_point=None)

    payload, _ = build_optimizer_request(depot, [facility])
    stops = payload["stops"]
    _check(len(stops) == 1)
    _check(float(stops[0]["demand"]) == pytest.approx(70.0))


@pytest.mark.django_db
def test_build_optimizer_request_uses_distance_matrix_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Matrix cache should prevent recomputation on repeated equivalent requests."""
    depot = DepotFactory(latitude=24.7, longitude=46.7)
    VehicleFactory(depot=depot, capacity=1000.0, cost_per_km=2.1, is_available=True)
    facilities = [
        FacilityFactory(
            latitude=24.8,
            longitude=46.8,
            current_inventory=200.0,
            dynamic_reorder_point=350.0,
            delivery_window_start=dt.time(8, 0),
            delivery_window_end=dt.time(14, 0),
        ),
    ]
    DepotFacilityAssignmentFactory(depot=depot, facility=facilities[0])
    monkeypatch.setenv("FUELSENSE_ENABLE_ROUTING_MATRIX_CACHE", "1")

    calls = {"count": 0}
    original = __import__("fuelsense.core.routing", fromlist=["_compute_distance_matrix"])._compute_distance_matrix  # noqa: SLF001

    def _wrapped(coords: list[tuple[float, float]]) -> list[list[float]]:
        calls["count"] += 1
        return original(coords)

    monkeypatch.setattr("fuelsense.core.routing._compute_distance_matrix", _wrapped)

    build_optimizer_request(depot, facilities)
    build_optimizer_request(depot, facilities)
    _check(calls["count"] == 1)


@pytest.mark.django_db
def test_build_optimizer_request_without_cache_recomputes(monkeypatch: pytest.MonkeyPatch) -> None:
    """When cache is disabled, matrix computation should run for each request."""
    depot = DepotFactory(latitude=24.7, longitude=46.7)
    VehicleFactory(depot=depot, capacity=1000.0, cost_per_km=2.1, is_available=True)
    facility = FacilityFactory(
        latitude=24.8,
        longitude=46.8,
        current_inventory=200.0,
        dynamic_reorder_point=350.0,
        delivery_window_start=dt.time(8, 0),
        delivery_window_end=dt.time(14, 0),
    )
    DepotFacilityAssignmentFactory(depot=depot, facility=facility)
    monkeypatch.setenv("FUELSENSE_ENABLE_ROUTING_MATRIX_CACHE", "0")

    calls = {"count": 0}
    original = __import__("fuelsense.core.routing", fromlist=["_compute_distance_matrix"])._compute_distance_matrix  # noqa: SLF001

    def _wrapped(coords: list[tuple[float, float]]) -> list[list[float]]:
        calls["count"] += 1
        return original(coords)

    monkeypatch.setattr("fuelsense.core.routing._compute_distance_matrix", _wrapped)
    build_optimizer_request(depot, [facility])
    build_optimizer_request(depot, [facility])
    _check(calls["count"] == EXPECTED_RECOMPUTE_COUNT_WITHOUT_CACHE)


@pytest.mark.django_db
def test_build_optimizer_request_large_facility_set_stress() -> None:
    """Large facility sets should build matrix and mapping with expected dimensions."""
    depot = DepotFactory(latitude=24.7, longitude=46.7)
    VehicleFactory(depot=depot, capacity=1000.0, cost_per_km=2.1, is_available=True)
    facilities = []
    for idx in range(80):
        facility = FacilityFactory(
            latitude=24.0 + (idx * 0.01),
            longitude=46.0 + (idx * 0.01),
            current_inventory=100.0,
            dynamic_reorder_point=200.0,
            delivery_window_start=dt.time(8, 0),
            delivery_window_end=dt.time(14, 0),
        )
        DepotFacilityAssignmentFactory(depot=depot, facility=facility)
        facilities.append(facility)

    payload, mapping = build_optimizer_request(depot, facilities)
    matrix = payload["distance_matrix"]
    _check(len(matrix) == EXPECTED_MATRIX_LARGE_SIZE)
    _check(len(mapping) == EXPECTED_MAPPING_LARGE_SIZE)
