"""Routing utilities for logistics planning."""

from __future__ import annotations

import math
from datetime import time
from typing import Any


def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius_km * c


def _time_to_minutes(value: time | None, default: int) -> int:
    if value is None:
        return default
    return int(value.hour * 60 + value.minute)


def build_optimizer_request(depot: Any, facilities: list[Any]) -> tuple[dict[str, object], dict[int, int]]:
    coords: list[tuple[float, float]] = [(float(depot.latitude), float(depot.longitude))]
    stops: list[dict[str, object]] = []
    facility_index_map: dict[int, int] = {}

    for idx, facility in enumerate(facilities, start=1):
        coords.append((float(facility.latitude), float(facility.longitude)))
        demand = max(float(facility.dynamic_reorder_point or 0.0) - float(facility.current_inventory), 0.0)
        stops.append(
            {
                "facility_index": idx,
                "demand": float(demand),
                "time_window_start": _time_to_minutes(getattr(facility, "delivery_window_start", None), 0),
                "time_window_end": _time_to_minutes(getattr(facility, "delivery_window_end", None), 24 * 60),
                "service_time": 30,
            }
        )
        facility_index_map[idx] = int(facility.id)

    n = len(coords)
    matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            matrix[i][j] = d
            matrix[j][i] = d

    vehicles = [
        {
            "capacity": float(vehicle.capacity),
            "cost_per_km": float(vehicle.cost_per_km),
        }
        for vehicle in depot.vehicles.filter(is_available=True).order_by("id")
    ]

    request = {
        "depot_lat": float(depot.latitude),
        "depot_lng": float(depot.longitude),
        "vehicles": vehicles,
        "stops": stops,
        "distance_matrix": matrix,
        "max_route_duration": 480,
    }
    return request, facility_index_map
