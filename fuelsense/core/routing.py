"""Routing utilities for logistics planning."""

from __future__ import annotations

import math
from datetime import time
from typing import Any

import numpy as np

from fuelsense.core.reorder import get_effective_reorder_point


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
        reorder_point = get_effective_reorder_point(
            getattr(facility, "dynamic_reorder_point", None),
            float(getattr(facility, "min_safe_inventory", 0.0)),
        )
        demand = max(float(reorder_point) - float(facility.current_inventory), 0.0)
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

    coords_arr = np.asarray(coords, dtype=np.float64)
    lat_rad = np.radians(coords_arr[:, 0])
    lon_rad = np.radians(coords_arr[:, 1])
    d_lat = lat_rad[:, None] - lat_rad[None, :]
    d_lon = lon_rad[:, None] - lon_rad[None, :]
    a = np.sin(d_lat / 2.0) ** 2 + np.cos(lat_rad)[:, None] * np.cos(lat_rad)[None, :] * np.sin(d_lon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    matrix = (6371.0 * c).tolist()

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
