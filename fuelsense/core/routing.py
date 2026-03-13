"""Routing utilities for logistics planning."""

from __future__ import annotations

import hashlib
import math
import os
from typing import TYPE_CHECKING, cast

import numpy as np
from django.core.cache import cache

from fuelsense.core.reorder import get_effective_reorder_point

if TYPE_CHECKING:
    from datetime import time

    from fuelsense.core.models import Depot, Facility


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
    hour = int(value.hour)
    minute = int(value.minute)
    return int(hour * 60 + minute)


def _distance_matrix_cache_key(depot: Depot, facilities: list[Facility]) -> str:
    depot_id = int(depot.id)
    depot_lat = float(depot.latitude)
    depot_lng = float(depot.longitude)
    parts: list[str] = [
        f"depot:{depot_id}",
        f"dlat:{depot_lat:.6f}",
        f"dlng:{depot_lng:.6f}",
    ]
    parts.extend(
        "|".join(
            [
                str(int(facility.id)),
                f"{float(facility.latitude):.6f}",
                f"{float(facility.longitude):.6f}",
            ],
        )
        for facility in facilities
    )
    digest = hashlib.sha256(";".join(parts).encode("utf-8")).hexdigest()
    return f"cache:routing:distance_matrix:{digest}"


def _compute_distance_matrix(coords: list[tuple[float, float]]) -> list[list[float]]:
    coords_arr = np.asarray(coords, dtype=np.float64)
    lat_rad = np.radians(coords_arr[:, 0])
    lon_rad = np.radians(coords_arr[:, 1])
    d_lat = lat_rad[:, None] - lat_rad[None, :]
    d_lon = lon_rad[:, None] - lon_rad[None, :]
    a = np.sin(d_lat / 2.0) ** 2 + np.cos(lat_rad)[:, None] * np.cos(lat_rad)[None, :] * np.sin(d_lon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return cast("list[list[float]]", (6371.0 * c).tolist())


def build_optimizer_request(depot: Depot, facilities: list[Facility]) -> tuple[dict[str, object], dict[int, int]]:
    """Build optimizer payload and facility-index mapping for a depot batch."""
    coords: list[tuple[float, float]] = [(float(depot.latitude), float(depot.longitude))]
    stops: list[dict[str, object]] = []
    facility_index_map: dict[int, int] = {}

    for idx, facility in enumerate(facilities, start=1):
        coords.append((float(facility.latitude), float(facility.longitude)))
        reorder_point = get_effective_reorder_point(facility.dynamic_reorder_point, float(facility.min_safe_inventory))
        demand = max(float(reorder_point) - float(facility.current_inventory), 0.0)
        stops.append(
            {
                "facility_index": idx,
                "demand": float(demand),
                "time_window_start": _time_to_minutes(facility.delivery_window_start, 0),
                "time_window_end": _time_to_minutes(facility.delivery_window_end, 24 * 60),
                "service_time": 30,
            },
        )
        facility_index_map[idx] = int(facility.id)

    cache_enabled = os.environ.get("FUELSENSE_ENABLE_ROUTING_MATRIX_CACHE", "0") == "1"
    if cache_enabled:
        cache_key = _distance_matrix_cache_key(depot, facilities)
        cached_matrix = cache.get(cache_key)
        if isinstance(cached_matrix, list):
            matrix = cast("list[list[float]]", cached_matrix)
        else:
            matrix = _compute_distance_matrix(coords)
            cache_ttl = int(os.environ.get("FUELSENSE_ROUTING_MATRIX_CACHE_TTL", "600"))
            cache.set(cache_key, matrix, timeout=max(cache_ttl, 1))
    else:
        matrix = _compute_distance_matrix(coords)

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
