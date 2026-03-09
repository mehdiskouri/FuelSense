"""Benchmark utility for optimizer CPU and GPU backends."""

from __future__ import annotations

import argparse
import os
from time import perf_counter

import numpy as np

from fuelsense_common.compute import DeviceType
from fuelsense_common.registry import get_backend
import optimizer.backends  # noqa: F401


def _synthetic_problem(stops: int, seed: int = 42) -> tuple[list[dict[str, float]], list[dict[str, float | int]], list[list[float]]]:
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 100.0, size=(stops + 1, 2)).astype(np.float32)
    matrix = np.zeros((stops + 1, stops + 1), dtype=np.float32)
    for i in range(stops + 1):
        for j in range(i + 1, stops + 1):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            matrix[i, j] = d
            matrix[j, i] = d

    vehicles = [{"capacity": 1500.0, "cost_per_km": 2.0} for _ in range(6)]
    stop_list: list[dict[str, float | int]] = []
    for idx in range(1, stops + 1):
        start_min = 0
        end_min = 1440
        stop_list.append(
            {
                "facility_index": idx,
                "demand": float(rng.uniform(30.0, 120.0)),
                "time_window_start": start_min,
                "time_window_end": end_min,
                "service_time": 30,
            }
        )
    return vehicles, stop_list, matrix.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark route optimizer backend")
    parser.add_argument("--stops", type=int, default=50)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    os.environ["FUELSENSE_DEVICE"] = args.device
    device = DeviceType.CUDA if args.device == "cuda" else DeviceType.CPU

    vehicles, stops, distance_matrix = _synthetic_problem(args.stops)
    backend = get_backend("route_optimizer", device)
    backend.warmup()

    start = perf_counter()
    result = backend.solve(
        depot_lat=24.7,
        depot_lng=46.7,
        vehicles=vehicles,
        stops=stops,
        distance_matrix=distance_matrix,
        max_route_duration=1440,
    )
    elapsed_ms = (perf_counter() - start) * 1000

    print("backend,status,stops,vehicles_used,total_cost,baseline_cost,cost_reduction_pct,elapsed_ms")
    print(
        f"{args.device},{result['status']},{args.stops},{result['vehicles_used']},"
        f"{result['total_cost']:.2f},{result['baseline_cost']:.2f},{result['cost_reduction_pct']:.2f},{elapsed_ms:.2f}"
    )


if __name__ == "__main__":
    main()
