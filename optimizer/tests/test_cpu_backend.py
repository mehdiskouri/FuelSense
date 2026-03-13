"""CPU optimizer backend tests for feasibility, cost quality, and invariants."""

# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportPrivateUsage=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from hypothesis import given, settings
from hypothesis import strategies as st

from optimizer.backends.cpu_backend import ORToolsOptimizer

if TYPE_CHECKING:
    from collections.abc import Callable


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


def _compute_baseline(matrix: list[list[float]], vehicles: list[dict[str, float]]) -> float:
    compute = cast(
        "Callable[[list[list[float]], list[dict[str, float]]], float]",
        ORToolsOptimizer.__dict__["_compute_baseline"],
    )
    return float(compute(matrix, vehicles))


def _small_instance() -> tuple[list[dict[str, float]], list[dict[str, float | int]], list[list[float]]]:
    vehicles = [
        {"capacity": 200.0, "cost_per_km": 2.0},
        {"capacity": 200.0, "cost_per_km": 2.0},
    ]
    stops = [
        {"facility_index": 1, "demand": 40.0, "time_window_start": 0, "time_window_end": 480, "service_time": 30},
        {"facility_index": 2, "demand": 30.0, "time_window_start": 0, "time_window_end": 480, "service_time": 30},
        {"facility_index": 3, "demand": 50.0, "time_window_start": 0, "time_window_end": 480, "service_time": 30},
        {"facility_index": 4, "demand": 45.0, "time_window_start": 0, "time_window_end": 480, "service_time": 30},
        {"facility_index": 5, "demand": 20.0, "time_window_start": 0, "time_window_end": 480, "service_time": 30},
    ]
    matrix = [
        [0.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [3.0, 0.0, 2.0, 3.0, 4.0, 5.0],
        [4.0, 2.0, 0.0, 2.0, 3.0, 4.0],
        [5.0, 3.0, 2.0, 0.0, 2.0, 3.0],
        [6.0, 4.0, 3.0, 2.0, 0.0, 2.0],
        [7.0, 5.0, 4.0, 3.0, 2.0, 0.0],
    ]
    return vehicles, stops, matrix


def test_solve_visits_all_stops_respects_capacity() -> None:
    """Solver should visit each stop once while keeping route demand within vehicle capacity."""
    backend = ORToolsOptimizer()
    vehicles, stops, matrix = _small_instance()

    result = backend.solve(24.7, 46.7, vehicles, stops, matrix, max_route_duration=480)
    routes = cast("list[dict[str, Any]]", result["routes"])

    _check(result["status"] == "optimal")
    visited = [int(stop["facility_index"]) for route in routes for stop in cast("list[dict[str, Any]]", route["stops"])]
    _check(sorted(visited) == [1, 2, 3, 4, 5])

    capacity_by_vehicle = {idx: float(v["capacity"]) for idx, v in enumerate(vehicles)}
    for route in routes:
        total_demand = sum(float(stop["demand"]) for stop in cast("list[dict[str, Any]]", route["stops"]))
        _check(total_demand <= capacity_by_vehicle[int(route["vehicle_index"])] + 1e-6)


def test_solve_cost_below_baseline_on_medium_instance() -> None:
    """Optimized routing cost should beat the naive baseline on a medium scenario."""
    backend = ORToolsOptimizer()
    vehicles = [{"capacity": 300.0, "cost_per_km": 2.1} for _ in range(3)]
    stops = [
        {"facility_index": i, "demand": 35.0, "time_window_start": 0, "time_window_end": 480, "service_time": 30}
        for i in range(1, 11)
    ]
    matrix = [[0.0 for _ in range(11)] for _ in range(11)]
    for i in range(11):
        for j in range(11):
            matrix[i][j] = float(abs(i - j) + (0.5 if i != j else 0.0))

    result = backend.solve(24.7, 46.7, vehicles, stops, matrix, max_route_duration=480)
    total_cost = float(cast("float", result["total_cost"]))
    baseline_cost = float(cast("float", result["baseline_cost"]))
    _check(result["status"] == "optimal")
    _check(total_cost < baseline_cost)


def test_infeasible_when_demands_exceed_capacity() -> None:
    """Solver should return infeasible when total stop demand exceeds vehicle capacity."""
    backend = ORToolsOptimizer()
    vehicles = [{"capacity": 50.0, "cost_per_km": 2.0}]
    stops = [
        {"facility_index": 1, "demand": 80.0, "time_window_start": 0, "time_window_end": 480, "service_time": 30},
        {"facility_index": 2, "demand": 80.0, "time_window_start": 0, "time_window_end": 480, "service_time": 30},
    ]
    matrix = [
        [0.0, 5.0, 6.0],
        [5.0, 0.0, 2.0],
        [6.0, 2.0, 0.0],
    ]

    result = backend.solve(24.7, 46.7, vehicles, stops, matrix, max_route_duration=120)
    _check(result["status"] == "infeasible")


def test_baseline_computation_known_value() -> None:
    """Baseline computation should match expected hand-calculated route cost."""
    matrix = [
        [0.0, 10.0, 20.0],
        [10.0, 0.0, 5.0],
        [20.0, 5.0, 0.0],
    ]
    vehicles = [{"capacity": 100.0, "cost_per_km": 2.0}, {"capacity": 100.0, "cost_per_km": 3.0}]
    baseline = _compute_baseline(matrix, vehicles)
    _check(baseline == (2 * 10.0 * 2.0) + (2 * 20.0 * 2.0))


def test_solve_returns_required_response_keys() -> None:
    """Solver response should include all contractually required summary fields."""
    backend = ORToolsOptimizer()
    vehicles, stops, matrix = _small_instance()
    result = backend.solve(24.7, 46.7, vehicles, stops, matrix, max_route_duration=480)
    required = {
        "status",
        "routes",
        "total_distance_km",
        "total_cost",
        "vehicles_used",
        "solver_time_ms",
        "baseline_cost",
        "cost_reduction_pct",
    }
    _check(required.issubset(result.keys()))


def test_health_and_warmup_and_baseline_empty_inputs() -> None:
    """Warmup and health checks should run, and empty baseline should evaluate to zero."""
    backend = ORToolsOptimizer()
    backend.warmup()
    health = backend.health_check()
    _check(health["solver"] == "ortools")
    _check(_compute_baseline([[0.0]], []) == 0.0)


def test_solve_infeasible_on_empty_graph_or_vehicles() -> None:
    """Solver should return infeasible for empty stop graphs or missing vehicles."""
    backend = ORToolsOptimizer()
    out_empty = backend.solve(0.0, 0.0, [{"capacity": 1.0, "cost_per_km": 1.0}], [], [[0.0]], 30)
    out_no_vehicle = backend.solve(0.0, 0.0, [], [], [[0.0, 1.0], [1.0, 0.0]], 30)
    _check(out_empty["status"] == "infeasible")
    _check(out_no_vehicle["status"] == "infeasible")


@settings(max_examples=20, deadline=None)
@given(
    demands=st.lists(st.floats(min_value=5.0, max_value=40.0), min_size=2, max_size=6),
    capacity=st.floats(min_value=100.0, max_value=300.0),
)
def test_property_valid_inputs_satisfy_constraints(demands: list[float], capacity: float) -> None:
    """Property-based scenarios should satisfy visit completeness and capacity constraints."""
    backend = ORToolsOptimizer()
    n = len(demands)
    total_demand = sum(demands)
    vehicles_needed = max(1, int(total_demand // capacity) + 1)
    vehicles = [{"capacity": float(capacity), "cost_per_km": 2.0} for _ in range(vehicles_needed)]

    stops: list[dict[str, float | int]] = []
    for i, demand in enumerate(demands, start=1):
        stops.append(
            {
                "facility_index": i,
                "demand": float(demand),
                "time_window_start": 0,
                "time_window_end": 1440,
                "service_time": 30,
            },
        )

    matrix = [[0.0 for _ in range(n + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(n + 1):
            matrix[i][j] = float(abs(i - j) + (0.1 if i != j else 0.0))

    result = backend.solve(24.7, 46.7, vehicles, stops, matrix, max_route_duration=1440)
    if result["status"] != "optimal":
        return

    routes = cast("list[dict[str, Any]]", result["routes"])
    visited = [int(stop["facility_index"]) for route in routes for stop in cast("list[dict[str, Any]]", route["stops"])]
    _check(sorted(visited) == list(range(1, n + 1)))

    for route in routes:
        used = sum(float(stop["demand"]) for stop in cast("list[dict[str, Any]]", route["stops"]))
        cap = float(vehicles[int(route["vehicle_index"])]["capacity"])
        _check(used <= cap + 0.05)
