"""CPU OR-Tools backend for route optimization."""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false

from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter
from typing import override

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from fuelsense_common.compute import ComputeBackend, DeviceType
from fuelsense_common.registry import register_backend

SMALL_INSTANCE_NODE_THRESHOLD = 20
MEDIUM_INSTANCE_NODE_THRESHOLD = 50
PATH_CHEAPEST_NODE_THRESHOLD = 30
SMALL_INSTANCE_TIME_LIMIT_MS = 2000
MEDIUM_INSTANCE_TIME_LIMIT_MS = 6000
CAPACITY_DIM_START_CUMUL_ZERO = True
TIME_DIM_START_CUMUL_ZERO = True


@dataclass(frozen=True)
class _TimeDimensionInput:
    travel_minutes: list[list[int]]
    service_times: dict[int, int]
    time_windows: dict[int, tuple[int, int]]
    max_route_duration: int


@dataclass(frozen=True)
class _RouteExtractionInput:
    n_vehicles: int
    distance_matrix: list[list[float]]
    demands: dict[int, float]
    vehicle_costs: list[float]


@dataclass(frozen=True)
class _SolutionSummary:
    routes: list[dict[str, object]]
    total_distance: float
    total_cost: float
    solver_time_ms: float
    baseline_cost: float


@register_backend("route_optimizer", DeviceType.CPU)
class ORToolsOptimizer(ComputeBackend):
    """CPU backend using Google OR-Tools for vehicle routing."""

    device = DeviceType.CPU

    def __init__(self) -> None:
        """Load solver time-limit options from environment."""
        self.time_limit_ms = int(os.environ.get("FUELSENSE_OPTIMIZER_TIME_LIMIT_MS", "10000"))
        self.adaptive_time_limit = os.environ.get("FUELSENSE_OPTIMIZER_ADAPTIVE_TIME_LIMIT", "1") == "1"

    def _effective_time_limit_ms(self, n_nodes: int) -> int:
        if not self.adaptive_time_limit:
            return self.time_limit_ms
        if n_nodes <= SMALL_INSTANCE_NODE_THRESHOLD:
            return min(self.time_limit_ms, SMALL_INSTANCE_TIME_LIMIT_MS)
        if n_nodes <= MEDIUM_INSTANCE_NODE_THRESHOLD:
            return min(self.time_limit_ms, MEDIUM_INSTANCE_TIME_LIMIT_MS)
        return self.time_limit_ms

    def warmup(self) -> None:
        """No-op warmup for CPU backend."""
        return

    def health_check(self) -> dict[str, object]:
        """Return backend health metadata for service probes."""
        return {
            "device": self.device.value,
            "solver": "ortools",
            "time_limit_ms": self.time_limit_ms,
        }

    @staticmethod
    def _build_infeasible_response(
        start: float,
        distance_matrix: list[list[float]],
        vehicles: list[dict[str, float]],
    ) -> dict[str, object]:
        solver_time_ms = (perf_counter() - start) * 1000
        baseline_cost = ORToolsOptimizer._compute_baseline(distance_matrix, vehicles)
        return {
            "status": "infeasible",
            "routes": [],
            "total_distance_km": 0.0,
            "total_cost": 0.0,
            "vehicles_used": 0,
            "solver_time_ms": solver_time_ms,
            "baseline_cost": baseline_cost,
            "cost_reduction_pct": 0.0,
        }

    @staticmethod
    def _extract_stop_metadata(
        stops: list[dict[str, float | int]],
        max_route_duration: int,
    ) -> tuple[dict[int, float], dict[int, int], dict[int, tuple[int, int]]]:
        stop_by_idx = {int(s.get("facility_index", -1)): s for s in stops}
        demands = {idx: float(s.get("demand", 0.0)) for idx, s in stop_by_idx.items()}
        service_times = {idx: int(s.get("service_time", 30)) for idx, s in stop_by_idx.items()}
        time_windows = {
            idx: (
                int(s.get("time_window_start", 0)),
                int(s.get("time_window_end", max_route_duration)),
            )
            for idx, s in stop_by_idx.items()
        }
        return demands, service_times, time_windows

    @staticmethod
    def _prepare_vehicle_constraints(
        vehicles: list[dict[str, float]],
        distance_matrix: list[list[float]],
        demand_scale: int,
    ) -> tuple[list[float], list[int], list[list[int]], list[list[int]], list[int]]:
        n_nodes = len(distance_matrix)
        vehicle_costs = [float(v.get("cost_per_km", 1.0)) for v in vehicles]
        capacities = [max(round(float(v.get("capacity", 0.0)) * demand_scale), 0) for v in vehicles]
        dist_cost_units = [
            [round(float(distance_matrix[i][j]) * 100.0) for j in range(n_nodes)] for i in range(n_nodes)
        ]
        travel_minutes = [[round(float(distance_matrix[i][j])) for j in range(n_nodes)] for i in range(n_nodes)]
        vehicle_cost_scale = [round(cost * 100.0) for cost in vehicle_costs]
        return vehicle_costs, capacities, dist_cost_units, travel_minutes, vehicle_cost_scale

    @staticmethod
    def _register_cost_callbacks(
        manager: pywrapcp.RoutingIndexManager,
        routing: pywrapcp.RoutingModel,
        dist_cost_units: list[list[int]],
        vehicle_cost_scale: list[int],
    ) -> None:
        for vehicle_idx in range(len(vehicle_cost_scale)):

            def cost_callback(from_index: int, to_index: int, vehicle_pos: int = vehicle_idx) -> int:
                try:
                    from_node = manager.IndexToNode(int(from_index))
                    to_node = manager.IndexToNode(int(to_index))
                except OverflowError:  # pragma: no cover
                    return 0
                scaled_cost = vehicle_cost_scale[vehicle_pos]
                return round((dist_cost_units[from_node][to_node] * scaled_cost) / 100.0)

            callback_idx = routing.RegisterTransitCallback(cost_callback)
            routing.SetArcCostEvaluatorOfVehicle(callback_idx, vehicle_idx)

    @staticmethod
    def _add_capacity_dimension(
        manager: pywrapcp.RoutingIndexManager,
        routing: pywrapcp.RoutingModel,
        demands: dict[int, float],
        capacities: list[int],
        demand_scale: int,
    ) -> None:
        def demand_callback(from_index: int) -> int:
            try:
                from_node = manager.IndexToNode(int(from_index))
            except OverflowError:  # pragma: no cover
                return 0
            return round(demands.get(from_node, 0.0) * demand_scale)

        demand_callback_idx = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_idx,
            0,
            capacities,
            CAPACITY_DIM_START_CUMUL_ZERO,
            "Capacity",
        )

    @staticmethod
    def _add_time_dimension(
        manager: pywrapcp.RoutingIndexManager,
        routing: pywrapcp.RoutingModel,
        time_input: _TimeDimensionInput,
    ) -> pywrapcp.RoutingDimension:
        def time_callback(from_index: int, to_index: int) -> int:
            try:
                from_node = manager.IndexToNode(int(from_index))
                to_node = manager.IndexToNode(int(to_index))
            except OverflowError:  # pragma: no cover
                return 0
            travel_time = time_input.travel_minutes[from_node][to_node]
            service_minutes = int(time_input.service_times.get(to_node, 0)) if to_node != 0 else 0
            return travel_time + service_minutes

        time_callback_idx = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(
            time_callback_idx,
            30,
            int(time_input.max_route_duration),
            TIME_DIM_START_CUMUL_ZERO,
            "Time",
        )
        time_dim = routing.GetDimensionOrDie("Time")
        for node in range(1, manager.GetNumberOfNodes()):
            start_min, end_min = time_input.time_windows.get(node, (0, int(time_input.max_route_duration)))
            idx = manager.NodeToIndex(node)
            time_dim.CumulVar(idx).SetRange(max(start_min, 0), max(end_min, 0))
        return time_dim

    def _solve_routing(
        self,
        manager: pywrapcp.RoutingIndexManager,
        routing: pywrapcp.RoutingModel,
    ) -> pywrapcp.Assignment | None:
        params = pywrapcp.DefaultRoutingSearchParameters()
        if manager.GetNumberOfNodes() <= PATH_CHEAPEST_NODE_THRESHOLD:
            params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        else:
            params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.FromMilliseconds(self._effective_time_limit_ms(manager.GetNumberOfNodes()))
        return routing.SolveWithParameters(params)

    @staticmethod
    def _extract_routes(
        manager: pywrapcp.RoutingIndexManager,
        routing: pywrapcp.RoutingModel,
        solution: pywrapcp.Assignment,
        time_dim: pywrapcp.RoutingDimension,
        route_input: _RouteExtractionInput,
    ) -> tuple[list[dict[str, object]], float, float]:
        routes: list[dict[str, object]] = []
        total_distance = 0.0
        total_cost = 0.0

        for vehicle_idx in range(route_input.n_vehicles):
            index = routing.Start(vehicle_idx)
            route_stops: list[dict[str, object]] = []
            route_distance = 0.0
            seq = 1
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                next_index = solution.Value(routing.NextVar(index))
                next_node = manager.IndexToNode(next_index)
                route_distance += float(route_input.distance_matrix[node][next_node])

                if node != 0:
                    arrival_min = int(solution.Value(time_dim.CumulVar(index)))
                    route_stops.append(
                        {
                            "facility_index": int(node),
                            "demand": float(route_input.demands.get(node, 0.0)),
                            "arrival_min": arrival_min,
                            "sequence": seq,
                        },
                    )
                    seq += 1
                index = next_index

            if route_stops:
                route_cost = route_distance * float(route_input.vehicle_costs[vehicle_idx])
                routes.append(
                    {
                        "vehicle_index": vehicle_idx,
                        "stops": route_stops,
                        "distance_km": float(route_distance),
                        "cost": float(route_cost),
                    },
                )
                total_distance += route_distance
                total_cost += route_cost

        return routes, total_distance, total_cost

    @staticmethod
    def _build_optimal_response(summary: _SolutionSummary, n_nodes: int) -> dict[str, object]:
        vehicles_used = len(summary.routes)
        cost_reduction_pct = 0.0
        if summary.baseline_cost > 0.0:
            cost_reduction_pct = max(
                (summary.baseline_cost - summary.total_cost) / summary.baseline_cost * 100.0,
                0.0,
            )
        return {
            "status": "optimal" if vehicles_used > 0 or n_nodes == 1 else "infeasible",
            "routes": summary.routes,
            "total_distance_km": float(summary.total_distance),
            "total_cost": float(summary.total_cost),
            "vehicles_used": int(vehicles_used),
            "solver_time_ms": float(summary.solver_time_ms),
            "baseline_cost": float(summary.baseline_cost),
            "cost_reduction_pct": float(cost_reduction_pct),
        }

    @override
    def solve(
        self,
        depot_lat: float,
        depot_lng: float,
        vehicles: list[dict[str, float]],
        stops: list[dict[str, float | int]],
        distance_matrix: list[list[float]],
        max_route_duration: int,
    ) -> dict[str, object]:
        """Solve routing with capacities and time windows using OR-Tools."""
        _ = depot_lat, depot_lng
        start = perf_counter()

        n_nodes = len(distance_matrix)
        n_vehicles = len(vehicles)
        if n_nodes <= 1 or n_vehicles == 0:
            return self._build_infeasible_response(start, distance_matrix, vehicles)

        demand_scale = 100
        demands, service_times, time_windows = self._extract_stop_metadata(stops, max_route_duration)

        manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        vehicle_costs, capacities, dist_cost_units, travel_minutes, vehicle_cost_scale = (
            self._prepare_vehicle_constraints(vehicles, distance_matrix, demand_scale)
        )
        self._register_cost_callbacks(manager, routing, dist_cost_units, vehicle_cost_scale)
        self._add_capacity_dimension(manager, routing, demands, capacities, demand_scale)
        time_dim = self._add_time_dimension(
            manager,
            routing,
            _TimeDimensionInput(
                travel_minutes=travel_minutes,
                service_times=service_times,
                time_windows=time_windows,
                max_route_duration=max_route_duration,
            ),
        )

        solution = self._solve_routing(manager, routing)
        solver_time_ms = (perf_counter() - start) * 1000
        baseline_cost = self._compute_baseline(distance_matrix, vehicles)
        if solution is None:
            return self._build_infeasible_response(start, distance_matrix, vehicles)

        routes, total_distance, total_cost = self._extract_routes(
            manager,
            routing,
            solution,
            time_dim,
            _RouteExtractionInput(
                n_vehicles=n_vehicles,
                distance_matrix=distance_matrix,
                demands=demands,
                vehicle_costs=vehicle_costs,
            ),
        )
        return self._build_optimal_response(
            _SolutionSummary(
                routes=routes,
                total_distance=total_distance,
                total_cost=total_cost,
                solver_time_ms=solver_time_ms,
                baseline_cost=baseline_cost,
            ),
            n_nodes,
        )

    @staticmethod
    def _compute_baseline(distance_matrix: list[list[float]], vehicles: list[dict[str, float]]) -> float:
        if len(distance_matrix) <= 1 or not vehicles:
            return 0.0
        min_cost_per_km = min(float(v.get("cost_per_km", 1.0)) for v in vehicles)
        return float(
            sum((2.0 * float(distance_matrix[0][node]) * min_cost_per_km) for node in range(1, len(distance_matrix))),
        )
