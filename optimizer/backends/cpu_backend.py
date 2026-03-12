"""CPU OR-Tools backend for route optimization."""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false

from __future__ import annotations

import os
from time import perf_counter

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from fuelsense_common.compute import ComputeBackend, DeviceType
from fuelsense_common.registry import register_backend


@register_backend("route_optimizer", DeviceType.CPU)
class ORToolsOptimizer(ComputeBackend):
    device = DeviceType.CPU

    def __init__(self) -> None:
        self.time_limit_ms = int(os.environ.get("FUELSENSE_OPTIMIZER_TIME_LIMIT_MS", "10000"))
        self.adaptive_time_limit = os.environ.get("FUELSENSE_OPTIMIZER_ADAPTIVE_TIME_LIMIT", "1") == "1"

    def _effective_time_limit_ms(self, n_nodes: int) -> int:
        if not self.adaptive_time_limit:
            return self.time_limit_ms
        if n_nodes <= 20:
            return min(self.time_limit_ms, 2000)
        if n_nodes <= 50:
            return min(self.time_limit_ms, 6000)
        return self.time_limit_ms

    def warmup(self) -> None:
        return None

    def health_check(self) -> dict[str, object]:
        return {
            "device": self.device.value,
            "solver": "ortools",
            "time_limit_ms": self.time_limit_ms,
        }

    def solve(
        self,
        depot_lat: float,
        depot_lng: float,
        vehicles: list[dict[str, float]],
        stops: list[dict[str, float | int]],
        distance_matrix: list[list[float]],
        max_route_duration: int,
    ) -> dict[str, object]:
        _ = depot_lat, depot_lng
        start = perf_counter()

        n_nodes = len(distance_matrix)
        n_vehicles = len(vehicles)
        if n_nodes <= 1 or n_vehicles == 0:
            return {
                "status": "infeasible",
                "routes": [],
                "total_distance_km": 0.0,
                "total_cost": 0.0,
                "vehicles_used": 0,
                "solver_time_ms": (perf_counter() - start) * 1000,
                "baseline_cost": self._compute_baseline(distance_matrix, vehicles),
                "cost_reduction_pct": 0.0,
            }

        stop_by_idx = {int(s.get("facility_index", -1)): s for s in stops}
        demands = {idx: float(s.get("demand", 0.0)) for idx, s in stop_by_idx.items()}
        demand_scale = 100
        service_times = {idx: int(s.get("service_time", 30)) for idx, s in stop_by_idx.items()}
        time_windows = {
            idx: (
                int(s.get("time_window_start", 0)),
                int(s.get("time_window_end", max_route_duration)),
            )
            for idx, s in stop_by_idx.items()
        }

        manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        vehicle_costs = [float(v.get("cost_per_km", 1.0)) for v in vehicles]
        capacities = [max(int(round(float(v.get("capacity", 0.0)) * demand_scale)), 0) for v in vehicles]

        dist_cost_units = [
            [int(round(float(distance_matrix[i][j]) * 100.0)) for j in range(n_nodes)] for i in range(n_nodes)
        ]
        travel_minutes = [[int(round(float(distance_matrix[i][j]))) for j in range(n_nodes)] for i in range(n_nodes)]
        vehicle_cost_scale = [int(round(cost * 100.0)) for cost in vehicle_costs]

        for vehicle_idx in range(len(vehicle_costs)):

            def cost_callback(from_index: int, to_index: int, vehicle_pos: int = vehicle_idx) -> int:
                try:
                    from_node = manager.IndexToNode(int(from_index))
                    to_node = manager.IndexToNode(int(to_index))
                except OverflowError:  # pragma: no cover
                    return 0
                scaled_cost = vehicle_cost_scale[vehicle_pos]
                return int(round((dist_cost_units[from_node][to_node] * scaled_cost) / 100.0))

            callback_idx = routing.RegisterTransitCallback(cost_callback)
            routing.SetArcCostEvaluatorOfVehicle(callback_idx, vehicle_idx)

        def demand_callback(from_index: int) -> int:
            try:
                from_node = manager.IndexToNode(int(from_index))
            except OverflowError:  # pragma: no cover
                return 0
            return int(round(demands.get(from_node, 0.0) * demand_scale))

        demand_callback_idx = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_idx,
            0,
            capacities,
            True,
            "Capacity",
        )

        def time_callback(from_index: int, to_index: int) -> int:
            try:
                from_node = manager.IndexToNode(int(from_index))
                to_node = manager.IndexToNode(int(to_index))
            except OverflowError:  # pragma: no cover
                return 0
            travel_time = travel_minutes[from_node][to_node]
            service_minutes = int(service_times.get(to_node, 0)) if to_node != 0 else 0
            return travel_time + service_minutes

        time_callback_idx = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(
            time_callback_idx,
            30,
            int(max_route_duration),
            True,
            "Time",
        )
        time_dim = routing.GetDimensionOrDie("Time")
        for node in range(1, n_nodes):
            start_min, end_min = time_windows.get(node, (0, int(max_route_duration)))
            idx = manager.NodeToIndex(node)
            time_dim.CumulVar(idx).SetRange(max(start_min, 0), max(end_min, 0))

        params = pywrapcp.DefaultRoutingSearchParameters()
        if n_nodes <= 30:
            params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        else:
            params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.FromMilliseconds(self._effective_time_limit_ms(n_nodes))

        solution = routing.SolveWithParameters(params)
        solver_time_ms = (perf_counter() - start) * 1000
        baseline_cost = self._compute_baseline(distance_matrix, vehicles)
        if solution is None:
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

        routes: list[dict[str, object]] = []
        total_distance = 0.0
        total_cost = 0.0
        for vehicle_idx in range(n_vehicles):
            index = routing.Start(vehicle_idx)
            route_stops: list[dict[str, object]] = []
            route_distance = 0.0
            seq = 1
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                next_index = solution.Value(routing.NextVar(index))
                next_node = manager.IndexToNode(next_index)
                route_distance += float(distance_matrix[node][next_node])

                if node != 0:
                    arrival_min = int(solution.Value(time_dim.CumulVar(index)))
                    route_stops.append(
                        {
                            "facility_index": int(node),
                            "demand": float(demands.get(node, 0.0)),
                            "arrival_min": arrival_min,
                            "sequence": seq,
                        }
                    )
                    seq += 1

                index = next_index

            if route_stops:
                route_cost = route_distance * float(vehicle_costs[vehicle_idx])
                routes.append(
                    {
                        "vehicle_index": vehicle_idx,
                        "stops": route_stops,
                        "distance_km": float(route_distance),
                        "cost": float(route_cost),
                    }
                )
                total_distance += route_distance
                total_cost += route_cost

        vehicles_used = len(routes)
        cost_reduction_pct = 0.0
        if baseline_cost > 0.0:
            cost_reduction_pct = max((baseline_cost - total_cost) / baseline_cost * 100.0, 0.0)

        return {
            "status": "optimal" if vehicles_used > 0 or n_nodes == 1 else "infeasible",
            "routes": routes,
            "total_distance_km": float(total_distance),
            "total_cost": float(total_cost),
            "vehicles_used": int(vehicles_used),
            "solver_time_ms": float(solver_time_ms),
            "baseline_cost": float(baseline_cost),
            "cost_reduction_pct": float(cost_reduction_pct),
        }

    @staticmethod
    def _compute_baseline(distance_matrix: list[list[float]], vehicles: list[dict[str, float]]) -> float:
        if len(distance_matrix) <= 1 or not vehicles:
            return 0.0
        min_cost_per_km = min(float(v.get("cost_per_km", 1.0)) for v in vehicles)
        return float(
            sum((2.0 * float(distance_matrix[0][node]) * min_cost_per_km) for node in range(1, len(distance_matrix)))
        )
