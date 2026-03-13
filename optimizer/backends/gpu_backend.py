"""CUDA backend for route optimization via parallel 2-opt."""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false

from __future__ import annotations

import os
from time import perf_counter
from typing import Any

import torch

from fuelsense_common.compute import ComputeBackend, DeviceType
from fuelsense_common.registry import register_backend

pynvml: Any | None
try:
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None


DELTA_TOLERANCE = 1e-6
MAX_NO_IMPROVE_ROUNDS = 10


@register_backend("route_optimizer", DeviceType.CUDA)
class CUDARouteOptimizer(ComputeBackend):
    """GPU-accelerated route optimizer using batched 2-opt local search."""

    device = DeviceType.CUDA
    DEFAULT_N_PARALLEL = 64
    DEFAULT_MAX_ITERATIONS = 1000

    def __init__(self) -> None:
        """Initialize CUDA runtime configuration and optimization parameters."""
        if not torch.cuda.is_available():
            msg = "CUDA backend requested but CUDA is not available"
            raise RuntimeError(msg)
        self.cuda_device = torch.device("cuda:0")
        self.time_limit_ms = int(os.environ.get("FUELSENSE_OPTIMIZER_TIME_LIMIT_MS", "10000"))
        self.n_parallel = int(os.environ.get("FUELSENSE_OPTIMIZER_N_PARALLEL", str(self.DEFAULT_N_PARALLEL)))
        self.max_iterations = int(
            os.environ.get("FUELSENSE_OPTIMIZER_GPU_MAX_ITERATIONS", str(self.DEFAULT_MAX_ITERATIONS)),
        )
        # Optional bound for 2-opt neighborhood width. 0 disables pruning.
        self.max_swap_span = int(os.environ.get("FUELSENSE_OPTIMIZER_MAX_SWAP_SPAN", "0"))

    def warmup(self) -> None:
        """Run a tiny solve to warm CUDA kernels and memory allocations."""
        tiny = [
            [0.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.0, 0.0, 2.0, 3.0, 4.0, 5.0],
            [3.0, 2.0, 0.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 0.0, 2.0, 3.0],
            [5.0, 4.0, 3.0, 2.0, 0.0, 2.0],
            [6.0, 5.0, 4.0, 3.0, 2.0, 0.0],
        ]
        _ = self.solve(
            depot_lat=0.0,
            depot_lng=0.0,
            vehicles=[{"capacity": 1000.0, "cost_per_km": 1.0}],
            stops=[
                {"facility_index": i, "demand": 10.0, "time_window_start": 0, "time_window_end": 480}
                for i in range(1, 6)
            ],
            distance_matrix=tiny,
            max_route_duration=480,
        )
        torch.cuda.synchronize(self.cuda_device)

    def health_check(self) -> dict[str, object]:
        """Return health metadata for the loaded CUDA optimizer backend."""
        mem_free, mem_total = torch.cuda.mem_get_info(device=self.cuda_device)
        out: dict[str, object] = {
            "device": self.device.value,
            "gpu_memory_free_gb": float(mem_free / (1024**3)),
            "gpu_memory_total_gb": float(mem_total / (1024**3)),
            "solver": "cuda-2opt",
            "n_parallel": self.n_parallel,
        }
        if pynvml is not None:
            nvml: Any = pynvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            out["gpu_name"] = str(nvml.nvmlDeviceGetName(handle))
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            out["gpu_utilization"] = int(util.gpu)
            nvml.nvmlShutdown()
        else:
            out["gpu_name"] = torch.cuda.get_device_name(self.cuda_device)
            out["gpu_utilization"] = None
        return out

    def _nearest_neighbor_init(self, dist: torch.Tensor, n_solutions: int) -> torch.Tensor:
        n = int(dist.shape[0])
        routes = torch.zeros((n_solutions, n), dtype=torch.long, device=self.cuda_device)
        base = torch.arange(n, device=self.cuda_device)
        for s in range(n_solutions):
            visited = torch.zeros(n, dtype=torch.bool, device=self.cuda_device)
            current = 0
            routes[s, 0] = 0
            visited[0] = True
            for pos in range(1, n):
                row = dist[current].clone()
                row[visited] = float("inf")
                noise = -torch.log(-torch.log(torch.rand(n, device=self.cuda_device).clamp_min(1e-6))).mul_(0.15)
                score = row + noise
                nxt = int(base[torch.argmin(score)].item())
                routes[s, pos] = nxt
                visited[nxt] = True
                current = nxt
        return routes

    def _evaluate_2opt_batch(self, dist: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        n_parallel, n = routes.shape

        # Build [P, N, N] index grids once per call and evaluate all swap deltas in parallel.
        device = self.cuda_device
        i_idx = torch.arange(n, device=device).view(1, n, 1).expand(n_parallel, n, n)
        j_idx = torch.arange(n, device=device).view(1, 1, n).expand(n_parallel, n, n)
        p_idx = torch.arange(n_parallel, device=device).view(n_parallel, 1, 1).expand(n_parallel, n, n)

        a_nodes = routes[p_idx, (i_idx - 1).clamp_min(0)]
        b_nodes = routes[p_idx, i_idx]
        c_nodes = routes[p_idx, j_idx]
        d_nodes = routes[p_idx, (j_idx + 1) % n]

        deltas = dist[a_nodes, c_nodes] + dist[b_nodes, d_nodes] - dist[a_nodes, b_nodes] - dist[c_nodes, d_nodes]

        invalid = (i_idx <= 0) | (j_idx <= 0) | (j_idx <= i_idx)
        if self.max_swap_span > 0:
            invalid = invalid | ((j_idx - i_idx) > self.max_swap_span)
        return deltas.masked_fill(invalid, float("inf"))

    @staticmethod
    def _route_cost(route: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        nxt = torch.roll(route, shifts=-1)
        return dist[route, nxt].sum()

    @staticmethod
    def _route_cost_batch(routes: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        nxt = torch.roll(routes, shifts=-1, dims=1)
        return dist[routes, nxt].sum(dim=1)

    @staticmethod
    def _build_infeasible_response(
        start: float,
        distance_matrix: list[list[float]],
        vehicles: list[dict[str, float]],
    ) -> dict[str, object]:
        solver_time_ms = (perf_counter() - start) * 1000
        baseline_cost = CUDARouteOptimizer._compute_baseline(distance_matrix, vehicles)
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

    def _run_parallel_2opt_search(
        self,
        dist: torch.Tensor,
        routes: torch.Tensor,
        start: float,
    ) -> torch.Tensor:
        costs = self._route_cost_batch(routes, dist)
        best_idx = int(torch.argmin(costs).item())
        best_route = routes[best_idx].clone()
        best_cost = float(costs[best_idx].item())

        no_improve = 0
        for _ in range(self.max_iterations):
            if (perf_counter() - start) * 1000 > self.time_limit_ms:
                break

            deltas = self._evaluate_2opt_batch(dist, routes)
            flat = deltas.view(deltas.shape[0], -1)
            best_delta_vals, best_delta_idx = torch.min(flat, dim=1)
            improved_any = self._apply_best_swaps(routes, best_delta_vals, best_delta_idx)

            if not improved_any:
                no_improve += 1
                if no_improve > MAX_NO_IMPROVE_ROUNDS:
                    break
                continue

            best_route, best_cost, no_improve = self._update_search_best_route(
                routes,
                dist,
                best_route,
                best_cost,
                no_improve,
            )

        return best_route

    @staticmethod
    def _apply_best_swaps(
        routes: torch.Tensor,
        best_delta_vals: torch.Tensor,
        best_delta_idx: torch.Tensor,
    ) -> bool:
        improved_any = False
        route_len = routes.shape[1]
        for solution_idx in range(routes.shape[0]):
            delta_val = float(best_delta_vals[solution_idx].item())
            if delta_val >= -DELTA_TOLERANCE:
                continue
            idx = int(best_delta_idx[solution_idx].item())
            i = idx // route_len
            j = idx % route_len
            routes[solution_idx, i : j + 1] = torch.flip(routes[solution_idx, i : j + 1], dims=[0])
            improved_any = True
        return improved_any

    def _update_search_best_route(
        self,
        routes: torch.Tensor,
        dist: torch.Tensor,
        best_route: torch.Tensor,
        best_cost: float,
        no_improve: int,
    ) -> tuple[torch.Tensor, float, int]:
        costs = self._route_cost_batch(routes, dist)
        idx = int(torch.argmin(costs).item())
        val = float(costs[idx].item())
        if val + DELTA_TOLERANCE < best_cost:
            return routes[idx].clone(), val, 0
        updated_no_improve = no_improve + 1
        if updated_no_improve > MAX_NO_IMPROVE_ROUNDS:
            return best_route, best_cost, updated_no_improve
        return best_route, best_cost, updated_no_improve

    @staticmethod
    def _split_route_by_capacity(
        best_path: list[int],
        stop_map: dict[int, dict[str, float | int]],
        capacities: list[float],
    ) -> list[list[int]] | None:
        split: list[list[int]] = [[] for _ in capacities]
        vehicle_idx = 0
        used = 0.0
        for node in best_path:
            demand = float(stop_map.get(node, {}).get("demand", 0.0))
            if vehicle_idx >= len(capacities):  # pragma: no cover
                return None
            if demand > capacities[vehicle_idx]:
                return None
            if used + demand > capacities[vehicle_idx] and split[vehicle_idx]:
                vehicle_idx += 1
                used = 0.0
                if vehicle_idx >= len(capacities):
                    return None
            split[vehicle_idx].append(node)
            used += demand
        return split

    @staticmethod
    def _build_routes_output(
        split: list[list[int]],
        distance_matrix: list[list[float]],
        stop_map: dict[int, dict[str, float | int]],
        costs_per_km: list[float],
    ) -> tuple[list[dict[str, object]], float, float]:
        routes_out: list[dict[str, object]] = []
        total_distance = 0.0
        total_cost = 0.0
        for v_idx, nodes in enumerate(split):
            if not nodes:
                continue
            seq = 1
            arrival = 0
            prev = 0
            distance = 0.0
            route_stops: list[dict[str, object]] = []
            for node in nodes:
                distance += float(distance_matrix[prev][node])
                arrival += round(float(distance_matrix[prev][node]))
                route_stops.append(
                    {
                        "facility_index": int(node),
                        "demand": float(stop_map.get(node, {}).get("demand", 0.0)),
                        "arrival_min": int(arrival),
                        "sequence": seq,
                    },
                )
                arrival += int(stop_map.get(node, {}).get("service_time", 30))
                seq += 1
                prev = node
            distance += float(distance_matrix[prev][0])

            route_cost = distance * costs_per_km[v_idx]
            routes_out.append(
                {
                    "vehicle_index": v_idx,
                    "stops": route_stops,
                    "distance_km": float(distance),
                    "cost": float(route_cost),
                },
            )
            total_distance += distance
            total_cost += route_cost
        return routes_out, total_distance, total_cost

    @staticmethod
    def _build_optimal_response(
        routes_out: list[dict[str, object]],
        total_distance: float,
        total_cost: float,
        start: float,
        baseline_cost: float,
    ) -> dict[str, object]:
        reduction = ((baseline_cost - total_cost) / baseline_cost * 100.0) if baseline_cost > 0 else 0.0
        return {
            "status": "optimal" if routes_out else "infeasible",
            "routes": routes_out,
            "total_distance_km": float(total_distance),
            "total_cost": float(total_cost),
            "vehicles_used": len(routes_out),
            "solver_time_ms": (perf_counter() - start) * 1000,
            "baseline_cost": float(baseline_cost),
            "cost_reduction_pct": float(max(reduction, 0.0)),
        }

    def solve(  # noqa: PLR0913
        self,
        depot_lat: float,
        depot_lng: float,
        vehicles: list[dict[str, float]],
        stops: list[dict[str, float | int]],
        distance_matrix: list[list[float]],
        max_route_duration: int,
    ) -> dict[str, object]:
        """Solve CVRP-like routing with parallel 2-opt and capacity-aware splitting."""
        _ = depot_lat, depot_lng, max_route_duration
        start = perf_counter()
        if len(distance_matrix) <= 1 or not vehicles:
            return self._build_infeasible_response(start, distance_matrix, vehicles)

        dist = torch.as_tensor(distance_matrix, dtype=torch.float32, device=self.cuda_device)
        routes = self._nearest_neighbor_init(dist, self.n_parallel)
        best_route = self._run_parallel_2opt_search(dist, routes, start)

        best_path = [int(x) for x in best_route.detach().cpu().tolist() if int(x) != 0]
        stop_map = {int(s.get("facility_index", -1)): s for s in stops}
        capacities = [float(v.get("capacity", 0.0)) for v in vehicles]
        costs_per_km = [float(v.get("cost_per_km", 1.0)) for v in vehicles]

        split = self._split_route_by_capacity(best_path, stop_map, capacities)
        if split is None:
            return self._build_infeasible_response(start, distance_matrix, vehicles)

        routes_out, total_distance, total_cost = self._build_routes_output(
            split,
            distance_matrix,
            stop_map,
            costs_per_km,
        )
        baseline_cost = self._compute_baseline(distance_matrix, vehicles)
        return self._build_optimal_response(routes_out, total_distance, total_cost, start, baseline_cost)

    @staticmethod
    def _compute_baseline(distance_matrix: list[list[float]], vehicles: list[dict[str, float]]) -> float:
        if len(distance_matrix) <= 1 or not vehicles:
            return 0.0
        min_cost_per_km = min(float(v.get("cost_per_km", 1.0)) for v in vehicles)
        return float(
            sum((2.0 * float(distance_matrix[0][node]) * min_cost_per_km) for node in range(1, len(distance_matrix))),
        )
