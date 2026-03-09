"""CUDA backend for route optimization via parallel 2-opt."""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false

from __future__ import annotations

import os
from time import perf_counter
from typing import Any

import numpy as np
import torch

from fuelsense_common.compute import ComputeBackend, DeviceType
from fuelsense_common.registry import register_backend


@register_backend("route_optimizer", DeviceType.CUDA)
class CUDARouteOptimizer(ComputeBackend):
	device = DeviceType.CUDA
	N_PARALLEL = 64
	MAX_ITERATIONS = 1000

	def __init__(self) -> None:
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA backend requested but CUDA is not available")
		self.cuda_device = torch.device("cuda:0")
		self.time_limit_ms = int(os.environ.get("FUELSENSE_OPTIMIZER_TIME_LIMIT_MS", "10000"))

	def warmup(self) -> None:
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
			stops=[{"facility_index": i, "demand": 10.0, "time_window_start": 0, "time_window_end": 480} for i in range(1, 6)],
			distance_matrix=tiny,
			max_route_duration=480,
		)
		torch.cuda.synchronize(self.cuda_device)

	def health_check(self) -> dict[str, object]:
		mem_free, mem_total = torch.cuda.mem_get_info(device=self.cuda_device)
		out: dict[str, object] = {
			"device": self.device.value,
			"gpu_memory_free_gb": float(mem_free / (1024**3)),
			"gpu_memory_total_gb": float(mem_total / (1024**3)),
			"solver": "cuda-2opt",
			"n_parallel": self.N_PARALLEL,
		}
		try:
			import pynvml

			nvml: Any = pynvml
			nvml.nvmlInit()
			handle = nvml.nvmlDeviceGetHandleByIndex(0)
			out["gpu_name"] = str(nvml.nvmlDeviceGetName(handle))
			util = nvml.nvmlDeviceGetUtilizationRates(handle)
			out["gpu_utilization"] = int(util.gpu)
			nvml.nvmlShutdown()
		except Exception:
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
		deltas = torch.full((n_parallel, n, n), float("inf"), device=self.cuda_device)
		for s in range(n_parallel):
			route = routes[s]
			for i in range(1, n - 1):
				a = int(route[i - 1].item())
				b = int(route[i].item())
				for j in range(i + 1, n):
					c = int(route[j].item())
					d = int(route[(j + 1) % n].item())
					delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
					deltas[s, i, j] = delta
		return deltas

	@staticmethod
	def _route_cost(route: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
		nxt = torch.roll(route, shifts=-1)
		return dist[route, nxt].sum()

	def solve(
		self,
		depot_lat: float,
		depot_lng: float,
		vehicles: list[dict[str, float]],
		stops: list[dict[str, float | int]],
		distance_matrix: list[list[float]],
		max_route_duration: int,
	) -> dict[str, object]:
		_ = depot_lat, depot_lng, max_route_duration
		start = perf_counter()
		if len(distance_matrix) <= 1 or not vehicles:
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

		dist = torch.as_tensor(distance_matrix, dtype=torch.float32, device=self.cuda_device)
		routes = self._nearest_neighbor_init(dist, self.N_PARALLEL)
		costs = torch.stack([self._route_cost(routes[i], dist) for i in range(routes.shape[0])])
		best_idx = int(torch.argmin(costs).item())
		best_route = routes[best_idx].clone()
		best_cost = float(costs[best_idx].item())

		no_improve = 0
		for _ in range(self.MAX_ITERATIONS):
			if (perf_counter() - start) * 1000 > self.time_limit_ms:
				break

			deltas = self._evaluate_2opt_batch(dist, routes)
			flat = deltas.view(deltas.shape[0], -1)
			best_delta_vals, best_delta_idx = torch.min(flat, dim=1)

			improved_any = False
			n = routes.shape[1]
			for s in range(routes.shape[0]):
				delta_val = float(best_delta_vals[s].item())
				if delta_val >= -1e-6:
					continue
				idx = int(best_delta_idx[s].item())
				i = idx // n
				j = idx % n
				routes[s, i : j + 1] = torch.flip(routes[s, i : j + 1], dims=[0])
				improved_any = True

			if not improved_any:
				no_improve += 1
				if no_improve > 10:
					break
				continue

			costs = torch.stack([self._route_cost(routes[i], dist) for i in range(routes.shape[0])])
			idx = int(torch.argmin(costs).item())
			val = float(costs[idx].item())
			if val + 1e-6 < best_cost:
				best_cost = val
				best_route = routes[idx].clone()
				no_improve = 0
			else:
				no_improve += 1
				if no_improve > 10:
					break

		best_path = [int(x) for x in best_route.detach().cpu().tolist() if int(x) != 0]
		stop_map = {int(s.get("facility_index", -1)): s for s in stops}
		capacities = [float(v.get("capacity", 0.0)) for v in vehicles]
		costs_per_km = [float(v.get("cost_per_km", 1.0)) for v in vehicles]

		split: list[list[int]] = [[] for _ in vehicles]
		vehicle_idx = 0
		used = 0.0
		for node in best_path:
			demand = float(stop_map.get(node, {}).get("demand", 0.0))
			if vehicle_idx >= len(capacities):
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
			if demand > capacities[vehicle_idx]:
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
			if used + demand > capacities[vehicle_idx] and split[vehicle_idx]:
				vehicle_idx += 1
				used = 0.0
				if vehicle_idx >= len(capacities):
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
			split[vehicle_idx].append(node)
			used += demand

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
				arrival += int(round(float(distance_matrix[prev][node])))
				route_stops.append(
					{
						"facility_index": int(node),
						"demand": float(stop_map.get(node, {}).get("demand", 0.0)),
						"arrival_min": int(arrival),
						"sequence": seq,
					}
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
				}
			)
			total_distance += distance
			total_cost += route_cost

		baseline_cost = self._compute_baseline(distance_matrix, vehicles)
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

	@staticmethod
	def _compute_baseline(distance_matrix: list[list[float]], vehicles: list[dict[str, float]]) -> float:
		if len(distance_matrix) <= 1 or not vehicles:
			return 0.0
		min_cost_per_km = min(float(v.get("cost_per_km", 1.0)) for v in vehicles)
		return float(
			sum((2.0 * float(distance_matrix[0][node]) * min_cost_per_km) for node in range(1, len(distance_matrix)))
		)
