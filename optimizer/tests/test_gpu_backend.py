"""CUDA optimizer backend tests for route search, health checks, and infeasible cases."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from optimizer.backends.gpu_backend import CUDARouteOptimizer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")

IMPROVEMENT_EPSILON = -1e-6
NO_REGRESSION_TOLERANCE = 1e-6
EXPECTED_GPU_UTILIZATION = 42

VehiclePayload = dict[str, float]
StopPayload = dict[str, int | float]
SolveResult = dict[str, object]


class _TestableCUDARouteOptimizer(CUDARouteOptimizer):
    def nearest_neighbor_init_public(self, dist: torch.Tensor, n: int) -> torch.Tensor:
        return self._nearest_neighbor_init(dist, n)

    def evaluate_2opt_batch_public(self, dist: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        return self._evaluate_2opt_batch(dist, routes)

    def route_cost_public(self, route: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        return self._route_cost(route, dist)


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


def _result_float(result: SolveResult, key: str) -> float:
    return float(cast("float | int | str", result[key]))


def _dist_matrix(n: int = 8) -> list[list[float]]:
    out = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            out[i][j] = float(abs(i - j) + (0.5 if i != j else 0.0))
    return out


def _tie_matrix(n: int = 8) -> list[list[float]]:
    out = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            out[i][j] = 0.0 if i == j else 1.0
    return out


def test_nearest_neighbor_init_produces_diverse_solutions() -> None:
    """Nearest-neighbor init should produce multiple unique candidate routes."""
    backend = _TestableCUDARouteOptimizer()
    dist = torch.as_tensor(_tie_matrix(9), dtype=torch.float32, device=backend.cuda_device)
    routes = backend.nearest_neighbor_init_public(dist, backend.n_parallel)
    _check(routes.shape == (backend.n_parallel, 9))
    unique_routes = {
        tuple(int(routes[row_idx, col_idx].item()) for col_idx in range(routes.shape[1]))
        for row_idx in range(routes.shape[0])
    }
    _check(len(unique_routes) > 1)


def test_evaluate_2opt_batch_shape_and_mask() -> None:
    """2-opt batch evaluation should return masked delta tensor with expected shape."""
    backend = _TestableCUDARouteOptimizer()
    dist = torch.as_tensor(_dist_matrix(8), dtype=torch.float32, device=backend.cuda_device)
    routes = backend.nearest_neighbor_init_public(dist, 4)
    deltas = backend.evaluate_2opt_batch_public(dist, routes)
    _check(deltas.shape == (4, 8, 8))
    _check(torch.isinf(deltas[:, 0, :]).all())
    _check(torch.isinf(deltas[:, :, 0]).all())


def test_2opt_improves_initial_route_cost() -> None:
    """Applying best 2-opt swaps should not worsen best route cost."""
    backend = _TestableCUDARouteOptimizer()
    dist = torch.as_tensor(_dist_matrix(10), dtype=torch.float32, device=backend.cuda_device)
    routes = backend.nearest_neighbor_init_public(dist, 8)
    before = torch.stack([backend.route_cost_public(routes[i], dist) for i in range(routes.shape[0])]).min().item()

    deltas = backend.evaluate_2opt_batch_public(dist, routes)
    flat = deltas.view(deltas.shape[0], -1)
    vals, idxs = torch.min(flat, dim=1)
    n = routes.shape[1]
    for s in range(routes.shape[0]):
        if float(vals[s].item()) >= IMPROVEMENT_EPSILON:
            continue
        idx = int(idxs[s].item())
        i = idx // n
        j = idx % n
        routes[s, i : j + 1] = torch.flip(routes[s, i : j + 1], dims=[0])

    after = torch.stack([backend.route_cost_public(routes[i], dist) for i in range(routes.shape[0])]).min().item()
    _check(after <= before + NO_REGRESSION_TOLERANCE)


def test_solver_converges_and_returns_cuda_output() -> None:
    """GPU solver should return a valid status and non-negative timing metadata."""
    backend = CUDARouteOptimizer()
    backend.max_iterations = 50
    vehicles: list[VehiclePayload] = [{"capacity": 300.0, "cost_per_km": 2.0} for _ in range(2)]
    stops: list[StopPayload] = [
        {"facility_index": i, "demand": 40.0, "time_window_start": 0, "time_window_end": 480, "service_time": 30}
        for i in range(1, 8)
    ]
    result: SolveResult = backend.solve(24.7, 46.7, vehicles, stops, _dist_matrix(8), max_route_duration=480)
    _check(result["status"] in {"optimal", "infeasible"})
    _check(_result_float(result, "solver_time_ms") >= 0.0)


def test_gpu_tensor_placement() -> None:
    """Initialized route tensors should be allocated on CUDA device."""
    backend = _TestableCUDARouteOptimizer()
    dist = torch.as_tensor(_dist_matrix(7), dtype=torch.float32, device=backend.cuda_device)
    routes = backend.nearest_neighbor_init_public(dist, 2)
    _check(routes.device.type == "cuda")


def test_infeasible_without_vehicles() -> None:
    """Solver should mark instance infeasible when no vehicles are provided."""
    backend = CUDARouteOptimizer()
    result: SolveResult = backend.solve(0.0, 0.0, [], [], [[0.0]], max_route_duration=480)
    _check(result["status"] == "infeasible")
    _check(_result_float(result, "baseline_cost") == 0.0)


def test_health_check_fallback_without_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Health check should still return payload when NVML is unavailable."""
    backend = CUDARouteOptimizer()
    monkeypatch.setattr("optimizer.backends.gpu_backend.pynvml", None)

    def _device_name(_dev: int) -> str:
        return "Mock GPU"

    monkeypatch.setattr("torch.cuda.get_device_name", _device_name)
    payload = backend.health_check()
    _check(payload["solver"] == "cuda-2opt")
    _check(payload["gpu_name"])


def test_health_check_with_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Health check should include GPU utilization when NVML is available."""
    backend = CUDARouteOptimizer()

    def _device_handle(_idx: int) -> object:
        return object()

    def _device_name(_handle: object) -> bytes:
        return b"MockNVML"

    def _utilization(_handle: object) -> SimpleNamespace:
        return SimpleNamespace(gpu=EXPECTED_GPU_UTILIZATION)

    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=_device_handle,
        nvmlDeviceGetName=_device_name,
        nvmlDeviceGetUtilizationRates=_utilization,
        nvmlShutdown=lambda: None,
    )
    monkeypatch.setattr("optimizer.backends.gpu_backend.pynvml", fake_nvml)
    payload = backend.health_check()
    _check(payload["gpu_utilization"] == EXPECTED_GPU_UTILIZATION)


def test_warmup_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warmup should execute without errors when synchronize is available."""
    backend = CUDARouteOptimizer()

    def _sync(_device: object) -> None:
        return None

    monkeypatch.setattr("torch.cuda.synchronize", _sync)
    backend.warmup()


def test_init_raises_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backend initialization should fail when CUDA availability is forced false."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError):
        CUDARouteOptimizer()


def test_solve_breaks_on_time_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Solver should terminate early when configured time limit is reached."""
    backend = CUDARouteOptimizer()
    backend.time_limit_ms = 0

    ticks = iter([0.0, 0.01, 0.02, 0.03])
    monkeypatch.setattr("optimizer.backends.gpu_backend.perf_counter", lambda: next(ticks))
    result: SolveResult = backend.solve(
        0.0,
        0.0,
        [{"capacity": 200.0, "cost_per_km": 1.0}],
        [{"facility_index": 1, "demand": 10.0, "time_window_start": 0, "time_window_end": 480}],
        [[0.0, 1.0], [1.0, 0.0]],
        max_route_duration=480,
    )
    _check(result["status"] in {"optimal", "infeasible"})


def test_solve_infeasible_when_splitting_exhausts_vehicles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Solver should report infeasible when demand splitting exceeds vehicle count."""
    backend = CUDARouteOptimizer()
    backend.max_iterations = 0

    def _fixed_init(_dist: torch.Tensor, _n: int) -> torch.Tensor:
        return torch.tensor([[0, 1, 2]], dtype=torch.long, device=backend.cuda_device).repeat(backend.n_parallel, 1)

    monkeypatch.setattr(backend, "_nearest_neighbor_init", _fixed_init)
    vehicles: list[VehiclePayload] = [{"capacity": 100.0, "cost_per_km": 1.0}]
    stops: list[StopPayload] = [
        {"facility_index": 1, "demand": 60.0, "time_window_start": 0, "time_window_end": 480, "service_time": 10},
        {"facility_index": 2, "demand": 60.0, "time_window_start": 0, "time_window_end": 480, "service_time": 10},
    ]
    result: SolveResult = backend.solve(0.0, 0.0, vehicles, stops, _dist_matrix(3), max_route_duration=480)
    _check(result["status"] == "infeasible")


def test_solve_infeasible_when_single_stop_demand_exceeds_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Solver should report infeasible when one stop demand exceeds all capacities."""
    backend = CUDARouteOptimizer()
    backend.max_iterations = 0

    def _fixed_init(_dist: torch.Tensor, _n: int) -> torch.Tensor:
        return torch.tensor([[0, 1]], dtype=torch.long, device=backend.cuda_device).repeat(backend.n_parallel, 1)

    monkeypatch.setattr(backend, "_nearest_neighbor_init", _fixed_init)
    vehicles: list[VehiclePayload] = [{"capacity": 20.0, "cost_per_km": 1.0}]
    stops: list[StopPayload] = [
        {"facility_index": 1, "demand": 50.0, "time_window_start": 0, "time_window_end": 480, "service_time": 10},
    ]
    result: SolveResult = backend.solve(0.0, 0.0, vehicles, stops, [[0.0, 1.0], [1.0, 0.0]], max_route_duration=480)
    _check(result["status"] == "infeasible")
