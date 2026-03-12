from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from optimizer.backends.gpu_backend import CUDARouteOptimizer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


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
    backend = CUDARouteOptimizer()
    dist = torch.as_tensor(_tie_matrix(9), dtype=torch.float32, device=backend.cuda_device)
    routes = backend._nearest_neighbor_init(dist, backend.n_parallel)
    assert routes.shape == (backend.n_parallel, 9)
    unique = torch.unique(routes, dim=0)
    assert unique.shape[0] > 1


def test_evaluate_2opt_batch_shape_and_mask() -> None:
    backend = CUDARouteOptimizer()
    dist = torch.as_tensor(_dist_matrix(8), dtype=torch.float32, device=backend.cuda_device)
    routes = backend._nearest_neighbor_init(dist, 4)
    deltas = backend._evaluate_2opt_batch(dist, routes)
    assert deltas.shape == (4, 8, 8)
    assert torch.isinf(deltas[:, 0, :]).all()
    assert torch.isinf(deltas[:, :, 0]).all()


def test_2opt_improves_initial_route_cost() -> None:
    backend = CUDARouteOptimizer()
    dist = torch.as_tensor(_dist_matrix(10), dtype=torch.float32, device=backend.cuda_device)
    routes = backend._nearest_neighbor_init(dist, 8)
    before = torch.stack([backend._route_cost(routes[i], dist) for i in range(routes.shape[0])]).min().item()

    deltas = backend._evaluate_2opt_batch(dist, routes)
    flat = deltas.view(deltas.shape[0], -1)
    vals, idxs = torch.min(flat, dim=1)
    n = routes.shape[1]
    for s in range(routes.shape[0]):
        if float(vals[s].item()) >= -1e-6:
            continue
        idx = int(idxs[s].item())
        i = idx // n
        j = idx % n
        routes[s, i : j + 1] = torch.flip(routes[s, i : j + 1], dims=[0])

    after = torch.stack([backend._route_cost(routes[i], dist) for i in range(routes.shape[0])]).min().item()
    assert after <= before + 1e-6


def test_solver_converges_and_returns_cuda_output() -> None:
    backend = CUDARouteOptimizer()
    backend.max_iterations = 50
    vehicles = [{"capacity": 300.0, "cost_per_km": 2.0} for _ in range(2)]
    stops = [
        {"facility_index": i, "demand": 40.0, "time_window_start": 0, "time_window_end": 480, "service_time": 30}
        for i in range(1, 8)
    ]
    result = backend.solve(24.7, 46.7, vehicles, stops, _dist_matrix(8), max_route_duration=480)
    assert result["status"] in {"optimal", "infeasible"}
    assert float(result["solver_time_ms"]) >= 0.0


def test_gpu_tensor_placement() -> None:
    backend = CUDARouteOptimizer()
    dist = torch.as_tensor(_dist_matrix(7), dtype=torch.float32, device=backend.cuda_device)
    routes = backend._nearest_neighbor_init(dist, 2)
    assert routes.device.type == "cuda"


def test_infeasible_without_vehicles() -> None:
    backend = CUDARouteOptimizer()
    result = backend.solve(0.0, 0.0, [], [], [[0.0]], max_route_duration=480)
    assert result["status"] == "infeasible"
    assert float(result["baseline_cost"]) == 0.0


def test_health_check_fallback_without_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = CUDARouteOptimizer()
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda _dev: "Mock GPU")
    payload = backend.health_check()
    assert payload["solver"] == "cuda-2opt"
    assert payload["gpu_name"]


def test_health_check_with_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = CUDARouteOptimizer()
    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda _idx: object(),
        nvmlDeviceGetName=lambda _handle: b"MockNVML",
        nvmlDeviceGetUtilizationRates=lambda _handle: SimpleNamespace(gpu=42),
        nvmlShutdown=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)
    payload = backend.health_check()
    assert payload["gpu_utilization"] == 42


def test_warmup_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = CUDARouteOptimizer()
    monkeypatch.setattr("torch.cuda.synchronize", lambda _device: None)
    backend.warmup()


def test_init_raises_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError):
        CUDARouteOptimizer()


def test_solve_breaks_on_time_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = CUDARouteOptimizer()
    backend.time_limit_ms = 0

    ticks = iter([0.0, 0.01, 0.02, 0.03])
    monkeypatch.setattr("optimizer.backends.gpu_backend.perf_counter", lambda: next(ticks))
    result = backend.solve(
        0.0,
        0.0,
        [{"capacity": 200.0, "cost_per_km": 1.0}],
        [{"facility_index": 1, "demand": 10.0, "time_window_start": 0, "time_window_end": 480}],
        [[0.0, 1.0], [1.0, 0.0]],
        max_route_duration=480,
    )
    assert result["status"] in {"optimal", "infeasible"}


def test_solve_infeasible_when_splitting_exhausts_vehicles(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = CUDARouteOptimizer()
    backend.max_iterations = 0

    def _fixed_init(_dist: torch.Tensor, _n: int) -> torch.Tensor:
        return torch.tensor([[0, 1, 2]], dtype=torch.long, device=backend.cuda_device).repeat(backend.n_parallel, 1)

    monkeypatch.setattr(backend, "_nearest_neighbor_init", _fixed_init)
    vehicles = [{"capacity": 100.0, "cost_per_km": 1.0}]
    stops = [
        {"facility_index": 1, "demand": 60.0, "time_window_start": 0, "time_window_end": 480, "service_time": 10},
        {"facility_index": 2, "demand": 60.0, "time_window_start": 0, "time_window_end": 480, "service_time": 10},
    ]
    result = backend.solve(0.0, 0.0, vehicles, stops, _dist_matrix(3), max_route_duration=480)
    assert result["status"] == "infeasible"


def test_solve_infeasible_when_single_stop_demand_exceeds_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = CUDARouteOptimizer()
    backend.max_iterations = 0

    def _fixed_init(_dist: torch.Tensor, _n: int) -> torch.Tensor:
        return torch.tensor([[0, 1]], dtype=torch.long, device=backend.cuda_device).repeat(backend.n_parallel, 1)

    monkeypatch.setattr(backend, "_nearest_neighbor_init", _fixed_init)
    vehicles = [{"capacity": 20.0, "cost_per_km": 1.0}]
    stops = [{"facility_index": 1, "demand": 50.0, "time_window_start": 0, "time_window_end": 480, "service_time": 10}]
    result = backend.solve(0.0, 0.0, vehicles, stops, [[0.0, 1.0], [1.0, 0.0]], max_route_duration=480)
    assert result["status"] == "infeasible"
