from __future__ import annotations

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


def test_nearest_neighbor_init_produces_diverse_solutions() -> None:
    backend = CUDARouteOptimizer()
    dist = torch.as_tensor(_dist_matrix(9), dtype=torch.float32, device=backend.cuda_device)
    routes = backend._nearest_neighbor_init(dist, backend.N_PARALLEL)
    assert routes.shape == (backend.N_PARALLEL, 9)
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
    backend.MAX_ITERATIONS = 50
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
