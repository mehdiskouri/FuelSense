"""Optimizer backend registrations."""

from optimizer.backends.cpu_backend import ORToolsOptimizer

try:
    from optimizer.backends.gpu_backend import CUDARouteOptimizer
except Exception:  # pragma: no cover - exercised in CPU-only images
    CUDARouteOptimizer = None  # type: ignore[assignment]

__all__ = ["CUDARouteOptimizer", "ORToolsOptimizer"]
