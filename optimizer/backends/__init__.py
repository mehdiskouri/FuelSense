"""Optimizer backend registrations."""

from typing import Any

from optimizer.backends.cpu_backend import ORToolsOptimizer

_cuda_backend_cls: Any | None
try:
    from optimizer.backends import gpu_backend

    _cuda_backend_cls = gpu_backend.CUDARouteOptimizer
except ImportError:  # pragma: no cover - exercised in CPU-only images
    _cuda_backend_cls = None

CUDARouteOptimizer = _cuda_backend_cls

__all__ = ["CUDARouteOptimizer", "ORToolsOptimizer"]
