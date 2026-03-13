"""Tests for backend registry registration, lookup, and reset behavior."""

from __future__ import annotations

import pytest

from fuelsense_common.compute import ComputeBackend, DeviceType
from fuelsense_common.registry import clear_registry, get_backend, register_backend


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _BaseBackend:
    device: DeviceType

    def warmup(self) -> None:
        return None

    def health_check(self) -> dict[str, object]:
        return {"status": "ok", "device": self.device.value}


class _CpuBackend(_BaseBackend):
    device = DeviceType.CPU


class _CudaBackend(_BaseBackend):
    device = DeviceType.CUDA


def test_register_and_get_backend_round_trip_cpu() -> None:
    """CPU backend registration should round-trip through lookup."""

    @register_backend("test_service", DeviceType.CPU)
    class TestCpu(_CpuBackend):
        pass

    _ = TestCpu

    backend = get_backend("test_service", DeviceType.CPU)
    _check(isinstance(backend, TestCpu), "Expected CPU backend round-trip")


def test_get_backend_prefers_requested_cuda() -> None:
    """Lookup should return CUDA backend when one is explicitly registered."""

    @register_backend("test_service", DeviceType.CPU)
    class TestCpu(_CpuBackend):
        pass

    _ = TestCpu

    @register_backend("test_service", DeviceType.CUDA)
    class TestCuda(_CudaBackend):
        pass

    backend = get_backend("test_service", DeviceType.CUDA)
    _check(isinstance(backend, TestCuda), "Expected explicit CUDA backend")


def test_get_backend_falls_back_to_cpu() -> None:
    """CUDA lookup should fall back to CPU when CUDA is unavailable."""

    @register_backend("test_service", DeviceType.CPU)
    class TestCpu(_CpuBackend):
        pass

    _ = TestCpu

    backend = get_backend("test_service", DeviceType.CUDA)
    _check(isinstance(backend, TestCpu), "Expected fallback to CPU backend")


def test_get_backend_raises_when_missing() -> None:
    """Lookup should fail with a clear error when service has no backend."""
    with pytest.raises(RuntimeError, match="No backend registered"):
        get_backend("missing", DeviceType.CPU)


def test_registry_supports_multiple_services() -> None:
    """Backend registry should isolate registrations by service name."""

    @register_backend("forecaster", DeviceType.CPU)
    class ForecasterCpu(_CpuBackend):
        pass

    @register_backend("optimizer", DeviceType.CPU)
    class OptimizerCpu(_CpuBackend):
        pass

    forecaster_backend = get_backend("forecaster", DeviceType.CPU)
    optimizer_backend = get_backend("optimizer", DeviceType.CPU)

    _check(isinstance(forecaster_backend, ForecasterCpu), "Forecaster service should return forecaster backend")
    _check(isinstance(optimizer_backend, OptimizerCpu), "Optimizer service should return optimizer backend")


def test_clear_registry_resets_state() -> None:
    """Registry clear operation should remove previously registered backends."""

    @register_backend("test_service", DeviceType.CPU)
    class TestCpu(_CpuBackend):
        pass

    _ = TestCpu

    _ = get_backend("test_service", DeviceType.CPU)
    clear_registry()

    with pytest.raises(RuntimeError, match="No backend registered"):
        get_backend("test_service", DeviceType.CPU)


def test_registered_backend_is_compute_backend() -> None:
    """Registered backends should satisfy the shared `ComputeBackend` contract."""

    @register_backend("test_service", DeviceType.CPU)
    class TestCpu(_CpuBackend):
        pass

    _ = TestCpu

    backend = get_backend("test_service", DeviceType.CPU)
    _check(isinstance(backend, ComputeBackend), "Registered backend should implement ComputeBackend")
