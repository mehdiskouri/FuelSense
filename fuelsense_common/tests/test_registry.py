from __future__ import annotations

import pytest

from fuelsense_common.compute import ComputeBackend, DeviceType
from fuelsense_common.registry import clear_registry, get_backend, register_backend


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
    @register_backend("test_service", DeviceType.CPU)
    class TestCpu(_CpuBackend):
        pass

    _ = TestCpu

    backend = get_backend("test_service", DeviceType.CPU)
    assert isinstance(backend, TestCpu)


def test_get_backend_prefers_requested_cuda() -> None:
    @register_backend("test_service", DeviceType.CPU)
    class TestCpu(_CpuBackend):
        pass

    _ = TestCpu

    @register_backend("test_service", DeviceType.CUDA)
    class TestCuda(_CudaBackend):
        pass

    backend = get_backend("test_service", DeviceType.CUDA)
    assert isinstance(backend, TestCuda)


def test_get_backend_falls_back_to_cpu() -> None:
    @register_backend("test_service", DeviceType.CPU)
    class TestCpu(_CpuBackend):
        pass

    _ = TestCpu

    backend = get_backend("test_service", DeviceType.CUDA)
    assert isinstance(backend, TestCpu)


def test_get_backend_raises_when_missing() -> None:
    with pytest.raises(RuntimeError, match="No backend registered"):
        get_backend("missing", DeviceType.CPU)


def test_registry_supports_multiple_services() -> None:
    @register_backend("forecaster", DeviceType.CPU)
    class ForecasterCpu(_CpuBackend):
        pass

    @register_backend("optimizer", DeviceType.CPU)
    class OptimizerCpu(_CpuBackend):
        pass

    forecaster_backend = get_backend("forecaster", DeviceType.CPU)
    optimizer_backend = get_backend("optimizer", DeviceType.CPU)

    assert isinstance(forecaster_backend, ForecasterCpu)
    assert isinstance(optimizer_backend, OptimizerCpu)


def test_clear_registry_resets_state() -> None:
    @register_backend("test_service", DeviceType.CPU)
    class TestCpu(_CpuBackend):
        pass

    _ = TestCpu

    _ = get_backend("test_service", DeviceType.CPU)
    clear_registry()

    with pytest.raises(RuntimeError, match="No backend registered"):
        get_backend("test_service", DeviceType.CPU)


def test_registered_backend_is_compute_backend() -> None:
    @register_backend("test_service", DeviceType.CPU)
    class TestCpu(_CpuBackend):
        pass

    _ = TestCpu

    backend = get_backend("test_service", DeviceType.CPU)
    assert isinstance(backend, ComputeBackend)
