from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from fuelsense_common.compute import ComputeBackend, DeviceType, resolve_device


class _TorchCudaAvailable:
    def __init__(self) -> None:
        self.cuda = Mock()
        self.cuda.is_available.return_value = True
        self.cuda.get_device_name.return_value = "Mock GPU"


class _TorchCudaUnavailable:
    def __init__(self) -> None:
        self.cuda = Mock()
        self.cuda.is_available.return_value = False


def test_resolve_device_explicit_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FUELSENSE_DEVICE", "cpu")
    assert resolve_device() == DeviceType.CPU


def test_resolve_device_explicit_cuda_available(monkeypatch: pytest.MonkeyPatch) -> None:
    def _import_torch(_name: str) -> _TorchCudaAvailable:
        return _TorchCudaAvailable()

    monkeypatch.setenv("FUELSENSE_DEVICE", "cuda")
    monkeypatch.setattr("fuelsense_common.compute.importlib.import_module", _import_torch)
    assert resolve_device() == DeviceType.CUDA


def test_resolve_device_explicit_cuda_fallback(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def _import_torch(_name: str) -> _TorchCudaUnavailable:
        return _TorchCudaUnavailable()

    monkeypatch.setenv("FUELSENSE_DEVICE", "cuda")
    monkeypatch.setattr("fuelsense_common.compute.importlib.import_module", _import_torch)
    result = resolve_device()
    assert result == DeviceType.CPU
    assert "Falling back to CPU" in caplog.text


def test_resolve_device_auto_cuda(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def _import_torch(_name: str) -> _TorchCudaAvailable:
        return _TorchCudaAvailable()

    monkeypatch.delenv("FUELSENSE_DEVICE", raising=False)
    monkeypatch.setattr("fuelsense_common.compute.importlib.import_module", _import_torch)
    result = resolve_device()
    assert result == DeviceType.CUDA
    assert "CUDA detected" in caplog.text


def test_resolve_device_auto_cpu_on_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FUELSENSE_DEVICE", raising=False)

    def _raise_import(_name: str) -> Any:
        raise ImportError("torch not available")

    monkeypatch.setattr("fuelsense_common.compute.importlib.import_module", _raise_import)
    assert resolve_device() == DeviceType.CPU


def test_compute_backend_protocol_runtime_check() -> None:
    class ValidBackend:
        device = DeviceType.CPU

        def warmup(self) -> None:
            return None

        def health_check(self) -> dict[str, object]:
            return {"status": "ok"}

    class InvalidBackend:
        pass

    assert isinstance(ValidBackend(), ComputeBackend)
    assert not isinstance(InvalidBackend(), ComputeBackend)
