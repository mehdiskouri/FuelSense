"""Tests for device resolution logic and compute backend runtime protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

from fuelsense_common.compute import ComputeBackend, DeviceType, resolve_device

if TYPE_CHECKING:
    import pytest


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


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
    """Explicit CPU environment setting should always resolve to CPU device."""
    monkeypatch.setenv("FUELSENSE_DEVICE", "cpu")
    _check(resolve_device() == DeviceType.CPU)


def test_resolve_device_explicit_cuda_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit CUDA setting should resolve to CUDA when CUDA is available."""
    def _import_torch(_name: str) -> _TorchCudaAvailable:
        return _TorchCudaAvailable()

    monkeypatch.setenv("FUELSENSE_DEVICE", "cuda")
    monkeypatch.setattr("fuelsense_common.compute.importlib.import_module", _import_torch)
    _check(resolve_device() == DeviceType.CUDA)


def test_resolve_device_explicit_cuda_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicit CUDA setting should fall back to CPU when CUDA is unavailable."""
    def _import_torch(_name: str) -> _TorchCudaUnavailable:
        return _TorchCudaUnavailable()

    monkeypatch.setenv("FUELSENSE_DEVICE", "cuda")
    monkeypatch.setattr("fuelsense_common.compute.importlib.import_module", _import_torch)
    result = resolve_device()
    _check(result == DeviceType.CPU)
    _check("Falling back to CPU" in caplog.text)


def test_resolve_device_auto_cuda(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Auto mode should choose CUDA when import succeeds and CUDA is available."""
    def _import_torch(_name: str) -> _TorchCudaAvailable:
        return _TorchCudaAvailable()

    monkeypatch.delenv("FUELSENSE_DEVICE", raising=False)
    monkeypatch.setattr("fuelsense_common.compute.importlib.import_module", _import_torch)
    result = resolve_device()
    _check(result == DeviceType.CUDA)
    _check("CUDA detected" in caplog.text)


def test_resolve_device_auto_cpu_on_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto mode should fall back to CPU when torch import fails."""
    monkeypatch.delenv("FUELSENSE_DEVICE", raising=False)

    def _raise_import(_name: str) -> object:
        msg = "torch not available"
        raise ImportError(msg)

    monkeypatch.setattr("fuelsense_common.compute.importlib.import_module", _raise_import)
    _check(resolve_device() == DeviceType.CPU)


def test_compute_backend_protocol_runtime_check() -> None:
    """Runtime protocol checks should accept valid backends and reject invalid ones."""
    class ValidBackend:
        device = DeviceType.CPU

        def warmup(self) -> None:
            return None

        def health_check(self) -> dict[str, object]:
            return {"status": "ok"}

    class InvalidBackend:
        pass

    valid_backend: object = ValidBackend()
    invalid_backend: object = InvalidBackend()
    _check(isinstance(valid_backend, ComputeBackend))  # pyright: ignore[reportUnnecessaryIsInstance]
    _check(not isinstance(invalid_backend, ComputeBackend))
