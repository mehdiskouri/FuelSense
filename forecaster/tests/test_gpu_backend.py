from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from forecaster.backends.gpu_backend import CUDAForecaster


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _synthetic_data(n: int = 16) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(n, 90, 6)).astype(np.float32)
    y = (x[:, -1, 0] * 0.5 + x[:, -1, 2] * 0.3).astype(np.float32)
    return x, y


def test_gpu_backend_predict_shape() -> None:
    backend = CUDAForecaster()
    x, _ = _synthetic_data(8)
    out = backend.predict(x)
    assert out.shape == (8, 14, 3)


def test_gpu_backend_health_fields() -> None:
    backend = CUDAForecaster()
    health = backend.health_check()
    assert health["device"] == "cuda"
    assert "gpu_name" in health


def test_gpu_backend_train_smoke() -> None:
    backend = CUDAForecaster()
    x, y = _synthetic_data(40)
    result = backend.train(
        train_data=x[:30],
        train_targets=y[:30],
        val_data=x[30:],
        val_targets=y[30:],
        epochs=3,
        batch_size=8,
        patience=3,
    )
    assert result["epochs_trained"] >= 1


def test_gpu_backend_init_raises_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("forecaster.backends.gpu_backend.torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError):
        CUDAForecaster()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_backend_predict_raises_if_model_none() -> None:
    backend = CUDAForecaster()
    backend.model = None
    x, _ = _synthetic_data(2)
    with pytest.raises(RuntimeError):
        backend.predict(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_backend_load_model_compile_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    backend = CUDAForecaster()
    assert backend.model is not None
    state_path = tmp_path / "gpu_state.pt"
    torch.save(backend.model.state_dict(), state_path)

    def _compile_fail(*args: object, **kwargs: object) -> object:
        _ = args, kwargs
        raise RuntimeError("compile fail")

    monkeypatch.setattr("forecaster.backends.gpu_backend.torch.compile", _compile_fail)
    backend.load_model(state_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_backend_health_check_nvml_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = CUDAForecaster()
    monkeypatch.setitem(sys.modules, "pynvml", ModuleType("pynvml"))
    # Missing required NVML attrs should drive fallback branch.
    health = backend.health_check()
    assert "gpu_utilization" in health
