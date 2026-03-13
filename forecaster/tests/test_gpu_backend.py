"""GPU backend tests for prediction, training, load, and NVML health branches."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
import torch

from forecaster.backends.gpu_backend import CUDAForecaster

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


def _synthetic_data(n: int = 16) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(n, 90, 6)).astype(np.float32)
    y = (x[:, -1, 0] * 0.5 + x[:, -1, 2] * 0.3).astype(np.float32)
    return x, y


def test_gpu_backend_predict_shape() -> None:
    """GPU backend prediction should return `(batch, horizon, quantiles)` output."""
    backend = CUDAForecaster()
    x, _ = _synthetic_data(8)
    out = backend.predict(x)
    _check(out.shape == (8, 14, 3))


def test_gpu_backend_health_fields() -> None:
    """GPU backend health payload should expose CUDA device metadata."""
    backend = CUDAForecaster()
    health = backend.health_check()
    _check(health["device"] == "cuda")
    _check("gpu_name" in health)


def test_gpu_backend_train_smoke() -> None:
    """GPU training smoke test should complete and report at least one epoch."""
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
    _check(result["epochs_trained"] >= 1)


def test_gpu_backend_train_accepts_horizon_targets() -> None:
    """GPU backend should train successfully on horizon-shaped targets."""
    backend = CUDAForecaster()
    x, _ = _synthetic_data(36)
    rng = np.random.default_rng(123)
    y = rng.normal(size=(36, 14)).astype(np.float32)
    result = backend.train(
        train_data=x[:28],
        train_targets=y[:28],
        val_data=x[28:],
        val_targets=y[28:],
        epochs=2,
        batch_size=8,
        patience=2,
    )
    _check(result["epochs_trained"] >= 1)


def test_gpu_backend_init_raises_without_cuda() -> None:
    """CUDA backend init should fail when CUDA availability is forced false."""
    with (
        patch("forecaster.backends.gpu_backend.torch.cuda.is_available", return_value=False),
        pytest.raises(RuntimeError),
    ):
        CUDAForecaster()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_backend_predict_raises_if_model_none() -> None:
    """Predict should raise when model is missing after backend initialization."""
    backend = CUDAForecaster()
    backend.model = None
    x, _ = _synthetic_data(2)
    with pytest.raises(RuntimeError):
        backend.predict(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_backend_load_model_compile_fallback(tmp_path: Path) -> None:
    """Model load should continue when optional `torch.compile` raises."""
    backend = CUDAForecaster()
    model = backend.model
    if model is None:
        msg = "Backend model should be initialized"
        raise AssertionError(msg)
    state_path = tmp_path / "gpu_state.pt"
    torch.save(model.state_dict(), state_path)

    def _compile_fail(*args: object, **kwargs: object) -> object:
        _ = args, kwargs
        msg = "compile fail"
        raise RuntimeError(msg)

    with patch("forecaster.backends.gpu_backend.torch.compile", side_effect=_compile_fail):
        backend.load_model(state_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_backend_health_check_nvml_failure() -> None:
    """Health check should fall back gracefully when NVML bindings are incomplete."""
    backend = CUDAForecaster()
    with patch.dict(sys.modules, {"pynvml": ModuleType("pynvml")}, clear=False):
        health = backend.health_check()
    _check("gpu_utilization" in health)
