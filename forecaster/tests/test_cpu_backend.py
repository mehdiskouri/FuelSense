"""CPU backend tests for prediction shape, training behavior, and thread config."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from forecaster.backends.cpu_backend import CPUForecaster

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


MIN_HISTORY_POINTS = 2
CPU_THREADS = 2
CPU_WORKERS = 3


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


def _synthetic_data(n: int = 96) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(n, 90, 6)).astype(np.float32)
    y = (x[:, -1, 0] * 0.7 + x[:, -1, 1] * 0.2 + 0.1).astype(np.float32)
    return x, y


def test_cpu_backend_predict_shape() -> None:
    """CPU backend prediction should return `(batch, horizon, quantiles)` output."""
    backend = CPUForecaster()
    x, _ = _synthetic_data(8)
    out = backend.predict(x)
    _check(out.shape == (8, 14, 3))


def test_cpu_backend_health_check_keys() -> None:
    """CPU backend health payload should contain required metadata keys."""
    backend = CPUForecaster()
    health = backend.health_check()
    _check({"device", "threads", "mkl_available", "model_loaded"}.issubset(health.keys()))


def test_cpu_backend_warmup_runs() -> None:
    """CPU backend warmup should execute without raising errors."""
    backend = CPUForecaster()
    backend.warmup()


def test_cpu_backend_load_model_roundtrip(tmp_path: Path) -> None:
    """Saving and reloading model weights should preserve prediction usability."""
    backend = CPUForecaster()
    model = backend.model
    _check(model is not None)
    model_path = tmp_path / "cpu_model.pt"
    torch.save(model.state_dict(), model_path)
    backend.load_model(model_path)
    x, _ = _synthetic_data(4)
    out = backend.predict(x)
    _check(out.shape == (4, 14, 3))


def test_cpu_backend_train_converges_smoke() -> None:
    """Training smoke test should report non-empty validation history."""
    backend = CPUForecaster()
    x, y = _synthetic_data(128)
    result = backend.train(
        train_data=x[:96],
        train_targets=y[:96],
        val_data=x[96:],
        val_targets=y[96:],
        epochs=25,
        batch_size=16,
        patience=25,
    )
    history = result["history"]
    _check(len(history["val_loss"]) >= MIN_HISTORY_POINTS)
    _check(history["val_loss"][-1] <= history["val_loss"][0] or min(history["val_loss"]) < history["val_loss"][0])


def test_cpu_backend_train_accepts_horizon_targets() -> None:
    """CPU backend training should accept horizon-shaped target matrices."""
    backend = CPUForecaster()
    x, _ = _synthetic_data(96)
    rng = np.random.default_rng(99)
    y = rng.normal(size=(96, 14)).astype(np.float32)
    result = backend.train(
        train_data=x[:72],
        train_targets=y[:72],
        val_data=x[72:],
        val_targets=y[72:],
        epochs=2,
        batch_size=16,
        patience=2,
    )
    _check(result["epochs_trained"] >= 1)


def test_cpu_backend_respects_env_thread_and_worker_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment thread/worker settings should be applied at backend init time."""
    monkeypatch.setenv("FUELSENSE_FORECAST_CPU_THREADS", "2")
    monkeypatch.setenv("FUELSENSE_FORECAST_CPU_WORKERS", "3")
    backend = CPUForecaster()
    _check(torch.get_num_threads() == CPU_THREADS)
    _check(backend.train_num_workers == CPU_WORKERS)
