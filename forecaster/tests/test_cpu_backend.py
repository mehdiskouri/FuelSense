from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from forecaster.backends.cpu_backend import CPUForecaster


def _synthetic_data(n: int = 96) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(n, 90, 6)).astype(np.float32)
    y = (x[:, -1, 0] * 0.7 + x[:, -1, 1] * 0.2 + 0.1).astype(np.float32)
    return x, y


def test_cpu_backend_predict_shape() -> None:
    backend = CPUForecaster()
    x, _ = _synthetic_data(8)
    out = backend.predict(x)
    assert out.shape == (8, 14, 3)


def test_cpu_backend_health_check_keys() -> None:
    backend = CPUForecaster()
    health = backend.health_check()
    assert {"device", "threads", "mkl_available", "model_loaded"}.issubset(health.keys())


def test_cpu_backend_warmup_runs() -> None:
    backend = CPUForecaster()
    backend.warmup()


def test_cpu_backend_load_model_roundtrip(tmp_path: Path) -> None:
    backend = CPUForecaster()
    assert backend.model is not None
    model_path = tmp_path / "cpu_model.pt"
    torch.save(backend.model.state_dict(), model_path)
    backend.load_model(model_path)
    x, _ = _synthetic_data(4)
    out = backend.predict(x)
    assert out.shape == (4, 14, 3)


def test_cpu_backend_train_converges_smoke() -> None:
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
    assert len(history["val_loss"]) >= 2
    assert history["val_loss"][-1] <= history["val_loss"][0] or min(history["val_loss"]) < history["val_loss"][0]


def test_cpu_backend_train_accepts_horizon_targets() -> None:
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
    assert result["epochs_trained"] >= 1


def test_cpu_backend_respects_env_thread_and_worker_policy(monkeypatch) -> None:
    monkeypatch.setenv("FUELSENSE_FORECAST_CPU_THREADS", "2")
    monkeypatch.setenv("FUELSENSE_FORECAST_CPU_WORKERS", "3")
    backend = CPUForecaster()
    assert torch.get_num_threads() == 2
    assert backend.train_num_workers == 3
