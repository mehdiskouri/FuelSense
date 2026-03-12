from __future__ import annotations

import numpy as np
import pytest
import torch

from forecaster.backends.cpu_backend import CPUForecaster
from forecaster.backends.gpu_backend import CUDAForecaster

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _synthetic_data(n: int = 12) -> np.ndarray:
    rng = np.random.default_rng(2026)
    return rng.normal(size=(n, 90, 6)).astype(np.float32)


def test_cpu_gpu_predict_parity_with_shared_weights() -> None:
    cpu_backend = CPUForecaster()
    assert cpu_backend.model is not None

    gpu_backend = CUDAForecaster()
    assert gpu_backend.model is not None

    # Align weights so inference parity is meaningful.
    gpu_backend.model.load_state_dict(cpu_backend.model.state_dict())
    gpu_backend.model.eval()

    x = _synthetic_data(8)
    cpu_pred = cpu_backend.predict(x)
    gpu_pred = gpu_backend.predict(x)

    assert cpu_pred.shape == gpu_pred.shape == (8, 14, 3)
    assert np.allclose(cpu_pred, gpu_pred, atol=1e-3, rtol=1e-3)
