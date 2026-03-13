"""Benchmark utility for forecaster CPU and GPU backends."""

from __future__ import annotations

import argparse
import json
import os
from time import perf_counter
from typing import Protocol, cast

import numpy as np
import torch

from forecaster.backends import cpu_backend as _cpu_backend
from forecaster.backends import gpu_backend as _gpu_backend
from fuelsense_common.compute import DeviceType
from fuelsense_common.registry import get_backend


class ForecasterBackend(Protocol):
    """Minimal benchmark-facing contract implemented by forecaster backends."""

    def warmup(self) -> None:
        """Prime backend runtime before measuring inference and training."""
        ...

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run batched inference for benchmark inputs."""
        ...

    def train(self, **kwargs: object) -> dict[str, object]:
        """Train backend model using benchmark-provided keyword arguments."""
        ...


def main() -> None:
    """Run a small synthetic benchmark for the selected forecaster backend."""
    parser = argparse.ArgumentParser(description="Benchmark forecaster backend")
    parser.add_argument("--facilities", type=int, default=50)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", choices=["csv", "json"], default="csv")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    _ = (_cpu_backend, _gpu_backend)

    os.environ["FUELSENSE_DEVICE"] = args.device
    device = DeviceType.CUDA if args.device == "cuda" else DeviceType.CPU

    rng = np.random.default_rng(42)
    n = max(args.facilities, 2)
    x = rng.normal(size=(n, 90, 6)).astype(np.float32)
    y = rng.normal(size=(n, 14)).astype(np.float32)

    backend = cast("ForecasterBackend", get_backend("demand_forecaster", device))
    backend.warmup()

    if device == DeviceType.CUDA:
        torch.cuda.reset_peak_memory_stats()

    start = perf_counter()
    _ = backend.predict(x)
    predict_ms = (perf_counter() - start) * 1000

    train_start = perf_counter()
    result = backend.train(
        train_data=x[: n // 2],
        train_targets=y[: n // 2],
        val_data=x[n // 2 :],
        val_targets=y[n // 2 :],
        epochs=max(args.epochs, 1),
        batch_size=min(256 if device == DeviceType.CUDA else 32, n),
    )
    train_ms = (perf_counter() - train_start) * 1000
    val_rmse_raw = result.get("validation_rmse", 0.0)
    val_rmse = float(val_rmse_raw) if isinstance(val_rmse_raw, (int, float, str)) else 0.0

    payload: dict[str, float | int | str] = {
        "backend": args.device,
        "facilities": n,
        "predict_ms": round(predict_ms, 2),
        "train_epoch_ms": round(train_ms, 2),
        "val_rmse": round(val_rmse, 4),
    }

    if device == DeviceType.CUDA:
        payload["peak_gpu_mem_gb"] = round(float(torch.cuda.max_memory_allocated() / (1024**3)), 4)

    if args.output == "json":
        print(json.dumps(payload))  # noqa: T201
        return

    print("backend,facilities,predict_ms,train_epoch_ms,val_rmse")  # noqa: T201
    print(  # noqa: T201
        f"{payload['backend']},{payload['facilities']},{payload['predict_ms']},{payload['train_epoch_ms']},"
        f"{payload['val_rmse']}",
    )


if __name__ == "__main__":
    main()
