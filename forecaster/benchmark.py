"""Benchmark utility for forecaster CPU and GPU backends."""

from __future__ import annotations

import argparse
import json
import os
from time import perf_counter
from typing import Any, Protocol, cast

import numpy as np

from fuelsense_common.compute import DeviceType
from fuelsense_common.registry import get_backend


class ForecasterBackend(Protocol):
    def warmup(self) -> None: ...

    def predict(self, x: np.ndarray) -> np.ndarray: ...

    def train(
        self,
        *,
        train_data: np.ndarray,
        train_targets: np.ndarray,
        val_data: np.ndarray,
        val_targets: np.ndarray,
        epochs: int,
        batch_size: int,
    ) -> dict[str, Any]: ...


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark forecaster backend")
    parser.add_argument("--facilities", type=int, default=50)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", choices=["csv", "json"], default="csv")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    # Import backend modules lazily to ensure registry side effects are applied.
    from forecaster.backends import cpu_backend as _cpu_backend
    from forecaster.backends import gpu_backend as _gpu_backend

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
        import torch

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

    payload: dict[str, float | int | str] = {
        "backend": args.device,
        "facilities": n,
        "predict_ms": round(predict_ms, 2),
        "train_epoch_ms": round(train_ms, 2),
        "val_rmse": round(float(result.get("validation_rmse", 0.0)), 4),
    }

    if device == DeviceType.CUDA:
        import torch

        payload["peak_gpu_mem_gb"] = round(float(torch.cuda.max_memory_allocated() / (1024**3)), 4)

    if args.output == "json":
        print(json.dumps(payload, separators=(",", ":")))
        return

    print("backend,facilities,predict_ms,train_epoch_ms,val_rmse")
    print(
        f"{payload['backend']},{payload['facilities']},{payload['predict_ms']:.2f},"
        f"{payload['train_epoch_ms']:.2f},{payload['val_rmse']:.4f}",
    )


if __name__ == "__main__":
    main()
