"""Benchmark utility for forecaster CPU and GPU backends."""

from __future__ import annotations

import argparse
import os
from time import perf_counter

import numpy as np

from fuelsense_common.compute import DeviceType
from fuelsense_common.registry import get_backend
import forecaster.backends.cpu_backend  # noqa: F401
import forecaster.backends.gpu_backend  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark forecaster backend")
    parser.add_argument("--facilities", type=int, default=50)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    os.environ["FUELSENSE_DEVICE"] = args.device
    device = DeviceType.CUDA if args.device == "cuda" else DeviceType.CPU

    rng = np.random.default_rng(42)
    n = max(args.facilities, 2)
    x = rng.normal(size=(n, 90, 6)).astype(np.float32)
    y = rng.normal(size=(n,)).astype(np.float32)

    backend = get_backend("demand_forecaster", device)
    backend.warmup()

    start = perf_counter()
    _ = backend.predict(x)
    predict_ms = (perf_counter() - start) * 1000

    train_start = perf_counter()
    result = backend.train(
        train_data=x[: n // 2],
        train_targets=y[: n // 2],
        val_data=x[n // 2 :],
        val_targets=y[n // 2 :],
        epochs=1,
        batch_size=min(32, n),
    )
    train_ms = (perf_counter() - train_start) * 1000

    print("backend,facilities,predict_ms,train_epoch_ms,val_rmse")
    print(f"{args.device},{n},{predict_ms:.2f},{train_ms:.2f},{float(result['validation_rmse']):.4f}")


if __name__ == "__main__":
    main()
