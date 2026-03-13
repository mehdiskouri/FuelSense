"""CLI and JSON-output tests for the forecaster benchmark entrypoint."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from forecaster import benchmark

if TYPE_CHECKING:
    import pytest


CSV_FIELDS = 5
FACILITY_COUNT_50 = 50
FACILITY_COUNT_10 = 10
SUBPROCESS_TIMEOUT_SECONDS = 30
EXPECTED_RMSE = 0.42
MIN_OUTPUT_LINES = 2


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


def test_forecaster_benchmark_cpu_50_facilities_runs() -> None:
    """CLI benchmark should emit CSV with expected shape and non-negative metrics."""
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["FUELSENSE_DEVICE"] = "cpu"

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "forecaster.benchmark", "--facilities", "50", "--device", "cpu"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    _check(len(lines) >= MIN_OUTPUT_LINES)
    _check(lines[0] == "backend,facilities,predict_ms,train_epoch_ms,val_rmse")
    values = lines[-1].split(",")
    _check(len(values) == CSV_FIELDS)
    _check(values[0] == "cpu")
    _check(int(values[1]) == FACILITY_COUNT_50)
    _check(float(values[2]) >= 0.0)
    _check(float(values[3]) >= 0.0)
    _check(float(values[4]) >= 0.0)


def test_benchmark_main_json_output_with_mock_backend(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """JSON benchmark output should include backend, facility count, and RMSE."""
    class _Backend:
        def warmup(self) -> None:
            return None

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.zeros((x.shape[0], 14, 3), dtype=np.float32)

        def train(self, **kwargs: object) -> dict[str, float]:
            _ = kwargs
            return {"validation_rmse": EXPECTED_RMSE}

    def _mock_get_backend(name: str, device: object) -> _Backend:
        _ = (name, device)
        return _Backend()

    monkeypatch.setattr("forecaster.benchmark.get_backend", _mock_get_backend)
    monkeypatch.setattr(
        "sys.argv",
        ["benchmark", "--facilities", "10", "--device", "cpu", "--output", "json"],
    )

    benchmark.main()
    payload = json.loads(capsys.readouterr().out.strip())
    _check(payload["backend"] == "cpu")
    _check(payload["facilities"] == FACILITY_COUNT_10)
    _check(payload["val_rmse"] == EXPECTED_RMSE)
