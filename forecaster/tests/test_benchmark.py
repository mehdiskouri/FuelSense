from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from forecaster import benchmark


def test_forecaster_benchmark_cpu_50_facilities_runs() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["FUELSENSE_DEVICE"] = "cpu"

    result = subprocess.run(
        [sys.executable, "-m", "forecaster.benchmark", "--facilities", "50", "--device", "cpu"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) >= 2
    assert lines[0] == "backend,facilities,predict_ms,train_epoch_ms,val_rmse"
    values = lines[-1].split(",")
    assert len(values) == 5
    assert values[0] == "cpu"
    assert int(values[1]) == 50
    assert float(values[2]) >= 0.0
    assert float(values[3]) >= 0.0
    assert float(values[4]) >= 0.0


def test_benchmark_main_json_output_with_mock_backend(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _Backend:
        def warmup(self) -> None:
            return None

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.zeros((x.shape[0], 14, 3), dtype=np.float32)

        def train(self, **kwargs: Any) -> dict[str, Any]:
            _ = kwargs
            return {"validation_rmse": 0.42}

    def _mock_get_backend(name: str, device: object) -> _Backend:
        _ = (name, device)
        return _Backend()

    monkeypatch.setattr("forecaster.benchmark.get_backend", cast(Any, _mock_get_backend))
    monkeypatch.setattr(
        "sys.argv",
        ["benchmark", "--facilities", "10", "--device", "cpu", "--output", "json"],
    )

    benchmark.main()
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["backend"] == "cpu"
    assert payload["facilities"] == 10
    assert payload["val_rmse"] == 0.42
