from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from optimizer import benchmark


def test_optimizer_benchmark_cpu_50_stops_fast_and_improving() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["FUELSENSE_DEVICE"] = "cpu"
    env["FUELSENSE_OPTIMIZER_TIME_LIMIT_MS"] = "9000"

    result = subprocess.run(
        [sys.executable, "-m", "optimizer.benchmark", "--stops", "50", "--device", "cpu"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) >= 2
    values = lines[-1].split(",")
    assert len(values) == 8
    cost_reduction_pct = float(values[6])
    elapsed_ms = float(values[7])

    assert elapsed_ms < 10000.0
    assert cost_reduction_pct > 0.0


def test_benchmark_main_runs_with_mock_backend(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _Backend:
        def warmup(self) -> None:
            return None

        def solve(self, **kwargs: object) -> dict[str, object]:
            _ = kwargs
            return {
                "status": "optimal",
                "vehicles_used": 2,
                "total_cost": 100.0,
                "baseline_cost": 200.0,
                "cost_reduction_pct": 50.0,
            }

    monkeypatch.setattr("optimizer.benchmark.get_backend", lambda name, device: _Backend())
    monkeypatch.setattr("sys.argv", ["benchmark", "--stops", "10", "--device", "cpu"])

    benchmark.main()
    output = capsys.readouterr().out
    assert "backend,status,stops" in output
    assert "cpu,optimal,10" in output


def test_benchmark_main_json_output_with_mock_backend(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _Backend:
        def warmup(self) -> None:
            return None

        def solve(self, **kwargs: object) -> dict[str, object]:
            _ = kwargs
            return {
                "status": "optimal",
                "vehicles_used": 1,
                "total_cost": 100.0,
                "baseline_cost": 200.0,
                "cost_reduction_pct": 50.0,
            }

    monkeypatch.setattr("optimizer.benchmark.get_backend", lambda name, device: _Backend())
    monkeypatch.setattr("sys.argv", ["benchmark", "--stops", "10", "--device", "cpu", "--output", "json"])

    benchmark.main()
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["backend"] == "cpu"
    assert payload["status"] == "optimal"
    assert payload["stops"] == 10
