"""Benchmark entrypoint tests for optimizer CLI output and runtime behavior."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from optimizer import benchmark

if TYPE_CHECKING:
    import pytest


MIN_OUTPUT_LINES = 2
CSV_FIELDS = 8
CLI_STOPS = 10
ELAPSED_LIMIT_MS = 10000.0


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


def test_optimizer_benchmark_cpu_50_stops_fast_and_improving() -> None:
    """CLI benchmark should emit valid CSV and show positive improvement quickly."""
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["FUELSENSE_DEVICE"] = "cpu"
    env["FUELSENSE_OPTIMIZER_TIME_LIMIT_MS"] = "9000"

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "optimizer.benchmark", "--stops", "50", "--device", "cpu"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    _check(len(lines) >= MIN_OUTPUT_LINES)
    values = lines[-1].split(",")
    _check(len(values) == CSV_FIELDS)
    cost_reduction_pct = float(values[6])
    elapsed_ms = float(values[7])

    _check(elapsed_ms < ELAPSED_LIMIT_MS)
    _check(cost_reduction_pct > 0.0)


def test_benchmark_main_runs_with_mock_backend(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Benchmark main should print CSV line when using a mock backend."""

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

    def _get_backend(_name: str, _device: object) -> _Backend:
        _ = _name, _device
        return _Backend()

    monkeypatch.setattr("optimizer.benchmark.get_backend", _get_backend)
    monkeypatch.setattr("sys.argv", ["benchmark", "--stops", "10", "--device", "cpu"])

    benchmark.main()
    output = capsys.readouterr().out
    _check("backend,status,stops" in output)
    _check("cpu,optimal,10" in output)


def test_benchmark_main_json_output_with_mock_backend(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Benchmark main should emit parseable JSON in JSON output mode."""

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

    def _get_backend(_name: str, _device: object) -> _Backend:
        _ = _name, _device
        return _Backend()

    monkeypatch.setattr("optimizer.benchmark.get_backend", _get_backend)
    monkeypatch.setattr("sys.argv", ["benchmark", "--stops", "10", "--device", "cpu", "--output", "json"])

    benchmark.main()
    payload = json.loads(capsys.readouterr().out.strip())
    _check(payload["backend"] == "cpu")
    _check(payload["status"] == "optimal")
    _check(payload["stops"] == CLI_STOPS)
