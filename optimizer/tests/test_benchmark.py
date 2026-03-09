from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


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
