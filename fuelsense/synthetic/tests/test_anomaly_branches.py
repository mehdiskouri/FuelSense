"""Branch coverage tests for synthetic anomaly application logic."""

from __future__ import annotations

import numpy as np

from fuelsense.synthetic.anomalies import AnomalyEvent, AnomalyInjector

ANOMALY_TYPES = ["LEAK", "THEFT", "EQUIPMENT_DEGRADATION", "SENSOR_FAULT", "DEMAND_SHIFT"]


class _DeterministicRng:
    def __init__(self, anomaly_type: str) -> None:
        self._anomaly_type = anomaly_type

    def random(self) -> float:
        return 0.0

    def choice(self, values: list[str]) -> str:
        _ = values
        return self._anomaly_type

    def integers(self, low: int, high: int) -> int:
        _ = high
        return low

    def uniform(self, low: float, high: float) -> float:
        _ = high
        return low


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_anomaly_branch_coverage() -> None:
    """Each anomaly type should execute its corresponding apply-branch logic."""
    for anomaly_type in ANOMALY_TYPES:
        injector = AnomalyInjector(trigger_probability=1.0)
        rng = _DeterministicRng(anomaly_type)
        value, event = injector.apply("f", day=1, consumption=100.0, base_load=100.0, rng=rng)
        _check(isinstance(event, AnomalyEvent), "Expected an anomaly event for forced trigger")
        _check(event is not None and event.anomaly_type == anomaly_type, "Injected anomaly type should match")
        if anomaly_type == "SENSOR_FAULT":
            _check(value == 0.0 or np.isnan(value), "Sensor fault branch should output 0.0 or NaN")
        else:
            _check(isinstance(value, float), "Non-sensor branches should yield float outputs")
