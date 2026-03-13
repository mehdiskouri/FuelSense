"""Tests for anomaly injector event triggering behavior."""

from __future__ import annotations

import numpy as np

from fuelsense.synthetic.anomalies import AnomalyInjector


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_anomaly_injector_emits_event_with_high_probability() -> None:
    """High trigger probability should create an anomaly event."""
    rng = np.random.default_rng(42)
    injector = AnomalyInjector(trigger_probability=1.0)
    _, event = injector.apply("f1", day=0, consumption=100, base_load=100, rng=rng)
    _check(event is not None, "High trigger probability should emit an anomaly event")
    _check(event.anomaly_type in injector.anomaly_types, "Event type should be from configured anomaly types")
