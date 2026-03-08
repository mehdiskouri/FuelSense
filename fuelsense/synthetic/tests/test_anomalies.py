from __future__ import annotations

import numpy as np

from fuelsense.synthetic.anomalies import AnomalyInjector


def test_anomaly_injector_emits_event_with_high_probability():
    rng = np.random.default_rng(42)
    injector = AnomalyInjector(trigger_probability=1.0)
    _, event = injector.apply("f1", day=0, consumption=100, base_load=100, rng=rng)
    assert event is not None
    assert event.anomaly_type in injector.anomaly_types
