from __future__ import annotations

import numpy as np

from fuelsense.synthetic.anomalies import AnomalyEvent, AnomalyInjector


def test_anomaly_branch_coverage():
    rng = np.random.default_rng(42)
    injector = AnomalyInjector(trigger_probability=0.0)

    for anomaly_type in ["LEAK", "THEFT", "EQUIPMENT_DEGRADATION", "SENSOR_FAULT", "DEMAND_SHIFT"]:
        event = AnomalyEvent(anomaly_type=anomaly_type, start_day=0, duration=3, magnitude=0.2)
        injector._active["f"] = event
        value, _ = injector.apply("f", day=1, consumption=100.0, base_load=100.0, rng=rng)
        if anomaly_type == "SENSOR_FAULT":
            assert value == 0.0 or np.isnan(value)
        else:
            assert isinstance(value, float)
