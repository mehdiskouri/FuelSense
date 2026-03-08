from __future__ import annotations

import numpy as np

from fuelsense.synthetic.consumption import generate_consumption


def test_generate_consumption_weekend_factor_effect():
    rng = np.random.default_rng(42)
    weekday = generate_consumption(200, day=10, temperature=20, day_of_week=2, rng=rng)
    rng = np.random.default_rng(42)
    weekend = generate_consumption(200, day=10, temperature=20, day_of_week=6, rng=rng)
    assert weekend < weekday


def test_generate_consumption_non_negative():
    rng = np.random.default_rng(1)
    value = generate_consumption(100, day=40, temperature=-5, day_of_week=1, rng=rng)
    assert value >= 0
