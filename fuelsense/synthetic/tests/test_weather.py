from __future__ import annotations

import numpy as np

from fuelsense.synthetic.weather import generate_weather_series


def test_weather_shape_and_ranges():
    rng = np.random.default_rng(42)
    arr = generate_weather_series(30, latitude=24.7, rng=rng)
    assert arr.shape == (30, 3)
    assert (arr[:, 1] >= 0).all()
    assert (arr[:, 2] >= 0).all()
