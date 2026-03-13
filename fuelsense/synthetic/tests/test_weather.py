"""Tests for synthetic weather feature generation."""

from __future__ import annotations

import numpy as np

from fuelsense.synthetic.weather import generate_weather_series


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_weather_shape_and_ranges() -> None:
    """Generated weather arrays should have expected shape and non-negative fields."""
    rng = np.random.default_rng(42)
    arr = generate_weather_series(30, latitude=24.7, rng=rng)
    _check(arr.shape == (30, 3), "Weather series should return day x feature matrix")
    _check((arr[:, 1] >= 0).all(), "Precipitation feature should be non-negative")
    _check((arr[:, 2] >= 0).all(), "Wind speed feature should be non-negative")
