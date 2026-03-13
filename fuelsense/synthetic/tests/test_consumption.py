"""Tests for synthetic consumption generation behavior."""

from __future__ import annotations

import numpy as np

from fuelsense.synthetic.consumption import generate_consumption


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_generate_consumption_weekend_factor_effect() -> None:
    """Weekend demand should be lower than weekday demand for same random seed."""
    rng = np.random.default_rng(42)
    weekday = generate_consumption(200, day=10, temperature=20, day_of_week=2, rng=rng)
    rng = np.random.default_rng(42)
    weekend = generate_consumption(200, day=10, temperature=20, day_of_week=6, rng=rng)
    _check(weekend < weekday, "Weekend demand should be below weekday demand for same seed")


def test_generate_consumption_non_negative() -> None:
    """Generated demand should never be negative."""
    rng = np.random.default_rng(1)
    value = generate_consumption(100, day=40, temperature=-5, day_of_week=1, rng=rng)
    _check(value >= 0, "Generated consumption should be non-negative")
