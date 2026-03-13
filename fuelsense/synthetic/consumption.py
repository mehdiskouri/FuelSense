"""Synthetic consumption generation logic."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

LOW_TEMPERATURE_THRESHOLD = 15.0
HIGH_TEMPERATURE_THRESHOLD = 25.0


def _temperature_response(temperature: float, base_load: float) -> float:
    if temperature < LOW_TEMPERATURE_THRESHOLD:
        return ((LOW_TEMPERATURE_THRESHOLD - temperature) / LOW_TEMPERATURE_THRESHOLD) * 0.5 * base_load
    if temperature > HIGH_TEMPERATURE_THRESHOLD:
        return ((temperature - HIGH_TEMPERATURE_THRESHOLD) / LOW_TEMPERATURE_THRESHOLD) * 0.3 * base_load
    return 0.0


def generate_consumption(
    base_load: float,
    day: int,
    temperature: float,
    day_of_week: int,
    rng: np.random.Generator,
) -> float:
    """Generate daily facility consumption.

    C(t) = base + seasonal + temp_response + weekday_factor + noise
    """
    seasonal = 0.2 * base_load * math.sin(2.0 * math.pi * (day % 365) / 365.0)
    weekday_factor = 0.85 if day_of_week in (5, 6) else 1.0
    noise = rng.normal(0.0, 0.05 * base_load)
    raw = (base_load + seasonal + _temperature_response(temperature, base_load)) * weekday_factor + noise
    return max(raw, 0.0)
