"""Synthetic weather generation logic."""

from __future__ import annotations

import math

import numpy as np


def generate_weather_series(days: int, latitude: float, rng: np.random.Generator) -> np.ndarray:
	"""Generate [days, 3] weather series: temperature, wind_speed, solar_irradiance."""
	temperature = np.zeros(days, dtype=float)
	wind_speed = np.zeros(days, dtype=float)
	solar = np.zeros(days, dtype=float)

	lat_norm = min(max(abs(latitude) / 90.0, 0.0), 1.0)
	t_base = 22.0 - 10.0 * lat_norm
	t_amp = 8.0 + 6.0 * lat_norm
	w_base = 3.0 + 4.0 * lat_norm

	temperature[0] = t_base + rng.normal(0.0, 2.0)
	wind_speed[0] = max(0.0, w_base + rng.normal(0.0, 1.0))

	for day in range(days):
		seasonal_temp = t_base + t_amp * math.sin(2.0 * math.pi * (day % 365) / 365.0)
		if day > 0:
			temperature[day] = seasonal_temp + 0.85 * (temperature[day - 1] - seasonal_temp) + rng.normal(0.0, 1.5)
			wind_speed[day] = max(0.0, w_base + 0.6 * (wind_speed[day - 1] - w_base) + rng.normal(0.0, 0.8))

		day_of_year = day % 365
		day_length = max(0.0, math.sin(2.0 * math.pi * (day_of_year - 80) / 365.0))
		cloud_factor = 1.0 - min(max(rng.normal(0.2, 0.2), 0.0), 0.6)
		solar[day] = max(0.0, 900.0 * day_length * cloud_factor)

	return np.column_stack([temperature, wind_speed, solar])
