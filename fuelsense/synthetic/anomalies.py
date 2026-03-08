"""Synthetic anomaly injection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class AnomalyEvent:
	anomaly_type: str
	start_day: int
	duration: int
	magnitude: float


class AnomalyInjector:
	"""Inject random anomalies into daily consumption traces."""

	anomaly_types = [
		"LEAK",
		"THEFT",
		"EQUIPMENT_DEGRADATION",
		"DEMAND_SHIFT",
		"SENSOR_FAULT",
	]

	def __init__(self, trigger_probability: float = 0.05) -> None:
		self.trigger_probability = trigger_probability
		self._active: Dict[str, AnomalyEvent] = {}
		self._shift_multiplier: Dict[str, float] = {}

	def _sample_event(self, day: int, rng: np.random.Generator) -> AnomalyEvent:
		anomaly_type = str(rng.choice(self.anomaly_types))
		if anomaly_type == "LEAK":
			return AnomalyEvent(anomaly_type, day, int(rng.integers(3, 8)), float(rng.uniform(0.15, 0.30)))
		if anomaly_type == "THEFT":
			return AnomalyEvent(anomaly_type, day, 1, float(rng.uniform(0.20, 0.50)))
		if anomaly_type == "EQUIPMENT_DEGRADATION":
			return AnomalyEvent(anomaly_type, day, int(rng.integers(10, 21)), 0.02)
		if anomaly_type == "DEMAND_SHIFT":
			return AnomalyEvent(anomaly_type, day, 1, float(rng.uniform(-0.15, 0.15)))
		return AnomalyEvent(anomaly_type, day, int(rng.integers(1, 4)), float(rng.uniform(0.0, 1.0)))

	def apply(
		self,
		facility_key: str,
		day: int,
		consumption: float,
		base_load: float,
		rng: np.random.Generator,
	) -> tuple[float, AnomalyEvent | None]:
		"""Apply anomaly process and return modified consumption plus optional label."""
		if facility_key not in self._shift_multiplier:
			self._shift_multiplier[facility_key] = 1.0

		output = consumption * self._shift_multiplier[facility_key]
		active = self._active.get(facility_key)

		if active and day >= active.start_day + active.duration:
			self._active.pop(facility_key, None)
			active = None

		if active is None and rng.random() < self.trigger_probability:
			active = self._sample_event(day, rng)
			if active.anomaly_type == "DEMAND_SHIFT":
				self._shift_multiplier[facility_key] *= 1.0 + active.magnitude
			else:
				self._active[facility_key] = active

		if active is None:
			return output, None

		if active.anomaly_type == "LEAK":
			return output * (1.0 + active.magnitude), active
		if active.anomaly_type == "THEFT":
			return output * (1.0 + active.magnitude), active
		if active.anomaly_type == "EQUIPMENT_DEGRADATION":
			day_offset = max(day - active.start_day + 1, 1)
			return output * (1.0 + active.magnitude * day_offset), active
		if active.anomaly_type == "SENSOR_FAULT":
			return (0.0 if rng.random() < 0.5 else float("nan")), active

		# Demand shift label for its trigger day.
		return output, active
