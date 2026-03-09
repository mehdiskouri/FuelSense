"""Two-stage anomaly detector implementation."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class AnomalyType(str, Enum):
    LEAK = "LEAK"
    THEFT = "THEFT"
    EQUIPMENT_DEGRADATION = "EQUIPMENT_DEGRADATION"
    DEMAND_SHIFT = "DEMAND_SHIFT"
    SENSOR_FAULT = "SENSOR_FAULT"
    UNKNOWN = "UNKNOWN"


class AnomalyDetector:
    Z_THRESHOLD = 3.0
    N_ESTIMATORS = 200
    CONTAMINATION = 0.05
    feature_names = [
        "z_score",
        "z_score_rolling_3d",
        "consumption_delta_pct",
        "temperature_residual",
        "day_of_week",
        "hours_since_delivery",
        "inventory_level_pct",
    ]
    _TYPE_BY_LABEL = {
        0: AnomalyType.LEAK,
        1: AnomalyType.THEFT,
        2: AnomalyType.EQUIPMENT_DEGRADATION,
        3: AnomalyType.DEMAND_SHIFT,
        4: AnomalyType.SENSOR_FAULT,
    }

    def __init__(self) -> None:
        self.forest: Any | None = None
        self.classifier: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self.forest is not None and self.classifier is not None

    def load(self, forest_path: str | Path, classifier_path: str | Path) -> None:
        self.forest = joblib.load(Path(forest_path))
        self.classifier = joblib.load(Path(classifier_path))

    def detect(self, actual: float, predicted: float, rolling_std: float, features: dict[str, float]) -> dict[str, Any]:
        std = max(abs(float(rolling_std)), 1e-6)
        z_score = (float(actual) - float(predicted)) / std

        if abs(z_score) < self.Z_THRESHOLD:
            return {
                "is_anomaly": False,
                "z_score": float(z_score),
                "anomaly_type": None,
                "confidence": 0.0,
                "if_score": None,
                "stage": 1,
            }

        vector = np.asarray(
            [[self._feature_value(name, z_score, features) for name in self.feature_names]], dtype=np.float32
        )

        if self.forest is None or self.classifier is None:
            return {
                "is_anomaly": False,
                "z_score": float(z_score),
                "anomaly_type": AnomalyType.UNKNOWN.value,
                "confidence": 0.4,
                "if_score": None,
                "stage": 2,
            }

        if_score = float(np.asarray(self.forest.decision_function(vector)).reshape(-1)[0])
        if_pred = int(np.asarray(self.forest.predict(vector)).reshape(-1)[0])

        if if_pred == 1:
            return {
                "is_anomaly": False,
                "z_score": float(z_score),
                "anomaly_type": AnomalyType.UNKNOWN.value,
                "confidence": 0.4,
                "if_score": if_score,
                "stage": 2,
            }

        raw_label = int(np.asarray(self.classifier.predict(vector)).reshape(-1)[0])
        proba = np.asarray(self.classifier.predict_proba(vector), dtype=np.float32)
        confidence = float(np.max(proba, axis=1).reshape(-1)[0]) if proba.size else 0.0
        anomaly_type = self._TYPE_BY_LABEL.get(raw_label, AnomalyType.UNKNOWN).value
        return {
            "is_anomaly": True,
            "z_score": float(z_score),
            "anomaly_type": anomaly_type,
            "confidence": confidence,
            "if_score": if_score,
            "stage": 2,
        }

    @staticmethod
    def _feature_value(name: str, z_score: float, features: dict[str, float]) -> float:
        if name == "z_score":
            return float(z_score)
        raw = features.get(name, 0.0)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0
