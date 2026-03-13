"""Two-stage anomaly detector implementation."""

from __future__ import annotations

import importlib
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast

import numpy as np


class _JoblibLike(Protocol):
    def load(self, filename: Path) -> object: ...


class _ForestLike(Protocol):
    def decision_function(self, features_array: np.ndarray) -> np.ndarray: ...

    def predict(self, features_array: np.ndarray) -> np.ndarray: ...


class _ClassifierLike(Protocol):
    def predict(self, features_array: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, features_array: np.ndarray) -> np.ndarray: ...


def _load_artifact(path: str | Path) -> object:
    module = importlib.import_module("joblib")
    load_fn = getattr(module, "load", None)
    if not callable(load_fn):
        msg = "joblib.load is unavailable"
        raise TypeError(msg)
    loader: _JoblibLike = module
    return loader.load(Path(path))


def _as_forest(model: object) -> _ForestLike:
    decision_fn = getattr(model, "decision_function", None)
    predict_fn = getattr(model, "predict", None)
    if callable(decision_fn) and callable(predict_fn):
        return cast("_ForestLike", model)
    msg = "Loaded forest artifact does not implement required interface"
    raise TypeError(msg)


def _as_classifier(model: object) -> _ClassifierLike:
    predict_fn = getattr(model, "predict", None)
    proba_fn = getattr(model, "predict_proba", None)
    if callable(predict_fn) and callable(proba_fn):
        return cast("_ClassifierLike", model)
    msg = "Loaded classifier artifact does not implement required interface"
    raise TypeError(msg)


class AnomalyType(StrEnum):
    """Supported anomaly categories emitted by the detector."""

    LEAK = "LEAK"
    THEFT = "THEFT"
    EQUIPMENT_DEGRADATION = "EQUIPMENT_DEGRADATION"
    DEMAND_SHIFT = "DEMAND_SHIFT"
    SENSOR_FAULT = "SENSOR_FAULT"
    UNKNOWN = "UNKNOWN"


class AnomalyDetector:
    """Two-stage detector combining thresholding and learned classifiers."""

    Z_THRESHOLD = 3.0
    N_ESTIMATORS = 200
    CONTAMINATION = 0.05
    feature_names = (
        "z_score",
        "z_score_rolling_3d",
        "consumption_delta_pct",
        "temperature_residual",
        "day_of_week",
        "hours_since_delivery",
        "inventory_level_pct",
    )
    _TYPE_BY_LABEL = MappingProxyType(
        {
            0: AnomalyType.LEAK,
            1: AnomalyType.THEFT,
            2: AnomalyType.EQUIPMENT_DEGRADATION,
            3: AnomalyType.DEMAND_SHIFT,
            4: AnomalyType.SENSOR_FAULT,
        },
    )

    def __init__(self) -> None:
        """Initialize unloaded detector components."""
        self.forest: _ForestLike | None = None
        self.classifier: _ClassifierLike | None = None

    @property
    def is_loaded(self) -> bool:
        """Return whether both stage-two models are loaded."""
        return self.forest is not None and self.classifier is not None

    def load(self, forest_path: str | Path, classifier_path: str | Path) -> None:
        """Load isolation forest and classifier artifacts from disk."""
        self.forest = _as_forest(_load_artifact(forest_path))
        self.classifier = _as_classifier(_load_artifact(classifier_path))

    def detect(self, actual: float, predicted: float, rolling_std: float, features: dict[str, float]) -> dict[str, Any]:
        """Detect anomalies using z-score gate then model-based classification."""
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
            [[self._feature_value(name, z_score, features) for name in self.feature_names]],
            dtype=np.float32,
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
        """Resolve one model feature value with robust numeric fallback."""
        if name == "z_score":
            return float(z_score)
        raw = features.get(name, 0.0)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0
