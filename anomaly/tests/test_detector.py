"""Unit tests for anomaly detector staging and label mapping behavior."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

from typing import TYPE_CHECKING

import joblib
import numpy as np

from anomaly.detector import AnomalyDetector, AnomalyType

if TYPE_CHECKING:
    from pathlib import Path


PAIR_COUNT = 2
STAGE_TWO = 2
LOW_CONFIDENCE = 0.4
EPSILON = 1e-6
VERY_LARGE_ZSCORE = 1_000_000


class _ForestAnomaly:
    def decision_function(self, features_array: np.ndarray) -> np.ndarray:
        _ = features_array
        return np.asarray([-0.23], dtype=np.float32)

    def predict(self, features_array: np.ndarray) -> np.ndarray:
        _ = features_array
        return np.asarray([-1], dtype=np.int32)


class _ForestNormal:
    def decision_function(self, features_array: np.ndarray) -> np.ndarray:
        _ = features_array
        return np.asarray([0.21], dtype=np.float32)

    def predict(self, features_array: np.ndarray) -> np.ndarray:
        _ = features_array
        return np.asarray([1], dtype=np.int32)


class _Classifier:
    def __init__(self, label: int) -> None:
        self.label = label

    def predict(self, features_array: np.ndarray) -> np.ndarray:
        _ = features_array
        return np.asarray([self.label], dtype=np.int32)

    def predict_proba(self, features_array: np.ndarray) -> np.ndarray:
        _ = features_array
        out = np.zeros((1, 5), dtype=np.float32)
        out[0, self.label] = 0.92
        out[0, (self.label + 1) % 5] = 0.08
        return out


def _features() -> dict[str, float]:
    return {
        "z_score_rolling_3d": 3.2,
        "consumption_delta_pct": 0.4,
        "temperature_residual": 1.2,
        "day_of_week": 2.0,
        "hours_since_delivery": 12.0,
        "inventory_level_pct": 0.3,
    }


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


def test_stage1_zscore_gate_returns_non_anomaly() -> None:
    """Stage-1 z-score gate should return non-anomaly for low residuals."""
    detector = AnomalyDetector()
    result = detector.detect(actual=101.0, predicted=100.0, rolling_std=1.0, features=_features())
    _check(result["is_anomaly"] is False)
    _check(result["stage"] == 1)
    _check(result["anomaly_type"] is None)


def test_stage2_if_rejects_returns_unknown() -> None:
    """Stage-2 should emit UNKNOWN when IF model rejects anomaly hypothesis."""
    detector = AnomalyDetector()
    detector.forest = _ForestNormal()
    detector.classifier = _Classifier(label=0)

    result = detector.detect(actual=120.0, predicted=100.0, rolling_std=3.0, features=_features())
    _check(result["stage"] == STAGE_TWO)
    _check(result["is_anomaly"] is False)
    _check(result["anomaly_type"] == AnomalyType.UNKNOWN.value)
    _check(result["confidence"] == LOW_CONFIDENCE)


def test_stage2_if_accepts_returns_typed_anomaly() -> None:
    """Stage-2 should emit typed anomaly and confidence when IF accepts."""
    detector = AnomalyDetector()
    detector.forest = _ForestAnomaly()
    detector.classifier = _Classifier(label=0)

    result = detector.detect(actual=120.0, predicted=100.0, rolling_std=3.0, features=_features())
    _check(result["stage"] == STAGE_TWO)
    _check(result["is_anomaly"] is True)
    _check(result["anomaly_type"] == AnomalyType.LEAK.value)
    _check(abs(float(result["confidence"]) - 0.92) < EPSILON)


def test_stage2_maps_all_supported_types() -> None:
    """Classifier labels must map to the expected anomaly type enum values."""
    detector = AnomalyDetector()
    detector.forest = _ForestAnomaly()

    expected = [
        AnomalyType.LEAK.value,
        AnomalyType.THEFT.value,
        AnomalyType.EQUIPMENT_DEGRADATION.value,
        AnomalyType.DEMAND_SHIFT.value,
        AnomalyType.SENSOR_FAULT.value,
    ]
    actual: list[str] = []
    for label in range(5):
        detector.classifier = _Classifier(label=label)
        result = detector.detect(actual=120.0, predicted=100.0, rolling_std=3.0, features=_features())
        actual.append(str(result["anomaly_type"]))

    _check(actual == expected)


def test_load_and_is_loaded(tmp_path: Path) -> None:
    """Model load should flip detector loaded state to true."""
    detector = AnomalyDetector()
    _check(detector.is_loaded is False)

    forest_path = tmp_path / "forest.joblib"
    classifier_path = tmp_path / "classifier.joblib"
    joblib.dump(_ForestAnomaly(), forest_path)
    joblib.dump(_Classifier(label=1), classifier_path)

    detector.load(forest_path, classifier_path)
    _check(detector.is_loaded is True)


def test_detect_handles_small_std_and_missing_features() -> None:
    """Detection should remain stable with near-zero std and sparse features."""
    detector = AnomalyDetector()
    detector.forest = _ForestAnomaly()
    detector.classifier = _Classifier(label=2)

    result = detector.detect(
        actual=150.0,
        predicted=100.0,
        rolling_std=0.0,
        features={"day_of_week": 5.0},
    )
    _check(result["stage"] == STAGE_TWO)
    _check(result["is_anomaly"] is True)
    _check(result["anomaly_type"] == AnomalyType.EQUIPMENT_DEGRADATION.value)
    _check(float(result["z_score"]) > VERY_LARGE_ZSCORE)


def test_stage2_without_models_returns_unknown() -> None:
    """Without loaded models, stage-2 should return UNKNOWN non-anomaly."""
    detector = AnomalyDetector()
    result = detector.detect(actual=120.0, predicted=100.0, rolling_std=2.0, features=_features())
    _check(result["stage"] == STAGE_TWO)
    _check(result["is_anomaly"] is False)
    _check(result["anomaly_type"] == AnomalyType.UNKNOWN.value)
