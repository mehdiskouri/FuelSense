"""Unit tests for anomaly detector staging and label mapping behavior."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

from anomaly.detector import AnomalyDetector, AnomalyType

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


def test_load_and_is_loaded() -> None:
    """Model load should flip detector loaded state to true."""

    class _JoblibModule:
        @staticmethod
        def load(path: object) -> object:
            path_str = str(path)
            if path_str.endswith("forest.joblib"):
                return _ForestAnomaly()
            if path_str.endswith("classifier.joblib"):
                return _Classifier(label=1)
            msg = f"unexpected artifact path: {path_str}"
            raise AssertionError(msg)

    def _import_module(name: str) -> object:
        _check(name == "joblib")
        return _JoblibModule()

    detector = AnomalyDetector()
    _check(detector.is_loaded is False)

    with patch("anomaly.detector.importlib.import_module", side_effect=_import_module):
        detector.load("forest.joblib", "classifier.joblib")
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


def test_load_artifact_requires_callable_joblib_loader() -> None:
    """Artifact loading should fail fast when imported joblib lacks callable load."""

    class _BrokenJoblibModule:
        load = "not-callable"

    detector = AnomalyDetector()
    with (
        patch("anomaly.detector.importlib.import_module", return_value=_BrokenJoblibModule()),
        pytest.raises(TypeError, match=r"joblib\.load is unavailable"),
    ):
        detector.load("forest.joblib", "classifier.joblib")


def test_stage2_type_guards_reject_invalid_loaded_models() -> None:
    """Model adapters should enforce the required forest/classifier interfaces."""

    class _BadForestJoblib:
        @staticmethod
        def load(path: object) -> object:
            _ = path
            return object()

    class _BadClassifierJoblib:
        @staticmethod
        def load(path: object) -> object:
            if str(path).endswith("forest.joblib"):
                return _ForestAnomaly()
            return object()

    detector = AnomalyDetector()
    with (
        patch("anomaly.detector.importlib.import_module", return_value=_BadForestJoblib()),
        pytest.raises(TypeError, match="forest artifact"),
    ):
        detector.load("forest.joblib", "classifier.joblib")

    with (
        patch("anomaly.detector.importlib.import_module", return_value=_BadClassifierJoblib()),
        pytest.raises(TypeError, match="classifier artifact"),
    ):
        detector.load("forest.joblib", "classifier.joblib")


def test_detect_feature_fallback_handles_non_numeric_values() -> None:
    """Feature coercion should fallback to 0.0 when stage-two features are non-numeric."""
    detector = AnomalyDetector()
    detector.forest = _ForestAnomaly()
    detector.classifier = _Classifier(label=0)

    noisy_features = _features()
    noisy_features["z_score_rolling_3d"] = cast("float", object())

    result = detector.detect(actual=120.0, predicted=100.0, rolling_std=3.0, features=noisy_features)
    _check(result["is_anomaly"] is True)
    _check(result["anomaly_type"] == AnomalyType.LEAK.value)
