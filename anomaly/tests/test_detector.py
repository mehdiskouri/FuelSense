# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from anomaly.detector import AnomalyDetector, AnomalyType


class _ForestAnomaly:
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        _ = X
        return np.asarray([-0.23], dtype=np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        _ = X
        return np.asarray([-1], dtype=np.int32)


class _ForestNormal:
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        _ = X
        return np.asarray([0.21], dtype=np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        _ = X
        return np.asarray([1], dtype=np.int32)


class _Classifier:
    def __init__(self, label: int) -> None:
        self.label = label

    def predict(self, X: np.ndarray) -> np.ndarray:
        _ = X
        return np.asarray([self.label], dtype=np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _ = X
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


def test_stage1_zscore_gate_returns_non_anomaly() -> None:
    detector = AnomalyDetector()
    result = detector.detect(actual=101.0, predicted=100.0, rolling_std=1.0, features=_features())
    assert result["is_anomaly"] is False
    assert result["stage"] == 1
    assert result["anomaly_type"] is None


def test_stage2_if_rejects_returns_unknown() -> None:
    detector = AnomalyDetector()
    detector.forest = _ForestNormal()
    detector.classifier = _Classifier(label=0)

    result = detector.detect(actual=120.0, predicted=100.0, rolling_std=3.0, features=_features())
    assert result["stage"] == 2
    assert result["is_anomaly"] is False
    assert result["anomaly_type"] == AnomalyType.UNKNOWN.value
    assert result["confidence"] == 0.4


def test_stage2_if_accepts_returns_typed_anomaly() -> None:
    detector = AnomalyDetector()
    detector.forest = _ForestAnomaly()
    detector.classifier = _Classifier(label=0)

    result = detector.detect(actual=120.0, predicted=100.0, rolling_std=3.0, features=_features())
    assert result["stage"] == 2
    assert result["is_anomaly"] is True
    assert result["anomaly_type"] == AnomalyType.LEAK.value
    assert abs(float(result["confidence"]) - 0.92) < 1e-6


def test_stage2_maps_all_supported_types() -> None:
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

    assert actual == expected


def test_load_and_is_loaded(tmp_path: Path) -> None:
    detector = AnomalyDetector()
    assert detector.is_loaded is False

    forest_path = tmp_path / "forest.joblib"
    classifier_path = tmp_path / "classifier.joblib"
    joblib.dump(_ForestAnomaly(), forest_path)
    joblib.dump(_Classifier(label=1), classifier_path)

    detector.load(forest_path, classifier_path)
    assert detector.is_loaded is True


def test_detect_handles_small_std_and_missing_features() -> None:
    detector = AnomalyDetector()
    detector.forest = _ForestAnomaly()
    detector.classifier = _Classifier(label=2)

    result = detector.detect(
        actual=150.0,
        predicted=100.0,
        rolling_std=0.0,
        features={"day_of_week": 5.0},
    )
    assert result["stage"] == 2
    assert result["is_anomaly"] is True
    assert result["anomaly_type"] == AnomalyType.EQUIPMENT_DEGRADATION.value
    assert float(result["z_score"]) > 1_000_000


def test_stage2_without_models_returns_unknown() -> None:
    detector = AnomalyDetector()
    result = detector.detect(actual=120.0, predicted=100.0, rolling_std=2.0, features=_features())
    assert result["stage"] == 2
    assert result["is_anomaly"] is False
    assert result["anomaly_type"] == AnomalyType.UNKNOWN.value
