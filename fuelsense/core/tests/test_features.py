"""Feature-engineering tests for lookback, drift payloads, and anomaly features."""

from __future__ import annotations

from datetime import timedelta
from typing import Protocol

import numpy as np
import pytest
from django.utils import timezone

from fuelsense.core.features import (
    build_anomaly_features,
    build_drift_data,
    build_lookback_matrix,
    extract_training_data,
)
from fuelsense.core.tests.factories import (
    create_delivery,
    create_delivery_item,
    create_forecast,
    create_inventory_log,
    create_model_registry,
)

LOOKBACK_SHAPE = (90, 6)
TEST_TARGET_HORIZON = 14
TRAIN_SEQUENCE_NDIM = 3
EXPECTED_TRAIN_SAMPLES = 39
ANOMALY_FEATURE_COUNT = 7
ZSCORE_TOLERANCE = 1e-5


class _HasPk(Protocol):
    pk: int


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


@pytest.mark.django_db
def test_build_lookback_matrix_shape_and_dow_features(sample_facilities: list[_HasPk]) -> None:
    """Lookback matrix builder should return finite tensors with expected shape."""
    facility = sample_facilities[0]
    base = timezone.now() - timedelta(days=120)
    for i in range(95):
        create_inventory_log(
            facility=facility,
            timestamp=base + timedelta(days=i),
            consumption=float(i),
            temperature=20.0,
            wind_speed=2.0,
            solar_irradiance=300.0,
        )

    matrix = build_lookback_matrix([int(facility.pk)])
    _check(matrix.shape == (1, 90, 6))
    _check(np.all(np.isfinite(matrix)))


@pytest.mark.django_db
def test_extract_training_data_temporal_split(sample_facilities: list[_HasPk]) -> None:
    """Training extractor should produce deterministic temporal split dimensions."""
    facility = sample_facilities[0]
    base = timezone.now() - timedelta(days=220)
    for i in range(170):
        create_inventory_log(
            facility=facility,
            timestamp=base + timedelta(days=i),
            consumption=10 + i,
            temperature=25.0,
            wind_speed=3.0,
            solar_irradiance=500.0,
        )

    data = extract_training_data(int(facility.pk))
    _check(data["test_data"].ndim == TRAIN_SEQUENCE_NDIM)
    _check(data["test_data"].shape[1:] == LOOKBACK_SHAPE)
    _check(data["test_targets"].shape[1] == TEST_TARGET_HORIZON)
    _check(data["test_data"].shape[0] == TEST_TARGET_HORIZON)
    _check(data["val_data"].shape[0] == TEST_TARGET_HORIZON)
    _check(data["train_data"].shape[0] == EXPECTED_TRAIN_SAMPLES)


@pytest.mark.django_db
def test_build_drift_data_structure(sample_facilities: list[_HasPk]) -> None:
    """Drift payload builder should expose required keys for monitoring."""
    facility = sample_facilities[0]
    for i in range(8):
        create_inventory_log(facility=facility, consumption=20 + i)
    create_forecast(facility=facility, predictions_json=[{"day": 1, "p50": 10, "p90": 12}])
    create_model_registry(
        facility=facility,
        model_type="DEMAND_FORECAST",
        is_active=True,
        validation_rmse=1.1,
    )

    payload = build_drift_data()
    _check(payload)
    first = payload[0]
    _check({"facility_id", "baseline_rmse", "recent_actuals", "recent_predictions"}.issubset(first.keys()))


@pytest.mark.django_db
def test_build_anomaly_features_returns_expected_shape(sample_facilities: list[_HasPk]) -> None:
    """Anomaly feature builder should emit full expected feature vector."""
    facility = sample_facilities[0]
    base = timezone.now() - timedelta(days=7)

    for i in range(7):
        create_inventory_log(
            facility=facility,
            timestamp=base + timedelta(days=i),
            consumption=30.0 + i,
            temperature=24.0 + i * 0.5,
            inventory_level=500.0 - i * 5,
        )

    create_forecast(
        facility=facility,
        predictions_json=[{"day": 1, "p10": 27.0, "p50": 30.0, "p90": 34.0}],
    )
    delivery = create_delivery()
    create_delivery_item(delivery=delivery, facility=facility, actual_arrival=timezone.now() - timedelta(hours=8))

    features = build_anomaly_features(int(facility.pk))
    if features is None:
        msg = "Expected anomaly features"
        raise AssertionError(msg)
    _check(len(features.keys()) == ANOMALY_FEATURE_COUNT)
    _check("z_score" in features)
    _check("inventory_level_pct" in features)


@pytest.mark.django_db
def test_build_anomaly_features_zscore_is_correct(sample_facilities: list[_HasPk]) -> None:
    """Anomaly z-score feature should match manual standard-score computation."""
    facility = sample_facilities[0]
    base = timezone.now() - timedelta(days=7)

    values = [20.0, 19.0, 21.0, 20.5, 20.2, 20.1, 30.0]
    for i, consumption in enumerate(values):
        create_inventory_log(
            facility=facility,
            timestamp=base + timedelta(days=i),
            consumption=consumption,
            temperature=25.0,
            inventory_level=500.0,
        )

    create_forecast(
        facility=facility,
        predictions_json=[{"day": 1, "p10": 21.0, "p50": 22.0, "p90": 25.0}],
    )

    features = build_anomaly_features(int(facility.pk))
    if features is None:
        msg = "Expected anomaly features"
        raise AssertionError(msg)

    expected_std = float(np.std(np.asarray(list(reversed(values)), dtype=np.float32)))
    expected_z = (30.0 - 22.0) / max(expected_std, 1e-6)
    z_score = features["z_score"]
    _check(abs(z_score - expected_z) < ZSCORE_TOLERANCE)


@pytest.mark.django_db
def test_build_anomaly_features_returns_none_when_insufficient_data(sample_facilities: list[_HasPk]) -> None:
    """Anomaly feature builder should return None for insufficient history."""
    facility = sample_facilities[0]
    create_inventory_log(facility=facility, consumption=20.0)
    _check(build_anomaly_features(int(facility.pk)) is None)
