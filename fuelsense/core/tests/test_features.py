from __future__ import annotations

from datetime import timedelta

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
    DeliveryFactory,
    DeliveryItemFactory,
    ForecastFactory,
    InventoryLogFactory,
    ModelRegistryFactory,
)


@pytest.mark.django_db
def test_build_lookback_matrix_shape_and_dow_features(sample_facilities: list[object]) -> None:
    facility = sample_facilities[0]
    base = timezone.now() - timedelta(days=120)
    for i in range(95):
        InventoryLogFactory(
            facility=facility,
            timestamp=base + timedelta(days=i),
            consumption=float(i),
            temperature=20.0,
            wind_speed=2.0,
            solar_irradiance=300.0,
        )

    matrix = build_lookback_matrix([int(facility.id)])
    assert matrix.shape == (1, 90, 6)
    assert np.all(np.isfinite(matrix))


@pytest.mark.django_db
def test_extract_training_data_temporal_split(sample_facilities: list[object]) -> None:
    facility = sample_facilities[0]
    base = timezone.now() - timedelta(days=70)
    for i in range(60):
        InventoryLogFactory(
            facility=facility,
            timestamp=base + timedelta(days=i),
            consumption=10 + i,
            temperature=25.0,
            wind_speed=3.0,
            solar_irradiance=500.0,
        )

    data = extract_training_data(int(facility.id))
    assert data["test_data"].shape[0] == 14
    assert data["val_data"].shape[0] == 14
    assert data["train_data"].shape[0] == 32


@pytest.mark.django_db
def test_build_drift_data_structure(sample_facilities: list[object]) -> None:
    facility = sample_facilities[0]
    for i in range(8):
        InventoryLogFactory(facility=facility, consumption=20 + i)
    ForecastFactory(facility=facility, predictions_json=[{"day": 1, "p50": 10, "p90": 12}])
    ModelRegistryFactory(
        facility=facility,
        model_type="DEMAND_FORECAST",
        is_active=True,
        validation_rmse=1.1,
    )

    payload = build_drift_data()
    assert payload
    first = payload[0]
    assert {"facility_id", "baseline_rmse", "recent_actuals", "recent_predictions"}.issubset(first.keys())


@pytest.mark.django_db
def test_build_anomaly_features_returns_expected_shape(sample_facilities: list[object]) -> None:
    facility = sample_facilities[0]
    base = timezone.now() - timedelta(days=7)

    for i in range(7):
        InventoryLogFactory(
            facility=facility,
            timestamp=base + timedelta(days=i),
            consumption=30.0 + i,
            temperature=24.0 + i * 0.5,
            inventory_level=500.0 - i * 5,
        )

    ForecastFactory(
        facility=facility,
        predictions_json=[{"day": 1, "p10": 27.0, "p50": 30.0, "p90": 34.0}],
    )
    delivery = DeliveryFactory()
    DeliveryItemFactory(delivery=delivery, facility=facility, actual_arrival=timezone.now() - timedelta(hours=8))

    features = build_anomaly_features(int(facility.id))
    assert features is not None
    assert len(features.keys()) == 7
    assert "z_score" in features
    assert "inventory_level_pct" in features


@pytest.mark.django_db
def test_build_anomaly_features_zscore_is_correct(sample_facilities: list[object]) -> None:
    facility = sample_facilities[0]
    base = timezone.now() - timedelta(days=7)

    values = [20.0, 19.0, 21.0, 20.5, 20.2, 20.1, 30.0]
    for i, consumption in enumerate(values):
        InventoryLogFactory(
            facility=facility,
            timestamp=base + timedelta(days=i),
            consumption=consumption,
            temperature=25.0,
            inventory_level=500.0,
        )

    ForecastFactory(
        facility=facility,
        predictions_json=[{"day": 1, "p10": 21.0, "p50": 22.0, "p90": 25.0}],
    )

    features = build_anomaly_features(int(facility.id))
    assert features is not None

    expected_std = float(np.std(np.asarray(list(reversed(values)), dtype=np.float32)))
    expected_z = (30.0 - 22.0) / max(expected_std, 1e-6)
    assert abs(features["z_score"] - expected_z) < 1e-5


@pytest.mark.django_db
def test_build_anomaly_features_returns_none_when_insufficient_data(sample_facilities: list[object]) -> None:
    facility = sample_facilities[0]
    InventoryLogFactory(facility=facility, consumption=20.0)
    assert build_anomaly_features(int(facility.id)) is None
