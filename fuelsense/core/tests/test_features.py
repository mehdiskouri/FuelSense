from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest
from django.utils import timezone

from fuelsense.core.features import build_drift_data, build_lookback_matrix, extract_training_data
from fuelsense.core.tests.factories import ForecastFactory, InventoryLogFactory, ModelRegistryFactory


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

    matrix = build_lookback_matrix([int(getattr(facility, "id"))])
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

    data = extract_training_data(int(getattr(facility, "id")))
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
