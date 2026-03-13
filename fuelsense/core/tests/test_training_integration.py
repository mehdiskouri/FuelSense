"""Integration-style tests for model retraining and promotion paths."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Protocol, cast
from unittest.mock import patch

import pytest
from django.utils import timezone

from fuelsense.core import tasks
from fuelsense.core.models import ModelRegistry
from fuelsense.core.tests.factories import create_facility, create_inventory_log, create_model_registry

WINDOW_SIZE = 90
FEATURE_DIMENSIONS = 6
FORECAST_HORIZON = 14
SERIES_NDIM = 3
TARGETS_NDIM = 2
EXPECTED_TRAINING_RMSE = 0.95
EXPECTED_VALIDATION_RMSE = 1.1


class _ArrayLike(Protocol):
    ndim: int
    shape: tuple[int, ...]


class _DatasetLike(Protocol):
    train_data: _ArrayLike
    train_targets: _ArrayLike
    val_data: _ArrayLike
    val_targets: _ArrayLike
    test_data: _ArrayLike
    test_targets: _ArrayLike


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


@pytest.mark.django_db
def test_retrain_integration_promotes_with_windowed_dataset() -> None:
    """Retraining should promote a new active demand model when metrics improve."""
    facility = create_facility()
    base = timezone.now() - timedelta(days=220)
    for i in range(170):
        create_inventory_log(
            facility=facility,
            timestamp=base + timedelta(days=i),
            consumption=50.0 + float(i) * 0.1,
            temperature=25.0,
            wind_speed=2.0,
            solar_irradiance=400.0,
        )

    existing = create_model_registry(
        facility=facility,
        model_type=ModelRegistry.ModelType.DEMAND_FORECAST,
        is_active=True,
        version=1,
        validation_rmse=1.5,
    )

    class _Trainer:
        def train_and_register(self, **kwargs: object) -> dict[str, float | str]:
            dataset = cast("_DatasetLike", kwargs["dataset"])
            train_data = dataset.train_data
            train_targets = dataset.train_targets
            val_data = dataset.val_data
            val_targets = dataset.val_targets
            test_data = dataset.test_data
            test_targets = dataset.test_targets

            _check(train_data.ndim == SERIES_NDIM and train_data.shape[1:] == (WINDOW_SIZE, FEATURE_DIMENSIONS))
            _check(train_targets.ndim == TARGETS_NDIM and train_targets.shape[1] == FORECAST_HORIZON)
            _check(val_data.ndim == SERIES_NDIM and val_data.shape[1:] == (WINDOW_SIZE, FEATURE_DIMENSIONS))
            _check(val_targets.ndim == TARGETS_NDIM and val_targets.shape[1] == FORECAST_HORIZON)
            _check(test_data.ndim == SERIES_NDIM and test_data.shape[1:] == (WINDOW_SIZE, FEATURE_DIMENSIONS))
            _check(test_targets.ndim == TARGETS_NDIM and test_targets.shape[1] == FORECAST_HORIZON)

            return {
                "run_id": "integration-run",
                "training_rmse": EXPECTED_TRAINING_RMSE,
                "validation_rmse": EXPECTED_VALIDATION_RMSE,
                "test_rmse": 1.2,
            }

    with (
        patch.dict(os.environ, {"FUELSENSE_ENABLE_TRAINING_TASKS": "1"}, clear=False),
        patch("fuelsense.core.tasks.ForecastTrainer", _Trainer),
    ):
        result = tasks.retrain_model(facility.id, ModelRegistry.ModelType.DEMAND_FORECAST)
    _check(result["status"] == "promoted")

    existing.refresh_from_db()
    _check(existing.is_active is False)

    promoted = ModelRegistry.objects.get(
        facility=facility,
        model_type=ModelRegistry.ModelType.DEMAND_FORECAST,
        is_active=True,
    )
    training_rmse = cast("float | int | str", getattr(promoted, "training_rmse"))  # noqa: B009
    validation_rmse = cast("float | int | str", getattr(promoted, "validation_rmse"))  # noqa: B009
    _check(float(training_rmse) == EXPECTED_TRAINING_RMSE)
    _check(float(validation_rmse) == EXPECTED_VALIDATION_RMSE)
