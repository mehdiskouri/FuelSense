"""Integration-style tests for model retraining and promotion paths."""

from __future__ import annotations

from datetime import timedelta
from typing import Protocol, cast

import pytest
from django.utils import timezone

from fuelsense.core import tasks
from fuelsense.core.models import ModelRegistry
from fuelsense.core.tests.factories import FacilityFactory, InventoryLogFactory, ModelRegistryFactory

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


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


@pytest.mark.django_db
def test_retrain_integration_promotes_with_windowed_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retraining should promote a new active demand model when metrics improve."""
    facility = FacilityFactory()
    base = timezone.now() - timedelta(days=220)
    for i in range(170):
        InventoryLogFactory(
            facility=facility,
            timestamp=base + timedelta(days=i),
            consumption=50.0 + float(i) * 0.1,
            temperature=25.0,
            wind_speed=2.0,
            solar_irradiance=400.0,
        )

    existing = ModelRegistryFactory(
        facility=facility,
        model_type=ModelRegistry.ModelType.DEMAND_FORECAST,
        is_active=True,
        version=1,
        validation_rmse=1.5,
    )

    monkeypatch.setenv("FUELSENSE_ENABLE_TRAINING_TASKS", "1")

    class _Trainer:
        def train_and_register(self, **kwargs: object) -> dict[str, float | str]:
            train_data = cast("_ArrayLike", kwargs["train_data"])
            train_targets = cast("_ArrayLike", kwargs["train_targets"])
            val_data = cast("_ArrayLike", kwargs["val_data"])
            val_targets = cast("_ArrayLike", kwargs["val_targets"])
            test_data = cast("_ArrayLike", kwargs["test_data"])
            test_targets = cast("_ArrayLike", kwargs["test_targets"])

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

    monkeypatch.setattr("fuelsense.core.tasks.ForecastTrainer", _Trainer)

    result = tasks.retrain_model(facility.id, ModelRegistry.ModelType.DEMAND_FORECAST)
    _check(result["status"] == "promoted")

    existing.refresh_from_db()
    _check(existing.is_active is False)

    promoted = ModelRegistry.objects.get(
        facility=facility, model_type=ModelRegistry.ModelType.DEMAND_FORECAST, is_active=True,
    )
    training_rmse = cast("float | int | str", getattr(promoted, "training_rmse"))  # noqa: B009
    validation_rmse = cast("float | int | str", getattr(promoted, "validation_rmse"))  # noqa: B009
    _check(float(training_rmse) == EXPECTED_TRAINING_RMSE)
    _check(float(validation_rmse) == EXPECTED_VALIDATION_RMSE)
