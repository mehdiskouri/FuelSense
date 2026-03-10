from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from django.utils import timezone

from fuelsense.core import tasks
from fuelsense.core.models import ModelRegistry
from fuelsense.core.tests.factories import FacilityFactory, InventoryLogFactory, ModelRegistryFactory


@pytest.mark.django_db
def test_retrain_integration_promotes_with_windowed_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
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
        def train_and_register(self, **kwargs: Any) -> dict[str, Any]:
            train_data = kwargs["train_data"]
            train_targets = kwargs["train_targets"]
            val_data = kwargs["val_data"]
            val_targets = kwargs["val_targets"]
            test_data = kwargs["test_data"]
            test_targets = kwargs["test_targets"]

            assert train_data.ndim == 3 and train_data.shape[1:] == (90, 6)
            assert train_targets.ndim == 2 and train_targets.shape[1] == 14
            assert val_data.ndim == 3 and val_data.shape[1:] == (90, 6)
            assert val_targets.ndim == 2 and val_targets.shape[1] == 14
            assert test_data.ndim == 3 and test_data.shape[1:] == (90, 6)
            assert test_targets.ndim == 2 and test_targets.shape[1] == 14

            return {
                "run_id": "integration-run",
                "training_rmse": 0.95,
                "validation_rmse": 1.1,
                "test_rmse": 1.2,
            }

    monkeypatch.setattr("fuelsense.core.tasks.ForecastTrainer", _Trainer)

    result = tasks.retrain_model(facility.id, ModelRegistry.ModelType.DEMAND_FORECAST)
    assert result["status"] == "promoted"

    existing.refresh_from_db()
    assert existing.is_active is False

    promoted = ModelRegistry.objects.get(
        facility=facility, model_type=ModelRegistry.ModelType.DEMAND_FORECAST, is_active=True
    )
    assert float(promoted.training_rmse) == 0.95
    assert float(promoted.validation_rmse) == 1.1
