from __future__ import annotations

from typing import Any

import pytest

from fuelsense.core import tasks
from fuelsense.core.models import AnomalyAlert, Forecast, ModelRegistry
from fuelsense.core.tests.factories import FacilityFactory, ForecastFactory, InventoryLogFactory, ModelRegistryFactory


@pytest.mark.django_db
def test_tasks_execute_and_return_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    facility = FacilityFactory()

    monkeypatch.setenv("FUELSENSE_ENABLE_TRAINING_TASKS", "0")

    assert "facility_count" in tasks.daily_tick()
    assert "active_facilities" in tasks.ingest_hourly()
    assert tasks.ingest_facility_data(facility.id)["facility_id"] == facility.id
    assert "recent_logs" in tasks.ingestion_complete()

    forecast_result = tasks.run_batch_forecasts([facility.id])
    assert "facility_count" in forecast_result
    assert Forecast.objects.filter(facility=facility).exists()

    assert "facility_count" in tasks.run_batch_anomaly_detection([facility.id])
    assert "models" in tasks.check_all_drift()
    assert tasks.retrain_model(facility.id, "DEMAND_FORECAST")["model_type"] == "DEMAND_FORECAST"
    assert "queued" in tasks.run_planning_cycle()
    assert tasks.trigger_emergency_delivery(facility.id)["exists"] is True


@pytest.mark.django_db
def test_retrain_model_promotes_when_improved(monkeypatch: pytest.MonkeyPatch) -> None:
    facility = FacilityFactory()
    current = ModelRegistryFactory(
        facility=facility,
        model_type="DEMAND_FORECAST",
        is_active=True,
        version=1,
        validation_rmse=2.0,
    )

    monkeypatch.setenv("FUELSENSE_ENABLE_TRAINING_TASKS", "1")

    monkeypatch.setattr(
        "fuelsense.core.tasks.extract_training_data",
        lambda _facility_id: {
            "train_data": __import__("numpy").zeros((10, 90, 6), dtype="float32"),
            "train_targets": __import__("numpy").zeros((10,), dtype="float32"),
            "val_data": __import__("numpy").zeros((5, 90, 6), dtype="float32"),
            "val_targets": __import__("numpy").zeros((5,), dtype="float32"),
            "test_data": __import__("numpy").zeros((5, 90, 6), dtype="float32"),
            "test_targets": __import__("numpy").zeros((5,), dtype="float32"),
        },
    )

    class _Trainer:
        def train_and_register(self, **kwargs: Any) -> dict[str, Any]:
            _ = kwargs
            return {
                "run_id": "new-run",
                "test_rmse": 1.2,
            }

    monkeypatch.setattr("fuelsense.core.tasks.ForecastTrainer", _Trainer)

    result = tasks.retrain_model(facility.id, "DEMAND_FORECAST")
    assert result["status"] == "promoted"

    current.refresh_from_db()
    assert current.is_active is False
    assert ModelRegistry.objects.filter(facility=facility, model_type="DEMAND_FORECAST", is_active=True).count() == 1


@pytest.mark.django_db
def test_run_batch_anomaly_detection_creates_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    facility = FacilityFactory()
    InventoryLogFactory(facility=facility, consumption=125.0)
    ForecastFactory(facility=facility, predictions_json=[{"day": 1, "p10": 80.0, "p50": 90.0, "p90": 100.0}])

    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_ANOMALY", "1")
    monkeypatch.setattr(
        "fuelsense.core.tasks.build_anomaly_features",
        lambda _facility_id: {
            "z_score": 4.2,
            "z_score_rolling_3d": 3.1,
            "consumption_delta_pct": 0.6,
            "temperature_residual": 1.5,
            "day_of_week": 2.0,
            "hours_since_delivery": 5.0,
            "inventory_level_pct": 0.22,
        },
    )

    class _Response:
        def raise_for_status(self) -> None:
            return None

        @staticmethod
        def json() -> dict[str, object]:
            return {
                "is_anomaly": True,
                "anomaly_type": "LEAK",
                "confidence": 0.91,
                "stage": 2,
                "z_score": 4.2,
                "if_score": -0.4,
            }

    class _Client:
        def __enter__(self) -> "_Client":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            _ = exc_type, exc, tb

        def post(self, _url: str, json: dict[str, object]) -> _Response:
            assert "features" in json
            return _Response()

    monkeypatch.setattr("fuelsense.core.tasks.httpx.Client", lambda timeout: _Client())

    result = tasks.run_batch_anomaly_detection([facility.id])
    assert result["anomaly_count"] == 1
    assert AnomalyAlert.objects.filter(facility=facility, anomaly_type="LEAK").count() == 1


@pytest.mark.django_db
def test_run_batch_anomaly_detection_handles_service_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    facility = FacilityFactory()
    InventoryLogFactory(facility=facility, consumption=125.0)
    ForecastFactory(facility=facility, predictions_json=[{"day": 1, "p10": 80.0, "p50": 90.0, "p90": 100.0}])

    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_ANOMALY", "1")
    monkeypatch.setattr(
        "fuelsense.core.tasks.build_anomaly_features",
        lambda _facility_id: {
            "z_score": 4.2,
            "z_score_rolling_3d": 3.1,
            "consumption_delta_pct": 0.6,
            "temperature_residual": 1.5,
            "day_of_week": 2.0,
            "hours_since_delivery": 5.0,
            "inventory_level_pct": 0.22,
        },
    )

    class _Client:
        def __enter__(self) -> "_Client":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            _ = exc_type, exc, tb

        def post(self, _url: str, json: dict[str, object]) -> object:
            _ = json
            raise tasks.httpx.ConnectError("service unavailable")

    monkeypatch.setattr("fuelsense.core.tasks.httpx.Client", lambda timeout: _Client())

    result = tasks.run_batch_anomaly_detection([facility.id])
    assert result["anomaly_count"] == 0
    assert AnomalyAlert.objects.filter(facility=facility).count() == 0
