from __future__ import annotations

from typing import Any

import pytest
from django.utils import timezone

from fuelsense.core import tasks
from fuelsense.core.models import AnomalyAlert, Delivery, DeliveryItem, Forecast, ModelRegistry, PlanningCycle
from fuelsense.core.tests.factories import (
    DepotFacilityAssignmentFactory,
    DepotFactory,
    FacilityFactory,
    ForecastFactory,
    InventoryLogFactory,
    ModelRegistryFactory,
    VehicleFactory,
)


@pytest.mark.django_db
def test_tasks_execute_and_return_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    facility = FacilityFactory()
    depot = DepotFactory()
    VehicleFactory(depot=depot, is_available=True)
    DepotFacilityAssignmentFactory(depot=depot, facility=facility)

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


@pytest.mark.django_db
def test_run_planning_cycle_creates_planning_and_deliveries(monkeypatch: pytest.MonkeyPatch) -> None:
    depot = DepotFactory()
    vehicle = VehicleFactory(depot=depot, is_available=True)
    facility = FacilityFactory(current_inventory=50.0, dynamic_reorder_point=120.0)
    DepotFacilityAssignmentFactory(depot=depot, facility=facility)

    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "1")
    monkeypatch.setattr(
        "fuelsense.core.tasks.build_optimizer_request",
        lambda _depot, _facilities: (
            {
                "depot_lat": 24.7,
                "depot_lng": 46.7,
                "vehicles": [{"capacity": 1000.0, "cost_per_km": 2.0}],
                "stops": [{"facility_index": 1, "demand": 70.0, "time_window_start": 300, "time_window_end": 900, "service_time": 30}],
                "distance_matrix": [[0.0, 12.0], [12.0, 0.0]],
                "max_route_duration": 480,
            },
            {1: facility.id},
        ),
    )

    class _Response:
        def raise_for_status(self) -> None:
            return None

        @staticmethod
        def json() -> dict[str, object]:
            return {
                "status": "optimal",
                "routes": [
                    {
                        "vehicle_index": 0,
                        "stops": [{"facility_index": 1, "demand": 70.0, "arrival_min": 360, "sequence": 1}],
                        "distance_km": 24.0,
                        "cost": 48.0,
                    }
                ],
                "total_distance_km": 24.0,
                "total_cost": 48.0,
                "vehicles_used": 1,
                "solver_time_ms": 20.0,
                "baseline_cost": 60.0,
                "cost_reduction_pct": 20.0,
            }

    class _Client:
        def __enter__(self) -> "_Client":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            _ = exc_type, exc, tb

        def post(self, _url: str, json: dict[str, object]) -> _Response:
            assert "stops" in json
            return _Response()

    monkeypatch.setattr("fuelsense.core.tasks.httpx.Client", lambda timeout: _Client())

    result = tasks.run_planning_cycle()
    assert result["deliveries_created"] == 1
    assert PlanningCycle.objects.filter(trigger_type=PlanningCycle.TriggerType.SCHEDULED).count() == 1
    assert Delivery.objects.filter(vehicle=vehicle).count() == 1
    assert DeliveryItem.objects.filter(facility=facility).count() == 1


@pytest.mark.django_db
def test_daily_tick_dispatches_chord_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    facility = FacilityFactory(is_active=True)
    monkeypatch.setenv("FUELSENSE_ENABLE_DAILY_TICK", "1")

    captured: dict[str, object] = {}

    class _Pipeline:
        def delay(self) -> None:
            captured["dispatched"] = True

    class _Sig:
        def __init__(self, name: str) -> None:
            self.name = name

        def __or__(self, _other: object) -> "_Sig":
            return self

    def _fake_chord(header: list[object], callback: object) -> _Sig:
        captured["header_len"] = len(header)
        captured["callback"] = callback
        return _Sig("chord")

    monkeypatch.setattr("fuelsense.core.tasks.chord", _fake_chord)
    monkeypatch.setattr("fuelsense.core.tasks.run_batch_forecasts", _Sig("forecast"))
    monkeypatch.setattr("fuelsense.core.tasks.run_batch_anomaly_detection", _Sig("anomaly"))
    monkeypatch.setattr("fuelsense.core.tasks.check_all_drift", _Sig("drift"))
    monkeypatch.setattr("fuelsense.core.tasks.run_planning_cycle", _Sig("planning"))
    monkeypatch.setattr("fuelsense.core.tasks.ingest_facility_data", _Sig("ingest"))
    monkeypatch.setattr("fuelsense.core.tasks.ingestion_complete", _Sig("complete"))
    monkeypatch.setattr(_Sig, "delay", lambda self: captured.setdefault("dispatched", True), raising=False)

    # Provide si on our fake task signatures.
    setattr(_Sig, "si", lambda self, *args, **kwargs: _Sig(f"{self.name}.si"))

    result = tasks.daily_tick()
    assert result["status"] == "dispatched"
    assert captured.get("header_len") == 1
    assert captured.get("dispatched") is True


@pytest.mark.django_db
def test_trigger_emergency_delivery_creates_emergency_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    depot = DepotFactory()
    VehicleFactory(depot=depot, is_available=True)
    facility = FacilityFactory(current_inventory=40.0, dynamic_reorder_point=140.0)
    DepotFacilityAssignmentFactory(depot=depot, facility=facility)

    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "1")
    monkeypatch.setattr(
        "fuelsense.core.tasks.build_optimizer_request",
        lambda _depot, _facilities: (
            {
                "depot_lat": 24.7,
                "depot_lng": 46.7,
                "vehicles": [{"capacity": 1000.0, "cost_per_km": 2.0}],
                "stops": [{"facility_index": 1, "demand": 90.0, "time_window_start": 300, "time_window_end": 900, "service_time": 30}],
                "distance_matrix": [[0.0, 10.0], [10.0, 0.0]],
                "max_route_duration": 480,
            },
            {1: facility.id},
        ),
    )

    class _Response:
        def raise_for_status(self) -> None:
            return None

        @staticmethod
        def json() -> dict[str, object]:
            return {
                "status": "optimal",
                "routes": [
                    {
                        "vehicle_index": 0,
                        "stops": [{"facility_index": 1, "demand": 90.0, "arrival_min": 240, "sequence": 1}],
                        "distance_km": 20.0,
                        "cost": 40.0,
                    }
                ],
                "total_distance_km": 20.0,
                "total_cost": 40.0,
                "vehicles_used": 1,
                "solver_time_ms": 15.0,
                "baseline_cost": 50.0,
                "cost_reduction_pct": 20.0,
            }

    class _Client:
        def __enter__(self) -> "_Client":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            _ = exc_type, exc, tb

        def post(self, _url: str, json: dict[str, object]) -> _Response:
            assert "stops" in json
            return _Response()

    monkeypatch.setattr("fuelsense.core.tasks.httpx.Client", lambda timeout: _Client())

    result = tasks.trigger_emergency_delivery(facility.id)
    assert result["exists"] is True
    assert result["deliveries_created"] == 1
    assert PlanningCycle.objects.filter(trigger_type=PlanningCycle.TriggerType.EMERGENCY).count() == 1
