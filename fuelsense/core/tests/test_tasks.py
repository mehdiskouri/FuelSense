"""Task orchestration regression tests for fallback, planning, and emergency flows."""

# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from typing import Protocol, Self, cast

import pytest

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

CPU_COUNT_12 = 12
MAJOR_VERSION_2 = 2
DEPOT_LAT_FAILURE_THRESHOLD = 24.9
DEPOT_LAT_PARALLEL_FAILURE_THRESHOLD = 24.5
EXPECTED_ENTITY_COUNT_20 = 20
EXPECTED_PLANNING_CYCLE_COUNT_2 = 2


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


class _HasPk(Protocol):
    pk: int


def _pk(obj: _HasPk) -> int:
    return int(obj.pk)


def _call_handle_remote_unavailable(service_name: str, reason: str) -> None:
    func = cast("object", tasks.__dict__["_handle_remote_unavailable"])
    cast("_HandleRemoteUnavailable", func)(service_name, reason)


def _call_fallback_optimizer_response(payload: dict[str, object]) -> dict[str, object]:
    func = cast("_FallbackOptimizerResponse", tasks.__dict__["_fallback_optimizer_response"])
    return func(payload)


def _call_run_optimizer(payload: dict[str, object], *, optimizer_client: object | None = None) -> dict[str, object]:
    func = cast("_RunOptimizer", tasks.__dict__["_run_optimizer"])
    return func(payload, optimizer_client=optimizer_client)


def _call_default_parallel_workers() -> int:
    func = cast("_DefaultParallelWorkers", tasks.__dict__["_default_parallel_workers"])
    return int(func())


class _HandleRemoteUnavailable(Protocol):
    def __call__(self, service_name: str, reason: str) -> None: ...


class _FallbackOptimizerResponse(Protocol):
    def __call__(self, payload: dict[str, object]) -> dict[str, object]: ...


class _RunOptimizer(Protocol):
    def __call__(self, payload: dict[str, object], *, optimizer_client: object | None = None) -> dict[str, object]: ...


class _DefaultParallelWorkers(Protocol):
    def __call__(self) -> int: ...


def test_handle_remote_unavailable_requires_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require-remote mode should raise when remote dependency is unavailable."""
    monkeypatch.setenv("FUELSENSE_REQUIRE_REMOTE_SERVICES", "1")
    with pytest.raises(RuntimeError):
        _call_handle_remote_unavailable("forecaster", "remote_disabled")


def test_handle_remote_unavailable_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Best-effort mode should log a warning instead of raising on remote failure."""
    monkeypatch.setenv("FUELSENSE_REQUIRE_REMOTE_SERVICES", "0")
    seen: dict[str, object] = {}

    def _warning(msg: str, **kwargs: object) -> None:
        seen["msg"] = msg
        seen["extra"] = kwargs.get("extra")

    monkeypatch.setattr(tasks.logger, "warning", _warning)
    _call_handle_remote_unavailable("forecaster", "remote_disabled")
    _check(seen.get("msg") == "using fallback mode")


def test_fallback_optimizer_response_handles_invalid_payload() -> None:
    """Fallback optimizer should handle malformed payloads and still return valid shapes."""
    infeasible = _call_fallback_optimizer_response({"vehicles": [], "distance_matrix": []})
    _check(infeasible["status"] == "infeasible")
    _check(infeasible["routes"] == [])

    payload = {
        "vehicles": [{"cost_per_km": 2.0}],
        "distance_matrix": [[0.0, 10.0], [10.0, 0.0]],
        "stops": ["bad-stop", {"facility_index": -1}, {"facility_index": 1, "demand": 5.0}],
    }
    out = _call_fallback_optimizer_response(payload)
    _check(out["status"] == "optimal")
    _check(len(cast("list[object]", out["routes"])) == 1)


def test_run_optimizer_remote_http_error_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remote optimizer HTTP failures should fall back to local infeasible response."""
    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "1")

    class _Client:
        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object,
        ) -> None:
            _ = exc_type, exc, tb

        def post(self, _url: str, json: dict[str, object]) -> object:
            _ = json
            msg = "boom"
            raise tasks.httpx.ConnectError(msg)

    monkeypatch.setattr("fuelsense.core.tasks.httpx.Client", lambda _timeout: _Client())

    result = _call_run_optimizer({"vehicles": [], "distance_matrix": []})
    _check(result["status"] == "infeasible")


def test_run_optimizer_with_provided_client_http_error_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provided optimizer client HTTP errors should also trigger fallback response."""
    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "1")

    class _Client:
        def post(self, _url: str, json: dict[str, object]) -> object:
            _ = json
            msg = "boom"
            raise tasks.httpx.ConnectError(msg)

    result = _call_run_optimizer(
        {"vehicles": [], "distance_matrix": []},
        optimizer_client=_Client(),
    )
    _check(result["status"] == "infeasible")


def test_default_parallel_workers_uses_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parallel worker helper should use CPU count with sensible fallback."""
    monkeypatch.setattr(tasks.os, "cpu_count", lambda: CPU_COUNT_12)
    _check(_call_default_parallel_workers() == CPU_COUNT_12)

    monkeypatch.setattr(tasks.os, "cpu_count", lambda: None)
    _check(_call_default_parallel_workers() == 1)


@pytest.mark.django_db
def test_run_batch_forecasts_empty_lookback_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forecast batch task should return empty summary when lookback matrix is empty."""
    monkeypatch.setattr(
        "fuelsense.core.tasks.build_lookback_matrix",
        lambda _facility_ids: __import__("numpy").zeros((0, 90, 6), dtype="float32"),
    )
    result = tasks.run_batch_forecasts()
    _check(result == {"facility_count": 0, "created": 0})


@pytest.mark.django_db
def test_run_batch_forecasts_fallback_preserves_prior_dynamic_reorder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback forecast mode should preserve existing reliable dynamic reorder points."""
    facility = FacilityFactory(dynamic_reorder_point=345.0, min_safe_inventory=200.0)

    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_FORECAST", "0")
    monkeypatch.setattr(
        "fuelsense.core.tasks.build_lookback_matrix",
        lambda _facility_ids: __import__("numpy").ones((len(_facility_ids), 90, 6), dtype="float32"),
    )

    result = tasks.run_batch_forecasts([facility.id])
    _check(result["created"] == 1)

    facility.refresh_from_db()
    _check(facility.dynamic_reorder_point == pytest.approx(345.0))


@pytest.mark.django_db
def test_run_batch_forecasts_fallback_uses_consumption_heuristic_without_prior_dynamic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback forecast mode should compute reorder point from recent consumption when needed."""
    facility = FacilityFactory(dynamic_reorder_point=None, min_safe_inventory=200.0)
    InventoryLogFactory(facility=facility, consumption=100.0)
    InventoryLogFactory(facility=facility, consumption=100.0)
    InventoryLogFactory(facility=facility, consumption=100.0)

    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_FORECAST", "0")
    monkeypatch.setattr(
        "fuelsense.core.tasks.build_lookback_matrix",
        lambda _facility_ids: __import__("numpy").ones((len(_facility_ids), 90, 6), dtype="float32"),
    )

    result = tasks.run_batch_forecasts([facility.id])
    _check(result["created"] == 1)

    facility.refresh_from_db()
    _check(facility.dynamic_reorder_point == pytest.approx(330.0))


@pytest.mark.django_db
def test_run_batch_forecasts_remote_http_error_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remote forecaster HTTP failures should still produce fallback forecast rows."""
    facility = FacilityFactory(dynamic_reorder_point=200.0, min_safe_inventory=100.0)
    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_FORECAST", "1")
    monkeypatch.setattr(
        "fuelsense.core.tasks.build_lookback_matrix",
        lambda _facility_ids: __import__("numpy").ones((len(_facility_ids), 90, 6), dtype="float32"),
    )

    class _Client:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> None:
            _ = exc_type, exc, tb

        def post(self, _url: str, json: dict[str, object]) -> object:
            _ = json
            msg = "service unavailable"
            raise tasks.httpx.ConnectError(msg)

    monkeypatch.setattr("fuelsense.core.tasks.httpx.Client", lambda _timeout: _Client())

    result = tasks.run_batch_forecasts([facility.id])
    _check(result["created"] == 1)


@pytest.mark.django_db
def test_tasks_execute_and_return_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Top-level orchestration tasks should run and return expected summary shapes."""
    facility = FacilityFactory()
    depot = DepotFactory()
    VehicleFactory(depot=depot, is_available=True)
    DepotFacilityAssignmentFactory(depot=depot, facility=facility)

    monkeypatch.setenv("FUELSENSE_ENABLE_TRAINING_TASKS", "0")

    _check("facility_count" in tasks.daily_tick())
    _check("active_facilities" in tasks.ingest_hourly())
    _check(tasks.ingest_facility_data(facility.id)["facility_id"] == facility.id)
    _check("recent_logs" in tasks.ingestion_complete())

    forecast_result = tasks.run_batch_forecasts([facility.id])
    _check("facility_count" in forecast_result)
    _check(Forecast.objects.filter(facility=facility).exists())

    _check("facility_count" in tasks.run_batch_anomaly_detection([facility.id]))
    _check("models" in tasks.check_all_drift())
    _check(tasks.retrain_model(facility.id, "DEMAND_FORECAST")["model_type"] == "DEMAND_FORECAST")
    _check("queued" in tasks.run_planning_cycle())
    _check(tasks.trigger_emergency_delivery(facility.id)["exists"] is True)


@pytest.mark.django_db
def test_retrain_model_promotes_when_improved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrain flow should promote a new model when metrics improve over active model."""
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
        def train_and_register(self, **kwargs: object) -> dict[str, object]:
            _ = kwargs
            return {
                "run_id": "new-run",
                "test_rmse": 1.2,
            }

    monkeypatch.setattr("fuelsense.core.tasks.ForecastTrainer", _Trainer)

    result = tasks.retrain_model(facility.id, "DEMAND_FORECAST")
    _check(result["status"] == "promoted")

    current.refresh_from_db()
    _check(current.is_active is False)
    _check(ModelRegistry.objects.filter(facility=facility, model_type="DEMAND_FORECAST", is_active=True).count() == 1)


@pytest.mark.django_db
def test_retrain_model_skips_unsupported_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrain entrypoint should skip unsupported model types."""
    monkeypatch.setenv("FUELSENSE_ENABLE_TRAINING_TASKS", "1")
    result = tasks.retrain_model(None, "UNSUPPORTED")
    _check(result["status"] == "skipped")


@pytest.mark.django_db
def test_retrain_model_promotes_anomaly_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anomaly retraining should promote the next version and demote previous active model."""
    current = ModelRegistryFactory(
        facility=None,
        model_type="ANOMALY_DETECTOR",
        is_active=True,
        version=1,
    )

    monkeypatch.setenv("FUELSENSE_ENABLE_TRAINING_TASKS", "1")

    class _AnomalyTrainer:
        def train_and_register(self, **kwargs: object) -> dict[str, object]:
            _ = kwargs
            return {
                "run_id": "anomaly-run-2",
                "f1": 0.82,
                "classifier_accuracy": 0.9,
            }

    monkeypatch.setattr("fuelsense.core.tasks.AnomalyTrainer", _AnomalyTrainer)

    result = tasks.retrain_model(None, "ANOMALY_DETECTOR")
    _check(result["status"] == "promoted")
    _check(result["model_type"] == "ANOMALY_DETECTOR")

    current.refresh_from_db()
    _check(current.is_active is False)
    promoted = ModelRegistry.objects.get(model_type="ANOMALY_DETECTOR", facility=None, is_active=True)
    _check(promoted.version == MAJOR_VERSION_2)
    _check(promoted.mlflow_run_id == "anomaly-run-2")


@pytest.mark.django_db
def test_retrain_model_handles_anomaly_training_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anomaly retraining should return failed status when trainer raises."""
    monkeypatch.setenv("FUELSENSE_ENABLE_TRAINING_TASKS", "1")

    class _AnomalyTrainer:
        def train_and_register(self, **kwargs: object) -> dict[str, object]:
            _ = kwargs
            msg = "anomaly train failed"
            raise RuntimeError(msg)

    monkeypatch.setattr("fuelsense.core.tasks.AnomalyTrainer", _AnomalyTrainer)
    result = tasks.retrain_model(None, "ANOMALY_DETECTOR")
    _check(result["status"] == "failed")


@pytest.mark.django_db
def test_retrain_model_handles_training_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forecast retraining should return failed status when training backend errors."""
    facility = FacilityFactory()
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
        def train_and_register(self, **kwargs: object) -> dict[str, object]:
            _ = kwargs
            msg = "train failed"
            raise RuntimeError(msg)

    monkeypatch.setattr("fuelsense.core.tasks.ForecastTrainer", _Trainer)
    result = tasks.retrain_model(facility.id, "DEMAND_FORECAST")
    _check(result["status"] == "failed")


@pytest.mark.django_db
def test_run_batch_anomaly_detection_creates_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anomaly batch task should create alerts from positive remote anomaly results."""
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
        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> None:
            _ = exc_type, exc, tb

        def post(self, _url: str, json: dict[str, object]) -> _Response:
            _check("features" in json)
            return _Response()

    monkeypatch.setattr("fuelsense.core.tasks.httpx.Client", lambda _timeout: _Client())

    result = tasks.run_batch_anomaly_detection([facility.id])
    _check(result["anomaly_count"] == 1)
    _check(AnomalyAlert.objects.filter(facility=facility, anomaly_type="LEAK").count() == 1)


@pytest.mark.django_db
def test_run_batch_anomaly_detection_handles_service_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anomaly batch task should handle remote service failures without creating alerts."""
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
        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> None:
            _ = exc_type, exc, tb

        def post(self, _url: str, json: dict[str, object]) -> object:
            _ = json
            msg = "service unavailable"
            raise tasks.httpx.ConnectError(msg)

    monkeypatch.setattr("fuelsense.core.tasks.httpx.Client", lambda _timeout: _Client())

    result = tasks.run_batch_anomaly_detection([facility.id])
    _check(result["anomaly_count"] == 0)
    _check(AnomalyAlert.objects.filter(facility=facility).count() == 0)


@pytest.mark.django_db
def test_run_planning_cycle_creates_planning_and_deliveries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Planning cycle should create delivery records for optimal optimizer responses."""
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
                "stops": [
                    {
                        "facility_index": 1,
                        "demand": 70.0,
                        "time_window_start": 300,
                        "time_window_end": 900,
                        "service_time": 30,
                    },
                ],
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
                    },
                ],
                "total_distance_km": 24.0,
                "total_cost": 48.0,
                "vehicles_used": 1,
                "solver_time_ms": 20.0,
                "baseline_cost": 60.0,
                "cost_reduction_pct": 20.0,
            }

    class _Client:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> None:
            _ = exc_type, exc, tb

        def post(self, _url: str, json: dict[str, object]) -> _Response:
            _check("stops" in json)
            return _Response()

    monkeypatch.setattr("fuelsense.core.tasks.httpx.Client", lambda _timeout: _Client())

    result = tasks.run_planning_cycle()
    _check(result["deliveries_created"] == 1)
    cycle = PlanningCycle.objects.get(trigger_type=PlanningCycle.TriggerType.SCHEDULED)
    _check(cycle.status == PlanningCycle.ExecutionStatus.COMPLETED)
    _check(Delivery.objects.filter(vehicle=vehicle).count() == 1)
    _check(DeliveryItem.objects.filter(facility=facility).count() == 1)


@pytest.mark.django_db
def test_run_planning_cycle_queues_facility_using_min_safe_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Planning cycle should queue facilities using min-safe fallback reorder logic."""
    depot = DepotFactory()
    vehicle = VehicleFactory(depot=depot, is_available=True)
    facility = FacilityFactory(current_inventory=180.0, min_safe_inventory=220.0, dynamic_reorder_point=None)
    DepotFacilityAssignmentFactory(depot=depot, facility=facility)

    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "1")
    monkeypatch.setattr(
        "fuelsense.core.tasks.build_optimizer_request",
        lambda _depot, _facilities: (
            {
                "depot_lat": 24.7,
                "depot_lng": 46.7,
                "vehicles": [{"capacity": 1000.0, "cost_per_km": 2.0}],
                "stops": [
                    {
                        "facility_index": 1,
                        "demand": 40.0,
                        "time_window_start": 300,
                        "time_window_end": 900,
                        "service_time": 30,
                    },
                ],
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
                        "stops": [{"facility_index": 1, "demand": 40.0, "arrival_min": 360, "sequence": 1}],
                        "distance_km": 24.0,
                        "cost": 48.0,
                    },
                ],
                "total_distance_km": 24.0,
                "total_cost": 48.0,
                "vehicles_used": 1,
                "solver_time_ms": 20.0,
                "baseline_cost": 60.0,
                "cost_reduction_pct": 20.0,
            }

    class _Client:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> None:
            _ = exc_type, exc, tb

        def post(self, _url: str, json: dict[str, object]) -> _Response:
            _check("stops" in json)
            return _Response()

    monkeypatch.setattr("fuelsense.core.tasks.httpx.Client", lambda _timeout: _Client())

    result = tasks.run_planning_cycle()
    _check(result["deliveries_created"] == 1)
    _check(Delivery.objects.filter(vehicle=vehicle).count() == 1)


@pytest.mark.django_db
def test_run_planning_cycle_updates_existing_queued_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Planning cycle should update an existing queued cycle when cycle id is provided."""
    depot = DepotFactory()
    VehicleFactory(depot=depot, is_available=True)
    facility = FacilityFactory(current_inventory=50.0, dynamic_reorder_point=120.0)
    DepotFacilityAssignmentFactory(depot=depot, facility=facility)

    cycle = PlanningCycle.objects.create(
        trigger_type=PlanningCycle.TriggerType.MANUAL,
        status=PlanningCycle.ExecutionStatus.QUEUED,
        facilities_in_queue=0,
        deliveries_created=0,
        total_distance_km=0.0,
        total_cost=0.0,
        solver_time_ms=0.0,
        baseline_cost=0.0,
        cost_reduction_pct=0.0,
    )

    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "1")
    monkeypatch.setattr(
        "fuelsense.core.tasks.build_optimizer_request",
        lambda _depot, _facilities: (
            {
                "depot_lat": 24.7,
                "depot_lng": 46.7,
                "vehicles": [{"capacity": 1000.0, "cost_per_km": 2.0}],
                "stops": [
                    {
                        "facility_index": 1,
                        "demand": 70.0,
                        "time_window_start": 300,
                        "time_window_end": 900,
                        "service_time": 30,
                    },
                ],
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
                    },
                ],
                "total_distance_km": 24.0,
                "total_cost": 48.0,
                "vehicles_used": 1,
                "solver_time_ms": 20.0,
                "baseline_cost": 60.0,
                "cost_reduction_pct": 20.0,
            }

    class _Client:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> None:
            _ = exc_type, exc, tb

        def post(self, _url: str, json: dict[str, object]) -> _Response:
            _check("stops" in json)
            return _Response()

    monkeypatch.setattr("fuelsense.core.tasks.httpx.Client", lambda _timeout: _Client())

    result = tasks.run_planning_cycle(_pk(cycle))
    _check(result["planning_cycles"] == 1)
    cycle.refresh_from_db()
    _check(cycle.status == PlanningCycle.ExecutionStatus.COMPLETED)
    _check(cycle.deliveries_created == 1)


@pytest.mark.django_db
def test_run_planning_cycle_observes_cycle_trigger_type_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Planning stage metrics should be emitted with the cycle trigger type label."""
    cycle = PlanningCycle.objects.create(
        trigger_type=PlanningCycle.TriggerType.MANUAL,
        status=PlanningCycle.ExecutionStatus.QUEUED,
        facilities_in_queue=0,
        deliveries_created=0,
        total_distance_km=0.0,
        total_cost=0.0,
        solver_time_ms=0.0,
        baseline_cost=0.0,
        cost_reduction_pct=0.0,
    )

    observed: list[tuple[str, str]] = []

    def _observe(stage: str, _duration: float, trigger_type: str) -> None:
        observed.append((stage, trigger_type))

    monkeypatch.setattr("fuelsense.core.tasks.observe_planning_stage", _observe)
    result = tasks.run_planning_cycle(_pk(cycle))

    _check(result["queued"] == 0)
    _check(("queue_selection", PlanningCycle.TriggerType.MANUAL) in observed)


@pytest.mark.django_db
def test_run_planning_cycle_adhoc_metrics_default_to_scheduled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ad-hoc planning runs should emit metrics under scheduled trigger type."""
    observed: list[tuple[str, str]] = []

    def _observe(stage: str, _duration: float, trigger_type: str) -> None:
        observed.append((stage, trigger_type))

    monkeypatch.setattr("fuelsense.core.tasks.observe_planning_stage", _observe)
    result = tasks.run_planning_cycle()

    _check(result["queued"] == 0)
    _check(("queue_selection", PlanningCycle.TriggerType.SCHEDULED) in observed)


@pytest.mark.django_db
def test_run_planning_cycle_idempotent_when_not_queued() -> None:
    """Planning cycle processing should be idempotent for non-queued cycles."""
    cycle = PlanningCycle.objects.create(
        trigger_type=PlanningCycle.TriggerType.MANUAL,
        status=PlanningCycle.ExecutionStatus.COMPLETED,
        facilities_in_queue=1,
        deliveries_created=0,
        total_distance_km=0.0,
        total_cost=0.0,
        solver_time_ms=0.0,
        baseline_cost=0.0,
        cost_reduction_pct=0.0,
    )

    result = tasks.run_planning_cycle(_pk(cycle))
    _check(result["already_processed"] is True)
    _check(result["status"] == PlanningCycle.ExecutionStatus.COMPLETED)


@pytest.mark.django_db
def test_run_planning_cycle_cycle_not_found() -> None:
    """Planning cycle invocation should return not-found payload for missing cycle ids."""
    result = tasks.run_planning_cycle(999999)
    _check(result["error"] == "cycle_not_found")


@pytest.mark.django_db
def test_run_planning_cycle_marks_empty_cycle_completed() -> None:
    """Queued cycles with no actionable facilities should complete with zero deliveries."""
    cycle = PlanningCycle.objects.create(
        trigger_type=PlanningCycle.TriggerType.MANUAL,
        status=PlanningCycle.ExecutionStatus.QUEUED,
        facilities_in_queue=1,
        deliveries_created=1,
        total_distance_km=5.0,
        total_cost=5.0,
        solver_time_ms=5.0,
        baseline_cost=5.0,
        cost_reduction_pct=1.0,
    )
    result = tasks.run_planning_cycle(_pk(cycle))
    _check(result["queued"] == 0)
    cycle.refresh_from_db()
    _check(cycle.status == PlanningCycle.ExecutionStatus.COMPLETED)
    _check(cycle.deliveries_created == 0)


@pytest.mark.django_db
def test_run_planning_cycle_marks_failed_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Planning cycle should transition to failed status when orchestration raises."""
    cycle = PlanningCycle.objects.create(
        trigger_type=PlanningCycle.TriggerType.MANUAL,
        status=PlanningCycle.ExecutionStatus.QUEUED,
        facilities_in_queue=0,
        deliveries_created=0,
        total_distance_km=0.0,
        total_cost=0.0,
        solver_time_ms=0.0,
        baseline_cost=0.0,
        cost_reduction_pct=0.0,
    )
    facility = FacilityFactory(current_inventory=0.0, dynamic_reorder_point=1.0)
    DepotFacilityAssignmentFactory(facility=facility)
    monkeypatch.setattr(
        "fuelsense.core.tasks.build_optimizer_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = tasks.run_planning_cycle(_pk(cycle))
    _check(result["status"] == PlanningCycle.ExecutionStatus.FAILED)
    cycle.refresh_from_db()
    _check(cycle.status == PlanningCycle.ExecutionStatus.FAILED)


@pytest.mark.django_db
def test_run_planning_cycle_parallel_partial_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parallel planning should support partial-success mode when one depot fails."""
    depot_ok = DepotFactory(latitude=24.7, longitude=46.7)
    depot_fail = DepotFactory(latitude=25.0, longitude=47.0)
    VehicleFactory(depot=depot_ok, is_available=True)
    VehicleFactory(depot=depot_fail, is_available=True)

    facility_ok = FacilityFactory(current_inventory=50.0, dynamic_reorder_point=120.0, latitude=24.71, longitude=46.71)
    facility_fail = FacilityFactory(
        current_inventory=40.0, dynamic_reorder_point=120.0, latitude=25.01, longitude=47.01,
    )
    DepotFacilityAssignmentFactory(depot=depot_ok, facility=facility_ok)
    DepotFacilityAssignmentFactory(depot=depot_fail, facility=facility_fail)

    monkeypatch.setenv("FUELSENSE_PLANNING_PARALLEL_DEPOTS", "1")
    monkeypatch.setenv("FUELSENSE_PLANNING_STRICT_DEPOT_SUCCESS", "0")
    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "0")

    def _run_optimizer(payload: dict[str, object], optimizer_client: object | None = None) -> dict[str, object]:
        _ = optimizer_client
        depot_lat_raw = payload.get("depot_lat", 0.0)
        depot_lat = float(depot_lat_raw) if isinstance(depot_lat_raw, (int, float, str)) else 0.0
        if depot_lat > DEPOT_LAT_FAILURE_THRESHOLD:
            msg = "optimizer exploded"
            raise RuntimeError(msg)
        return {
            "status": "optimal",
            "routes": [
                {
                    "vehicle_index": 0,
                    "stops": [{"facility_index": 1, "demand": 70.0, "arrival_min": 360, "sequence": 1}],
                    "distance_km": 24.0,
                    "cost": 48.0,
                },
            ],
            "total_distance_km": 24.0,
            "total_cost": 48.0,
            "vehicles_used": 1,
            "solver_time_ms": 20.0,
            "baseline_cost": 60.0,
            "cost_reduction_pct": 20.0,
        }

    monkeypatch.setattr("fuelsense.core.tasks._run_optimizer", _run_optimizer)

    result = tasks.run_planning_cycle()
    _check(result["deliveries_created"] == 1)
    _check(result["failed_depots"] == 1)
    _check(result["partial_success"] is True)
    _check(DeliveryItem.objects.filter(facility=facility_ok).count() == 1)
    _check(DeliveryItem.objects.filter(facility=facility_fail).count() == 0)


@pytest.mark.django_db
def test_run_planning_cycle_strict_mode_aborts_on_depot_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict planning mode should fail the cycle when any depot optimization fails."""
    depot_ok = DepotFactory(latitude=24.7, longitude=46.7)
    depot_fail = DepotFactory(latitude=25.0, longitude=47.0)
    VehicleFactory(depot=depot_ok, is_available=True)
    VehicleFactory(depot=depot_fail, is_available=True)

    facility_ok = FacilityFactory(current_inventory=50.0, dynamic_reorder_point=120.0, latitude=24.71, longitude=46.71)
    facility_fail = FacilityFactory(
        current_inventory=40.0, dynamic_reorder_point=120.0, latitude=25.01, longitude=47.01,
    )
    DepotFacilityAssignmentFactory(depot=depot_ok, facility=facility_ok)
    DepotFacilityAssignmentFactory(depot=depot_fail, facility=facility_fail)

    monkeypatch.setenv("FUELSENSE_PLANNING_PARALLEL_DEPOTS", "1")
    monkeypatch.setenv("FUELSENSE_PLANNING_STRICT_DEPOT_SUCCESS", "1")
    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "0")

    def _run_optimizer(payload: dict[str, object], optimizer_client: object | None = None) -> dict[str, object]:
        _ = optimizer_client
        depot_lat_raw = payload.get("depot_lat", 0.0)
        depot_lat = float(depot_lat_raw) if isinstance(depot_lat_raw, (int, float, str)) else 0.0
        if depot_lat > DEPOT_LAT_FAILURE_THRESHOLD:
            msg = "optimizer exploded"
            raise RuntimeError(msg)
        return {
            "status": "optimal",
            "routes": [
                {
                    "vehicle_index": 0,
                    "stops": [{"facility_index": 1, "demand": 70.0, "arrival_min": 360, "sequence": 1}],
                    "distance_km": 24.0,
                    "cost": 48.0,
                },
            ],
            "total_distance_km": 24.0,
            "total_cost": 48.0,
            "vehicles_used": 1,
            "solver_time_ms": 20.0,
            "baseline_cost": 60.0,
            "cost_reduction_pct": 20.0,
        }

    monkeypatch.setattr("fuelsense.core.tasks._run_optimizer", _run_optimizer)

    result = tasks.run_planning_cycle()
    _check(result["status"] == PlanningCycle.ExecutionStatus.FAILED)
    _check(result["deliveries_created"] == 0)
    _check(Delivery.objects.count() == 0)
    _check(DeliveryItem.objects.count() == 0)


@pytest.mark.django_db
def test_run_planning_cycle_serial_optimizer_exception_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serial planning path should surface optimizer exceptions as failed depot outcomes."""
    depot = DepotFactory()
    VehicleFactory(depot=depot, is_available=True)
    facility = FacilityFactory(current_inventory=50.0, dynamic_reorder_point=120.0)
    DepotFacilityAssignmentFactory(depot=depot, facility=facility)

    monkeypatch.setenv("FUELSENSE_PLANNING_PARALLEL_DEPOTS", "0")
    monkeypatch.setenv("FUELSENSE_PLANNING_STRICT_DEPOT_SUCCESS", "0")
    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "0")
    monkeypatch.setattr(
        "fuelsense.core.tasks._run_optimizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("serial optimizer boom")),
    )

    result = tasks.run_planning_cycle()
    _check(result["failed_depots"] >= 1)
    _check(result["deliveries_created"] == 0)


@pytest.mark.django_db
def test_run_planning_cycle_parallel_many_depots_stress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parallel planning should process many depots and materialize expected delivery volume."""
    monkeypatch.setenv("FUELSENSE_PLANNING_PARALLEL_DEPOTS", "1")
    monkeypatch.setenv("FUELSENSE_PLANNING_PARALLEL_WORKERS", "8")
    monkeypatch.setenv("FUELSENSE_PLANNING_STRICT_DEPOT_SUCCESS", "1")
    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "0")

    depots = []
    facilities = []
    for idx in range(20):
        depot = DepotFactory(latitude=24.0 + (idx * 0.1), longitude=46.0 + (idx * 0.1))
        VehicleFactory(depot=depot, is_available=True)
        facility = FacilityFactory(
            current_inventory=10.0,
            dynamic_reorder_point=100.0,
            latitude=24.01 + (idx * 0.1),
            longitude=46.01 + (idx * 0.1),
        )
        DepotFacilityAssignmentFactory(depot=depot, facility=facility)
        depots.append(depot)
        facilities.append(facility)

    def _run_optimizer(payload: dict[str, object], optimizer_client: object | None = None) -> dict[str, object]:
        _ = optimizer_client, payload
        return {
            "status": "optimal",
            "routes": [
                {
                    "vehicle_index": 0,
                    "stops": [{"facility_index": 1, "demand": 90.0, "arrival_min": 240, "sequence": 1}],
                    "distance_km": 20.0,
                    "cost": 40.0,
                },
            ],
            "total_distance_km": 20.0,
            "total_cost": 40.0,
            "vehicles_used": 1,
            "solver_time_ms": 15.0,
            "baseline_cost": 50.0,
            "cost_reduction_pct": 20.0,
        }

    monkeypatch.setattr("fuelsense.core.tasks._run_optimizer", _run_optimizer)

    result = tasks.run_planning_cycle()
    _check(result["failed_depots"] == 0)
    _check(result["deliveries_created"] == EXPECTED_ENTITY_COUNT_20)
    _check(Delivery.objects.count() == EXPECTED_ENTITY_COUNT_20)
    _check(DeliveryItem.objects.count() == EXPECTED_ENTITY_COUNT_20)


@pytest.mark.django_db
def test_run_planning_cycle_parallel_reuses_single_optimizer_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parallel planning should reuse one remote optimizer client across depot jobs."""
    depot_a = DepotFactory(latitude=24.0, longitude=46.0)
    depot_b = DepotFactory(latitude=25.0, longitude=47.0)
    VehicleFactory(depot=depot_a, is_available=True)
    VehicleFactory(depot=depot_b, is_available=True)
    facility_a = FacilityFactory(current_inventory=40.0, dynamic_reorder_point=140.0, latitude=24.1, longitude=46.1)
    facility_b = FacilityFactory(current_inventory=40.0, dynamic_reorder_point=140.0, latitude=25.1, longitude=47.1)
    DepotFacilityAssignmentFactory(depot=depot_a, facility=facility_a)
    DepotFacilityAssignmentFactory(depot=depot_b, facility=facility_b)

    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "1")
    monkeypatch.setenv("FUELSENSE_PLANNING_PARALLEL_DEPOTS", "1")
    monkeypatch.setenv("FUELSENSE_PLANNING_STRICT_DEPOT_SUCCESS", "0")

    clients: list[object] = []

    class _Client:
        def __init__(self, timeout: float) -> None:
            _ = timeout
            self.closed = False
            clients.append(self)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("fuelsense.core.tasks.httpx.Client", _Client)

    seen_client_ids: set[int] = set()

    def _run_optimizer(payload: dict[str, object], optimizer_client: object | None = None) -> dict[str, object]:
        _ = payload
        _check(optimizer_client is not None)
        _check(clients and optimizer_client is clients[0])
        seen_client_ids.add(id(optimizer_client))
        return {
            "status": "optimal",
            "routes": [
                {
                    "vehicle_index": 0,
                    "stops": [{"facility_index": 1, "demand": 90.0, "arrival_min": 240, "sequence": 1}],
                    "distance_km": 20.0,
                    "cost": 40.0,
                },
            ],
            "total_distance_km": 20.0,
            "total_cost": 40.0,
            "vehicles_used": 1,
            "solver_time_ms": 15.0,
            "baseline_cost": 50.0,
            "cost_reduction_pct": 20.0,
        }

    monkeypatch.setattr("fuelsense.core.tasks._run_optimizer", _run_optimizer)

    result = tasks.run_planning_cycle()
    _check(result["deliveries_created"] == EXPECTED_PLANNING_CYCLE_COUNT_2)
    _check(len(clients) == 1)
    _check(len(seen_client_ids) == 1)
    _check(getattr(clients[0], "closed", False) is True)


@pytest.mark.django_db
def test_run_planning_cycle_parallel_closes_optimizer_client_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parallel planning should close optimizer client even when one depot path errors."""
    depot_a = DepotFactory(latitude=24.0, longitude=46.0)
    depot_b = DepotFactory(latitude=25.0, longitude=47.0)
    VehicleFactory(depot=depot_a, is_available=True)
    VehicleFactory(depot=depot_b, is_available=True)
    facility_a = FacilityFactory(current_inventory=40.0, dynamic_reorder_point=140.0, latitude=24.1, longitude=46.1)
    facility_b = FacilityFactory(current_inventory=40.0, dynamic_reorder_point=140.0, latitude=25.1, longitude=47.1)
    DepotFacilityAssignmentFactory(depot=depot_a, facility=facility_a)
    DepotFacilityAssignmentFactory(depot=depot_b, facility=facility_b)

    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "1")
    monkeypatch.setenv("FUELSENSE_PLANNING_PARALLEL_DEPOTS", "1")
    monkeypatch.setenv("FUELSENSE_PLANNING_STRICT_DEPOT_SUCCESS", "0")

    clients: list[object] = []

    class _Client:
        def __init__(self, timeout: float) -> None:
            _ = timeout
            self.closed = False
            clients.append(self)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("fuelsense.core.tasks.httpx.Client", _Client)

    def _run_optimizer(payload: dict[str, object], optimizer_client: object | None = None) -> dict[str, object]:
        _check(optimizer_client is not None)
        depot_lat_raw = payload.get("depot_lat", 0.0)
        depot_lat = float(depot_lat_raw) if isinstance(depot_lat_raw, (int, float, str)) else 0.0
        if depot_lat > DEPOT_LAT_PARALLEL_FAILURE_THRESHOLD:
            msg = "parallel optimizer boom"
            raise RuntimeError(msg)
        return {
            "status": "optimal",
            "routes": [
                {
                    "vehicle_index": 0,
                    "stops": [{"facility_index": 1, "demand": 90.0, "arrival_min": 240, "sequence": 1}],
                    "distance_km": 20.0,
                    "cost": 40.0,
                },
            ],
            "total_distance_km": 20.0,
            "total_cost": 40.0,
            "vehicles_used": 1,
            "solver_time_ms": 15.0,
            "baseline_cost": 50.0,
            "cost_reduction_pct": 20.0,
        }

    monkeypatch.setattr("fuelsense.core.tasks._run_optimizer", _run_optimizer)

    result = tasks.run_planning_cycle()
    _check(result["failed_depots"] >= 1)
    _check(len(clients) == 1)
    _check(getattr(clients[0], "closed", False) is True)


@pytest.mark.django_db
def test_daily_tick_dispatches_chord_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Daily tick should dispatch the expected chord/chain orchestration graph."""
    FacilityFactory(is_active=True)
    monkeypatch.setenv("FUELSENSE_ENABLE_DAILY_TICK", "1")

    captured: dict[str, object] = {}

    class _Sig:
        def __init__(self, name: str) -> None:
            self.name = name

        def __or__(self, _other: object) -> _Sig:
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
    monkeypatch.setattr(_Sig, "delay", lambda _self: captured.setdefault("dispatched", True), raising=False)

    # Provide si on our fake task signatures.
    _Sig.si = lambda _self, *_args, **_kwargs: _Sig(f"{_self.name}.si")  # type: ignore[attr-defined]

    result = tasks.daily_tick()
    _check(result["status"] == "dispatched")
    _check(captured.get("header_len") == 1)
    _check(captured.get("dispatched") is True)


@pytest.mark.django_db
def test_trigger_emergency_delivery_creates_emergency_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Emergency trigger should create an emergency planning cycle and deliveries."""
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
                "stops": [
                    {
                        "facility_index": 1,
                        "demand": 90.0,
                        "time_window_start": 300,
                        "time_window_end": 900,
                        "service_time": 30,
                    },
                ],
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
                    },
                ],
                "total_distance_km": 20.0,
                "total_cost": 40.0,
                "vehicles_used": 1,
                "solver_time_ms": 15.0,
                "baseline_cost": 50.0,
                "cost_reduction_pct": 20.0,
            }

    class _Client:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> None:
            _ = exc_type, exc, tb

        def post(self, _url: str, json: dict[str, object]) -> _Response:
            _check("stops" in json)
            return _Response()

    monkeypatch.setattr("fuelsense.core.tasks.httpx.Client", lambda _timeout: _Client())

    result = tasks.trigger_emergency_delivery(facility.id)
    _check(result["exists"] is True)
    _check(result["deliveries_created"] == 1)
    cycle = PlanningCycle.objects.get(trigger_type=PlanningCycle.TriggerType.EMERGENCY)
    _check(cycle.status == PlanningCycle.ExecutionStatus.COMPLETED)


@pytest.mark.django_db
def test_run_emergency_planning_cycle_uses_single_parent_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Emergency planning should write deliveries under the provided parent cycle."""
    depot = DepotFactory()
    VehicleFactory(depot=depot, is_available=True)
    facility = FacilityFactory(current_inventory=40.0, dynamic_reorder_point=140.0)
    DepotFacilityAssignmentFactory(depot=depot, facility=facility)

    cycle = PlanningCycle.objects.create(
        trigger_type=PlanningCycle.TriggerType.EMERGENCY,
        status=PlanningCycle.ExecutionStatus.QUEUED,
        facilities_in_queue=1,
        deliveries_created=0,
        total_distance_km=0.0,
        total_cost=0.0,
        solver_time_ms=0.0,
        baseline_cost=0.0,
        cost_reduction_pct=0.0,
    )

    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "1")
    monkeypatch.setattr(
        "fuelsense.core.tasks.build_optimizer_request",
        lambda _depot, _facilities: (
            {
                "depot_lat": 24.7,
                "depot_lng": 46.7,
                "vehicles": [{"capacity": 1000.0, "cost_per_km": 2.0}],
                "stops": [
                    {
                        "facility_index": 1,
                        "demand": 90.0,
                        "time_window_start": 300,
                        "time_window_end": 900,
                        "service_time": 30,
                    },
                ],
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
                    },
                ],
                "total_distance_km": 20.0,
                "total_cost": 40.0,
                "vehicles_used": 1,
                "solver_time_ms": 15.0,
                "baseline_cost": 50.0,
                "cost_reduction_pct": 20.0,
            }

    class _Client:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> None:
            _ = exc_type, exc, tb

        def post(self, _url: str, json: dict[str, object]) -> _Response:
            _check("stops" in json)
            return _Response()

    monkeypatch.setattr("fuelsense.core.tasks.httpx.Client", lambda _timeout: _Client())

    result = tasks.run_emergency_planning_cycle(_pk(cycle), [facility.id])
    _check(result["deliveries_created"] == 1)
    cycle.refresh_from_db()
    _check(cycle.status == PlanningCycle.ExecutionStatus.COMPLETED)
    _check(Delivery.objects.filter(created_by_planning_cycle=cycle).count() == 1)


@pytest.mark.django_db
def test_run_emergency_planning_cycle_not_found() -> None:
    """Emergency planning should return not-found payload for missing cycle ids."""
    result = tasks.run_emergency_planning_cycle(999999, [1])
    _check(result["error"] == "cycle_not_found")


@pytest.mark.django_db
def test_run_emergency_planning_cycle_idempotent_if_not_queued() -> None:
    """Emergency planning should be idempotent for cycles already processed."""
    cycle = PlanningCycle.objects.create(
        trigger_type=PlanningCycle.TriggerType.EMERGENCY,
        status=PlanningCycle.ExecutionStatus.COMPLETED,
        facilities_in_queue=1,
        deliveries_created=0,
        total_distance_km=0.0,
        total_cost=0.0,
        solver_time_ms=0.0,
        baseline_cost=0.0,
        cost_reduction_pct=0.0,
    )
    result = tasks.run_emergency_planning_cycle(_pk(cycle), [1])
    _check(result["already_processed"] is True)


@pytest.mark.django_db
def test_run_emergency_planning_cycle_marks_failed_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Emergency planning should mark cycle failed when trigger task raises."""
    cycle = PlanningCycle.objects.create(
        trigger_type=PlanningCycle.TriggerType.EMERGENCY,
        status=PlanningCycle.ExecutionStatus.QUEUED,
        facilities_in_queue=1,
        deliveries_created=0,
        total_distance_km=0.0,
        total_cost=0.0,
        solver_time_ms=0.0,
        baseline_cost=0.0,
        cost_reduction_pct=0.0,
    )
    monkeypatch.setattr(
        "fuelsense.core.tasks.trigger_emergency_delivery",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    result = tasks.run_emergency_planning_cycle(_pk(cycle), [1])
    _check(result["status"] == PlanningCycle.ExecutionStatus.FAILED)
    cycle.refresh_from_db()
    _check(cycle.status == PlanningCycle.ExecutionStatus.FAILED)


@pytest.mark.django_db
def test_run_emergency_planning_cycle_partial_success_when_not_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-strict emergency mode should allow partial success when some facilities fail."""
    depot = DepotFactory()
    VehicleFactory(depot=depot, is_available=True)
    facility = FacilityFactory(current_inventory=40.0, dynamic_reorder_point=140.0)
    DepotFacilityAssignmentFactory(depot=depot, facility=facility)

    cycle = PlanningCycle.objects.create(
        trigger_type=PlanningCycle.TriggerType.EMERGENCY,
        status=PlanningCycle.ExecutionStatus.QUEUED,
        facilities_in_queue=2,
        deliveries_created=0,
        total_distance_km=0.0,
        total_cost=0.0,
        solver_time_ms=0.0,
        baseline_cost=0.0,
        cost_reduction_pct=0.0,
    )

    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "0")
    monkeypatch.setenv("FUELSENSE_EMERGENCY_STRICT_FACILITY_SUCCESS", "0")

    result = tasks.run_emergency_planning_cycle(_pk(cycle), [facility.id, 999999])
    _check(result["deliveries_created"] == 1)
    _check(result["failed_facilities"] == 1)
    _check(result["partial_success"] is True)
    cycle.refresh_from_db()
    _check(cycle.status == PlanningCycle.ExecutionStatus.COMPLETED)


@pytest.mark.django_db
def test_run_emergency_planning_cycle_observes_emergency_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Emergency planning should emit stage metrics tagged with emergency trigger type."""
    cycle = PlanningCycle.objects.create(
        trigger_type=PlanningCycle.TriggerType.EMERGENCY,
        status=PlanningCycle.ExecutionStatus.QUEUED,
        facilities_in_queue=1,
        deliveries_created=0,
        total_distance_km=0.0,
        total_cost=0.0,
        solver_time_ms=0.0,
        baseline_cost=0.0,
        cost_reduction_pct=0.0,
    )

    observed: list[tuple[str, str]] = []

    def _observe(stage: str, _duration: float, trigger_type: str) -> None:
        observed.append((stage, trigger_type))

    monkeypatch.setattr("fuelsense.core.tasks.observe_planning_stage", _observe)
    monkeypatch.setenv("FUELSENSE_EMERGENCY_STRICT_FACILITY_SUCCESS", "1")

    result = tasks.run_emergency_planning_cycle(_pk(cycle), [999999])
    _check(result["status"] == PlanningCycle.ExecutionStatus.FAILED)
    _check(any(trigger == str(PlanningCycle.TriggerType.EMERGENCY) for _, trigger in observed))


@pytest.mark.django_db
def test_run_emergency_planning_cycle_parallel_optimizer_exception_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parallel emergency planning should continue under partial-success when one facility fails."""
    depot_a = DepotFactory(latitude=24.0, longitude=46.0)
    depot_b = DepotFactory(latitude=25.0, longitude=47.0)
    VehicleFactory(depot=depot_a, is_available=True)
    VehicleFactory(depot=depot_b, is_available=True)
    facility_a = FacilityFactory(current_inventory=40.0, dynamic_reorder_point=140.0, latitude=24.1, longitude=46.1)
    facility_b = FacilityFactory(current_inventory=40.0, dynamic_reorder_point=140.0, latitude=25.1, longitude=47.1)
    DepotFacilityAssignmentFactory(depot=depot_a, facility=facility_a)
    DepotFacilityAssignmentFactory(depot=depot_b, facility=facility_b)
    cycle = PlanningCycle.objects.create(
        trigger_type=PlanningCycle.TriggerType.EMERGENCY,
        status=PlanningCycle.ExecutionStatus.QUEUED,
        facilities_in_queue=2,
        deliveries_created=0,
        total_distance_km=0.0,
        total_cost=0.0,
        solver_time_ms=0.0,
        baseline_cost=0.0,
        cost_reduction_pct=0.0,
    )

    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "0")
    monkeypatch.setenv("FUELSENSE_EMERGENCY_PARALLEL", "1")
    monkeypatch.setenv("FUELSENSE_EMERGENCY_STRICT_FACILITY_SUCCESS", "0")

    def _run_optimizer(payload: dict[str, object], optimizer_client: object | None = None) -> dict[str, object]:
        _ = optimizer_client
        depot_lat_raw = payload.get("depot_lat", 0.0)
        depot_lat = float(depot_lat_raw) if isinstance(depot_lat_raw, (int, float, str)) else 0.0
        if depot_lat > DEPOT_LAT_PARALLEL_FAILURE_THRESHOLD:
            msg = "parallel emergency optimizer boom"
            raise RuntimeError(msg)
        return {
            "status": "optimal",
            "routes": [
                {
                    "vehicle_index": 0,
                    "stops": [{"facility_index": 1, "demand": 90.0, "arrival_min": 240, "sequence": 1}],
                    "distance_km": 20.0,
                    "cost": 40.0,
                },
            ],
            "total_distance_km": 20.0,
            "total_cost": 40.0,
            "vehicles_used": 1,
            "solver_time_ms": 15.0,
            "baseline_cost": 50.0,
            "cost_reduction_pct": 20.0,
        }

    monkeypatch.setattr("fuelsense.core.tasks._run_optimizer", _run_optimizer)

    result = tasks.run_emergency_planning_cycle(_pk(cycle), [facility_a.id, facility_b.id])
    _check(result["deliveries_created"] == 1)
    _check(result["failed_facilities"] == 1)
    _check(result["partial_success"] is True)


@pytest.mark.django_db
def test_run_emergency_planning_cycle_parallel_reuses_single_optimizer_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parallel emergency planning should reuse and close a single optimizer client."""
    depot_a = DepotFactory(latitude=24.0, longitude=46.0)
    depot_b = DepotFactory(latitude=25.0, longitude=47.0)
    VehicleFactory(depot=depot_a, is_available=True)
    VehicleFactory(depot=depot_b, is_available=True)
    facility_a = FacilityFactory(current_inventory=40.0, dynamic_reorder_point=140.0, latitude=24.1, longitude=46.1)
    facility_b = FacilityFactory(current_inventory=40.0, dynamic_reorder_point=140.0, latitude=25.1, longitude=47.1)
    DepotFacilityAssignmentFactory(depot=depot_a, facility=facility_a)
    DepotFacilityAssignmentFactory(depot=depot_b, facility=facility_b)

    cycle = PlanningCycle.objects.create(
        trigger_type=PlanningCycle.TriggerType.EMERGENCY,
        status=PlanningCycle.ExecutionStatus.QUEUED,
        facilities_in_queue=2,
        deliveries_created=0,
        total_distance_km=0.0,
        total_cost=0.0,
        solver_time_ms=0.0,
        baseline_cost=0.0,
        cost_reduction_pct=0.0,
    )

    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "1")
    monkeypatch.setenv("FUELSENSE_EMERGENCY_PARALLEL", "1")
    monkeypatch.setenv("FUELSENSE_EMERGENCY_STRICT_FACILITY_SUCCESS", "0")

    clients: list[object] = []

    class _Client:
        def __init__(self, timeout: float) -> None:
            _ = timeout
            self.closed = False
            clients.append(self)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("fuelsense.core.tasks.httpx.Client", _Client)

    seen_client_ids: set[int] = set()

    def _run_optimizer(payload: dict[str, object], optimizer_client: object | None = None) -> dict[str, object]:
        _ = payload
        _check(optimizer_client is not None)
        _check(clients and optimizer_client is clients[0])
        seen_client_ids.add(id(optimizer_client))
        return {
            "status": "optimal",
            "routes": [
                {
                    "vehicle_index": 0,
                    "stops": [{"facility_index": 1, "demand": 90.0, "arrival_min": 240, "sequence": 1}],
                    "distance_km": 20.0,
                    "cost": 40.0,
                },
            ],
            "total_distance_km": 20.0,
            "total_cost": 40.0,
            "vehicles_used": 1,
            "solver_time_ms": 15.0,
            "baseline_cost": 50.0,
            "cost_reduction_pct": 20.0,
        }

    monkeypatch.setattr("fuelsense.core.tasks._run_optimizer", _run_optimizer)

    result = tasks.run_emergency_planning_cycle(_pk(cycle), [facility_a.id, facility_b.id])
    _check(result["deliveries_created"] == EXPECTED_PLANNING_CYCLE_COUNT_2)
    _check(len(clients) == 1)
    _check(len(seen_client_ids) == 1)
    _check(getattr(clients[0], "closed", False) is True)


@pytest.mark.django_db
def test_run_emergency_planning_cycle_strict_persist_failure_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict emergency mode should fail the cycle when route persistence fails."""
    depot = DepotFactory()
    VehicleFactory(depot=depot, is_available=True)
    facility = FacilityFactory(current_inventory=40.0, dynamic_reorder_point=140.0)
    DepotFacilityAssignmentFactory(depot=depot, facility=facility)
    cycle = PlanningCycle.objects.create(
        trigger_type=PlanningCycle.TriggerType.EMERGENCY,
        status=PlanningCycle.ExecutionStatus.QUEUED,
        facilities_in_queue=1,
        deliveries_created=0,
        total_distance_km=0.0,
        total_cost=0.0,
        solver_time_ms=0.0,
        baseline_cost=0.0,
        cost_reduction_pct=0.0,
    )

    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "0")
    monkeypatch.setenv("FUELSENSE_EMERGENCY_STRICT_FACILITY_SUCCESS", "1")
    monkeypatch.setattr(
        "fuelsense.core.tasks._materialize_delivery_routes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("persist boom")),
    )

    result = tasks.run_emergency_planning_cycle(_pk(cycle), [facility.id])
    _check(result["status"] == PlanningCycle.ExecutionStatus.FAILED)
    _check(result["error"] == "strict_mode_persist_failure")
    cycle.refresh_from_db()
    _check(cycle.status == PlanningCycle.ExecutionStatus.FAILED)


@pytest.mark.django_db
def test_run_emergency_planning_cycle_strict_failure_with_missing_assignment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict emergency mode should fail the cycle on missing facility assignment."""
    depot = DepotFactory()
    VehicleFactory(depot=depot, is_available=True)
    facility = FacilityFactory(current_inventory=40.0, dynamic_reorder_point=140.0)
    DepotFacilityAssignmentFactory(depot=depot, facility=facility)

    cycle = PlanningCycle.objects.create(
        trigger_type=PlanningCycle.TriggerType.EMERGENCY,
        status=PlanningCycle.ExecutionStatus.QUEUED,
        facilities_in_queue=2,
        deliveries_created=0,
        total_distance_km=0.0,
        total_cost=0.0,
        solver_time_ms=0.0,
        baseline_cost=0.0,
        cost_reduction_pct=0.0,
    )

    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "0")
    monkeypatch.setenv("FUELSENSE_EMERGENCY_STRICT_FACILITY_SUCCESS", "1")

    result = tasks.run_emergency_planning_cycle(_pk(cycle), [facility.id, 999999])
    _check(result["status"] == PlanningCycle.ExecutionStatus.FAILED)
    _check(result["deliveries_created"] == 0)
    cycle.refresh_from_db()
    _check(cycle.status == PlanningCycle.ExecutionStatus.FAILED)
    _check(Delivery.objects.filter(created_by_planning_cycle=cycle).count() == 0)


@pytest.mark.django_db
def test_trigger_emergency_delivery_missing_assignment() -> None:
    """Emergency trigger should return exists=false for unknown facilities."""
    result = tasks.trigger_emergency_delivery(999999)
    _check(result["exists"] is False)


@pytest.mark.django_db
def test_trigger_emergency_delivery_skips_invalid_route_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Emergency trigger should skip malformed route entries without crashing."""
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
                "stops": [
                    {
                        "facility_index": 1,
                        "demand": 90.0,
                        "time_window_start": 300,
                        "time_window_end": 900,
                        "service_time": 30,
                    },
                ],
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
                        "vehicle_index": 9,
                        "stops": ["bad-stop", {"facility_index": 404, "demand": 1.0, "arrival_min": 30, "sequence": 1}],
                        "distance_km": 20.0,
                        "cost": 40.0,
                    },
                ],
                "total_distance_km": 20.0,
                "total_cost": 40.0,
                "vehicles_used": 1,
                "solver_time_ms": 15.0,
                "baseline_cost": 50.0,
                "cost_reduction_pct": 20.0,
            }

    class _Client:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> None:
            _ = exc_type, exc, tb

        def post(self, _url: str, json: dict[str, object]) -> _Response:
            _check("stops" in json)
            return _Response()

    monkeypatch.setattr("fuelsense.core.tasks.httpx.Client", lambda _timeout: _Client())

    result = tasks.trigger_emergency_delivery(facility.id)
    _check(result["exists"] is True)
    _check(result["deliveries_created"] == 0)


@pytest.mark.django_db
def test_trigger_emergency_delivery_is_idempotent_for_same_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Emergency trigger should be idempotent when invoked repeatedly for same cycle."""
    depot = DepotFactory()
    VehicleFactory(depot=depot, is_available=True)
    facility = FacilityFactory(current_inventory=40.0, dynamic_reorder_point=140.0)
    DepotFacilityAssignmentFactory(depot=depot, facility=facility)
    cycle = PlanningCycle.objects.create(
        trigger_type=PlanningCycle.TriggerType.EMERGENCY,
        status=PlanningCycle.ExecutionStatus.RUNNING,
        facilities_in_queue=1,
        deliveries_created=0,
        total_distance_km=0.0,
        total_cost=0.0,
        solver_time_ms=0.0,
        baseline_cost=0.0,
        cost_reduction_pct=0.0,
    )

    monkeypatch.setenv("FUELSENSE_ENABLE_REMOTE_OPTIMIZER", "0")
    first = tasks.trigger_emergency_delivery(facility.id, cycle_id=_pk(cycle))
    second = tasks.trigger_emergency_delivery(facility.id, cycle_id=_pk(cycle))

    _check(first["deliveries_created"] == 1)
    _check(second["deliveries_created"] == 0)
    _check(Delivery.objects.filter(created_by_planning_cycle=cycle).count() == 1)
    _check(DeliveryItem.objects.filter(delivery__created_by_planning_cycle=cycle).count() == 1)
