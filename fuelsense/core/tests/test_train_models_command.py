from __future__ import annotations

from typing import Any

import pytest
from django.core.management import call_command, get_commands
from django.core.management.base import CommandError

from fuelsense.core.tests.factories import FacilityFactory


class _AsyncResult:
    def __init__(self, task_id: str, payload: dict[str, object]) -> None:
        self.id = task_id
        self._payload = payload

    def get(self, timeout: int) -> dict[str, object]:
        _ = timeout
        return self._payload


@pytest.mark.django_db
def test_train_models_command_is_registered() -> None:
    commands = get_commands()
    assert "train_models" in commands


@pytest.mark.django_db
def test_train_models_dry_run_does_not_dispatch(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    FacilityFactory(is_active=True)
    monkeypatch.setenv("FUELSENSE_ENABLE_TRAINING_TASKS", "1")

    def _no_dispatch(*args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        raise AssertionError("delay should not be called during dry-run")

    monkeypatch.setattr("fuelsense.core.management.commands.train_models.retrain_model.delay", _no_dispatch)

    call_command("train_models", "--dry-run")
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "demand:facility" in out
    assert "anomaly:global" in out


@pytest.mark.django_db
def test_train_models_enqueues_demand_and_anomaly(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    facility = FacilityFactory(is_active=True)
    monkeypatch.setenv("FUELSENSE_ENABLE_TRAINING_TASKS", "1")

    seen: list[tuple[int | None, str]] = []

    def _delay(facility_id: int | None, model_type: str) -> _AsyncResult:
        seen.append((facility_id, model_type))
        return _AsyncResult(task_id=f"task-{len(seen)}", payload={"status": "promoted"})

    monkeypatch.setattr("fuelsense.core.management.commands.train_models.retrain_model.delay", _delay)

    call_command("train_models")
    out = capsys.readouterr().out

    assert (facility.id, "DEMAND_FORECAST") in seen
    assert (None, "ANOMALY_DETECTOR") in seen
    assert "Enqueued" in out


@pytest.mark.django_db
def test_train_models_wait_raises_on_failed_task(monkeypatch: pytest.MonkeyPatch) -> None:
    FacilityFactory(is_active=True)
    monkeypatch.setenv("FUELSENSE_ENABLE_TRAINING_TASKS", "1")

    queue: list[_AsyncResult] = [
        _AsyncResult("task-1", {"status": "promoted"}),
        _AsyncResult("task-2", {"status": "failed"}),
    ]

    def _delay(facility_id: int | None, model_type: str) -> _AsyncResult:
        _ = facility_id, model_type
        return queue.pop(0)

    monkeypatch.setattr("fuelsense.core.management.commands.train_models.retrain_model.delay", _delay)

    with pytest.raises(CommandError):
        call_command("train_models", "--models", "all", "--wait")


@pytest.mark.django_db
def test_train_models_anomaly_ignores_facility_selectors(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    FacilityFactory(is_active=True)
    monkeypatch.setenv("FUELSENSE_ENABLE_TRAINING_TASKS", "1")

    seen: list[tuple[int | None, str]] = []

    def _delay(facility_id: int | None, model_type: str) -> _AsyncResult:
        seen.append((facility_id, model_type))
        return _AsyncResult(task_id="task-a", payload={"status": "promoted"})

    monkeypatch.setattr("fuelsense.core.management.commands.train_models.retrain_model.delay", _delay)

    call_command("train_models", "--models", "anomaly", "--facility-id", "123")
    out = capsys.readouterr().out

    assert seen == [(None, "ANOMALY_DETECTOR")]
    assert "ignored for anomaly training" in out


@pytest.mark.django_db
def test_train_models_respects_disabled_gate(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    FacilityFactory(is_active=True)
    monkeypatch.setenv("FUELSENSE_ENABLE_TRAINING_TASKS", "0")

    def _no_dispatch(*args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        raise AssertionError("delay should not be called when training is disabled")

    monkeypatch.setattr("fuelsense.core.management.commands.train_models.retrain_model.delay", _no_dispatch)

    call_command("train_models")
    out = capsys.readouterr().out
    assert "Training tasks are disabled" in out
