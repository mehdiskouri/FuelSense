"""Tests for train_models management command registration and dispatch behavior."""

from __future__ import annotations

import pytest
from django.core.management import call_command, get_commands
from django.core.management.base import CommandError

from fuelsense.core.tests.factories import FacilityFactory


def _check(condition: object, message: str | None = None) -> None:
    if not bool(condition):
        raise AssertionError(message if message is not None else "check failed")


class _AsyncResult:
    def __init__(self, task_id: str, payload: dict[str, object]) -> None:
        self.id = task_id
        self._payload = payload

    def get(self, timeout: int) -> dict[str, object]:
        _ = timeout
        return self._payload


@pytest.mark.django_db
def test_train_models_command_is_registered() -> None:
    """The command should be visible in Django's registered command set."""
    commands = get_commands()
    _check("train_models" in commands)


@pytest.mark.django_db
def test_train_models_dry_run_does_not_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dry-run should print planned work but never enqueue Celery tasks."""
    FacilityFactory(is_active=True)
    monkeypatch.setenv("FUELSENSE_ENABLE_TRAINING_TASKS", "1")

    def _no_dispatch(*args: object, **kwargs: object) -> None:
        _ = args, kwargs
        msg = "delay should not be called during dry-run"
        raise AssertionError(msg)

    monkeypatch.setattr("fuelsense.core.management.commands.train_models.retrain_model.delay", _no_dispatch)

    call_command("train_models", "--dry-run")
    out = capsys.readouterr().out
    _check("DRY-RUN" in out)
    _check("demand:facility" in out)
    _check("anomaly:global" in out)


@pytest.mark.django_db
def test_train_models_enqueues_demand_and_anomaly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Default run should enqueue both demand and anomaly training jobs."""
    facility = FacilityFactory(is_active=True)
    monkeypatch.setenv("FUELSENSE_ENABLE_TRAINING_TASKS", "1")

    seen: list[tuple[int | None, str]] = []

    def _delay(facility_id: int | None, model_type: str) -> _AsyncResult:
        seen.append((facility_id, model_type))
        return _AsyncResult(task_id=f"task-{len(seen)}", payload={"status": "promoted"})

    monkeypatch.setattr("fuelsense.core.management.commands.train_models.retrain_model.delay", _delay)

    call_command("train_models")
    out = capsys.readouterr().out

    _check((facility.id, "DEMAND_FORECAST") in seen)
    _check((None, "ANOMALY_DETECTOR") in seen)
    _check("Enqueued" in out)


@pytest.mark.django_db
def test_train_models_wait_raises_on_failed_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wait mode should raise when any enqueued task reports failure."""
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
def test_train_models_anomaly_ignores_facility_selectors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Anomaly-only mode should ignore facility filters and run globally."""
    FacilityFactory(is_active=True)
    monkeypatch.setenv("FUELSENSE_ENABLE_TRAINING_TASKS", "1")

    seen: list[tuple[int | None, str]] = []

    def _delay(facility_id: int | None, model_type: str) -> _AsyncResult:
        seen.append((facility_id, model_type))
        return _AsyncResult(task_id="task-a", payload={"status": "promoted"})

    monkeypatch.setattr("fuelsense.core.management.commands.train_models.retrain_model.delay", _delay)

    call_command("train_models", "--models", "anomaly", "--facility-id", "123")
    out = capsys.readouterr().out

    _check(seen == [(None, "ANOMALY_DETECTOR")])
    _check("ignored for anomaly training" in out)


@pytest.mark.django_db
def test_train_models_respects_disabled_gate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Disabled training gate should skip all task dispatching."""
    FacilityFactory(is_active=True)
    monkeypatch.setenv("FUELSENSE_ENABLE_TRAINING_TASKS", "0")

    def _no_dispatch(*args: object, **kwargs: object) -> None:
        _ = args, kwargs
        msg = "delay should not be called when training is disabled"
        raise AssertionError(msg)

    monkeypatch.setattr("fuelsense.core.management.commands.train_models.retrain_model.delay", _no_dispatch)

    call_command("train_models")
    out = capsys.readouterr().out
    _check("Training tasks are disabled" in out)
