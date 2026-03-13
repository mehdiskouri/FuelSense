"""Tests for train_models management command registration and dispatch behavior."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from django.core.management import call_command, get_commands
from django.core.management.base import CommandError

from fuelsense.core.tests.factories import create_facility


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
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dry-run should print planned work but never enqueue Celery tasks."""
    create_facility(is_active=True)

    def _no_dispatch(*args: object, **kwargs: object) -> None:
        _ = args, kwargs
        msg = "delay should not be called during dry-run"
        raise AssertionError(msg)

    with (
        patch.dict(os.environ, {"FUELSENSE_ENABLE_TRAINING_TASKS": "1"}, clear=False),
        patch("fuelsense.core.management.commands.train_models.retrain_model.delay", side_effect=_no_dispatch),
    ):
        call_command("train_models", "--dry-run")
    out = capsys.readouterr().out
    _check("DRY-RUN" in out)
    _check("demand:facility" in out)
    _check("anomaly:global" in out)


@pytest.mark.django_db
def test_train_models_enqueues_demand_and_anomaly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Default run should enqueue both demand and anomaly training jobs."""
    facility = create_facility(is_active=True)
    facility_id = int(facility.id)

    seen: list[tuple[int | None, str]] = []

    def _delay(facility_id: int | None, model_type: str) -> _AsyncResult:
        seen.append((facility_id, model_type))
        return _AsyncResult(task_id=f"task-{len(seen)}", payload={"status": "promoted"})

    with (
        patch.dict(os.environ, {"FUELSENSE_ENABLE_TRAINING_TASKS": "1"}, clear=False),
        patch("fuelsense.core.management.commands.train_models.retrain_model.delay", side_effect=_delay),
    ):
        call_command("train_models")
    out = capsys.readouterr().out

    _check((facility_id, "DEMAND_FORECAST") in seen)
    _check((None, "ANOMALY_DETECTOR") in seen)
    _check("Enqueued" in out)


@pytest.mark.django_db
def test_train_models_wait_raises_on_failed_task() -> None:
    """Wait mode should raise when any enqueued task reports failure."""
    create_facility(is_active=True)
    queue: list[_AsyncResult] = [
        _AsyncResult("task-1", {"status": "promoted"}),
        _AsyncResult("task-2", {"status": "failed"}),
    ]

    def _delay(facility_id: int | None, model_type: str) -> _AsyncResult:
        _ = facility_id, model_type
        return queue.pop(0)

    with (
        patch.dict(os.environ, {"FUELSENSE_ENABLE_TRAINING_TASKS": "1"}, clear=False),
        patch("fuelsense.core.management.commands.train_models.retrain_model.delay", side_effect=_delay),
        pytest.raises(CommandError),
    ):
        call_command("train_models", "--models", "all", "--wait")


@pytest.mark.django_db
def test_train_models_anomaly_ignores_facility_selectors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Anomaly-only mode should ignore facility filters and run globally."""
    create_facility(is_active=True)
    seen: list[tuple[int | None, str]] = []

    def _delay(facility_id: int | None, model_type: str) -> _AsyncResult:
        seen.append((facility_id, model_type))
        return _AsyncResult(task_id="task-a", payload={"status": "promoted"})

    with (
        patch.dict(os.environ, {"FUELSENSE_ENABLE_TRAINING_TASKS": "1"}, clear=False),
        patch("fuelsense.core.management.commands.train_models.retrain_model.delay", side_effect=_delay),
    ):
        call_command("train_models", "--models", "anomaly", "--facility-id", "123")
    out = capsys.readouterr().out

    _check(seen == [(None, "ANOMALY_DETECTOR")])
    _check("ignored for anomaly training" in out)


@pytest.mark.django_db
def test_train_models_respects_disabled_gate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Disabled training gate should skip all task dispatching."""
    create_facility(is_active=True)

    def _no_dispatch(*args: object, **kwargs: object) -> None:
        _ = args, kwargs
        msg = "delay should not be called when training is disabled"
        raise AssertionError(msg)

    with (
        patch.dict(os.environ, {"FUELSENSE_ENABLE_TRAINING_TASKS": "0"}, clear=False),
        patch("fuelsense.core.management.commands.train_models.retrain_model.delay", side_effect=_no_dispatch),
    ):
        call_command("train_models")
    out = capsys.readouterr().out
    _check("Training tasks are disabled" in out)
