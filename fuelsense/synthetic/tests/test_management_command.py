"""Tests for synthetic data management command record creation behavior."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from django.core.management import call_command

from fuelsense.core.models import Facility, InventoryLog
from fuelsense.synthetic.management.commands.generate_synthetic_data import Command

EXPECTED_FACILITIES = 2
EXPECTED_INVENTORY_LOGS = 10


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@pytest.mark.django_db
def test_management_command_creates_expected_records() -> None:
    """Synthetic command should create expected facility and inventory records."""
    call_command("generate_synthetic_data", facilities=2, days=5, seed=42)
    _check(Facility.objects.count() == EXPECTED_FACILITIES, "Expected two facilities to be generated")
    _check(InventoryLog.objects.count() == EXPECTED_INVENTORY_LOGS, "Expected facilities * days inventory logs")


def test_command_coerce_int_handles_bool_string_and_fallback() -> None:
    """Command handle should coerce bool/string options and fallback unsupported types."""
    expected_days = 12
    expected_seed = 42
    parsed: dict[str, int] = {}

    def _fake_generate_synthetic_data(*, facilities: int, days: int, seed: int) -> dict[str, int]:
        parsed["facilities"] = facilities
        parsed["days"] = days
        parsed["seed"] = seed
        return parsed

    command = Command()
    with patch(
        "fuelsense.synthetic.management.commands.generate_synthetic_data.generate_synthetic_data",
        side_effect=_fake_generate_synthetic_data,
    ):
        command.handle(facilities=True, days=str(expected_days), seed=object())

    _check(parsed["facilities"] == 1, "bool should coerce to int")
    _check(parsed["days"] == expected_days, "numeric strings should coerce to int")
    _check(parsed["seed"] == expected_seed, "unsupported values should fallback to defaults")
