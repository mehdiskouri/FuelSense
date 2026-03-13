"""Tests for synthetic data management command record creation behavior."""

from __future__ import annotations

import pytest
from django.core.management import call_command

from fuelsense.core.models import Facility, InventoryLog

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
