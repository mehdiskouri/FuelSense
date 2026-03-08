from __future__ import annotations

import pytest
from django.core.management import call_command

from fuelsense.core.models import Facility, InventoryLog


@pytest.mark.django_db
def test_management_command_creates_expected_records():
    call_command("generate_synthetic_data", facilities=2, days=5, seed=42)
    assert Facility.objects.count() == 2
    assert InventoryLog.objects.count() == 10
