from __future__ import annotations

import pytest

from fuelsense.synthetic.generator import generate_synthetic_data


@pytest.mark.django_db
def test_generator_is_deterministic_summary():
    a = generate_synthetic_data(facilities=3, days=10, seed=42)
    b = generate_synthetic_data(facilities=3, days=10, seed=42)
    assert a == b
    assert a["facilities"] == 3
    assert a["inventory_logs"] == 30
