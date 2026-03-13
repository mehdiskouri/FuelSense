"""Tests for deterministic synthetic data generation summaries."""

from __future__ import annotations

import pytest

from fuelsense.synthetic.generator import generate_synthetic_data

EXPECTED_FACILITIES = 3
EXPECTED_LOG_COUNT = 30


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@pytest.mark.django_db
def test_generator_is_deterministic_summary() -> None:
    """Generator output should be deterministic for fixed seed and parameters."""
    a = generate_synthetic_data(facilities=EXPECTED_FACILITIES, days=10, seed=42)
    b = generate_synthetic_data(facilities=EXPECTED_FACILITIES, days=10, seed=42)
    _check(a == b, "Generator output should be deterministic with same seed")
    _check(a["facilities"] == EXPECTED_FACILITIES, "Expected facilities count in summary")
    _check(a["inventory_logs"] == EXPECTED_LOG_COUNT, "Expected inventory logs count in summary")
