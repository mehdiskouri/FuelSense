"""Shared test fixtures for fuelsense_common package tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fuelsense_common.registry import clear_registry

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def reset_backend_registry() -> Iterator[None]:
    """Clear backend registry before and after every test."""
    clear_registry()
    yield
    clear_registry()
