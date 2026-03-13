"""Shared pytest fixtures for synthetic data tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_django.fixtures import SettingsWrapper


@pytest.fixture(autouse=True)
def use_locmem_cache(settings: SettingsWrapper) -> None:
    """Use an isolated in-memory cache backend for synthetic tests."""
    settings.CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "fuelsense-synthetic-tests",
        },
    }
