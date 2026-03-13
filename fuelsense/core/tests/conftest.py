"""Shared pytest fixtures for core API/admin tests."""

from __future__ import annotations

import datetime as dt
import importlib
import secrets
from typing import TYPE_CHECKING, Protocol, cast

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone

from fuelsense.core.models import InventoryLog
from fuelsense.core.tests.factories import FacilityFactory

if TYPE_CHECKING:
    from django.conf import LazySettings
    from django.contrib.auth.base_user import AbstractBaseUser
    from django.test.client import Client


DEFAULT_SAMPLE_FACILITY_COUNT = 5
DEFAULT_SAMPLE_LOG_COUNT = 30


class _AuthClient(Protocol):
    def credentials(self, **kwargs: object) -> None: ...


class _ApiClientConstructor(Protocol):
    def __call__(self) -> _AuthClient: ...


class _Token(Protocol):
    key: str


class _TokenManager(Protocol):
    def get_or_create(self, user: object) -> tuple[_Token, bool]: ...


class _TokenModel(Protocol):
    objects: _TokenManager


class _RestFrameworkTestModule(Protocol):
    APIClient: _ApiClientConstructor


class _RestFrameworkTokenModule(Protocol):
    Token: _TokenModel


class _UserManager(Protocol):
    def create_user(self, username: str, password: str) -> object: ...

    def create_superuser(self, username: str, email: str, password: str) -> object: ...


def _new_api_client() -> _AuthClient:
    module = cast("_RestFrameworkTestModule", importlib.import_module("rest_framework.test"))
    return module.APIClient()


def _token_model() -> _TokenModel:
    module = cast("_RestFrameworkTokenModule", importlib.import_module("rest_framework.authtoken.models"))
    return module.Token


def _user_manager() -> _UserManager:
    user_model = get_user_model()
    return cast("_UserManager", user_model.objects)


@pytest.fixture
def api_client(_db: object) -> object:
    """Create authenticated DRF client fixture using token auth credentials."""
    generated_password = secrets.token_urlsafe(12)
    user = _user_manager().create_user(username="apiuser", password=generated_password)
    token, _ = _token_model().objects.get_or_create(user=user)
    client = _new_api_client()
    client.credentials(HTTP_AUTHORIZATION=f"Token {token.key}")
    return client


@pytest.fixture(autouse=True)
def use_locmem_cache(settings: object) -> None:
    """Force local-memory cache backend for deterministic test isolation."""
    django_settings = cast("LazySettings", settings)
    django_settings.CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "fuelsense-tests",
        },
    }


@pytest.fixture
def admin_client(_db: object, client: Client) -> Client:
    """Create authenticated Django admin client fixture with superuser login."""
    user = _user_manager().create_superuser("admin", "admin@example.com", "adminpass")
    client.force_login(cast("AbstractBaseUser", user))
    return client


@pytest.fixture
def sample_facilities(_db: object) -> list[object]:
    """Create a default set of facilities used across dashboard/cache tests."""
    return [FacilityFactory() for _ in range(DEFAULT_SAMPLE_FACILITY_COUNT)]


@pytest.fixture
def sample_inventory_logs(_db: object, sample_facilities: list[object]) -> list[InventoryLog]:
    """Create default rolling inventory logs for the first sample facility."""
    facility = sample_facilities[0]
    base = timezone.now()
    return [
        InventoryLog.objects.create(
            facility=facility,
            timestamp=base - dt.timedelta(days=i),
            inventory_level=500 - i,
            consumption=20 + i * 0.1,
            temperature=25.0,
            wind_speed=3.0,
            solar_irradiance=500.0,
        )
        for i in range(DEFAULT_SAMPLE_LOG_COUNT)
    ]
