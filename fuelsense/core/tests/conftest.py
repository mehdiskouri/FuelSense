from __future__ import annotations

# pyright: reportMissingTypeStubs=false

import datetime as dt
from typing import Any

import pytest
from django.contrib.auth import get_user_model
from django.test.client import Client
from django.utils import timezone
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient

from fuelsense.core.models import InventoryLog
from fuelsense.core.tests.factories import FacilityFactory


@pytest.fixture
def api_client(db: Any) -> APIClient:
    user = get_user_model().objects.create_user(username="apiuser", password="password123")
    token, _ = Token.objects.get_or_create(user=user)
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION=f"Token {token.key}")
    return client


@pytest.fixture(autouse=True)
def use_locmem_cache(settings: Any) -> None:
    settings.CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "fuelsense-tests",
        }
    }


@pytest.fixture
def admin_client(db: Any, client: Client) -> Client:
    user = get_user_model().objects.create_superuser("admin", "admin@example.com", "adminpass")
    client.force_login(user)
    return client


@pytest.fixture
def sample_facilities(db: Any) -> list[Any]:
    return [FacilityFactory() for _ in range(5)]


@pytest.fixture
def sample_inventory_logs(db: Any, sample_facilities: list[Any]) -> list[InventoryLog]:
    facility = sample_facilities[0]
    base = timezone.now()
    logs: list[InventoryLog] = []
    for i in range(30):
        logs.append(
            InventoryLog.objects.create(
                facility=facility,
                timestamp=base - dt.timedelta(days=i),
                inventory_level=500 - i,
                consumption=20 + i * 0.1,
                temperature=25.0,
                wind_speed=3.0,
                solar_irradiance=500.0,
            )
        )
    return logs
