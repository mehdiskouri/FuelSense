from __future__ import annotations

import datetime as dt

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient

from fuelsense.core.models import InventoryLog
from fuelsense.core.tests.factories import FacilityFactory


@pytest.fixture
def api_client(db):
	user = get_user_model().objects.create_user(username="apiuser", password="password123")
	token, _ = Token.objects.get_or_create(user=user)
	client = APIClient()
	client.credentials(HTTP_AUTHORIZATION=f"Token {token.key}")
	return client


@pytest.fixture(autouse=True)
def use_locmem_cache(settings):
	settings.CACHES = {
		"default": {
			"BACKEND": "django.core.cache.backends.locmem.LocMemCache",
			"LOCATION": "fuelsense-tests",
		}
	}


@pytest.fixture
def admin_client(db, client):
	user = get_user_model().objects.create_superuser("admin", "admin@example.com", "adminpass")
	client.force_login(user)
	return client


@pytest.fixture
def sample_facilities(db):
	return [FacilityFactory() for _ in range(5)]


@pytest.fixture
def sample_inventory_logs(db, sample_facilities):
	facility = sample_facilities[0]
	base = timezone.now()
	logs = []
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
