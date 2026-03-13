"""Tests for health URL behavior under cache-roundtrip failure conditions."""

from __future__ import annotations

import pytest
from django.test import Client

HTTP_SERVICE_UNAVAILABLE = 503


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@pytest.mark.django_db
def test_healthz_returns_503_when_cache_roundtrip_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Health endpoint should report 503 with detail when cache check fails."""

    def _cache_get(_key: str) -> str:
        return "not-ok"

    monkeypatch.setattr("fuelsense.urls.cache.get", _cache_get)

    client = Client()
    response = client.get("/healthz")

    _check(
        response.status_code == HTTP_SERVICE_UNAVAILABLE,
        "Health endpoint should return 503 on cache failure",
    )
    _check(response.json()["detail"] == "Cache check failed", "Health endpoint should explain cache failure")
