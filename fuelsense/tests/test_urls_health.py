from __future__ import annotations

import pytest
from django.test import Client


@pytest.mark.django_db
def test_healthz_returns_503_when_cache_roundtrip_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fuelsense.urls.cache.get", lambda _key: "not-ok")

    client = Client()
    response = client.get("/healthz")

    assert response.status_code == 503
    assert response.json()["detail"] == "Cache check failed"
