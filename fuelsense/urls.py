"""FuelSense URL configuration."""

from django.contrib import admin
from django.core.cache import cache
from django.db import DatabaseError, connection
from django.http import HttpRequest, JsonResponse
from django.urls import include, path


def healthz(_request: HttpRequest) -> JsonResponse:
    """Report app health by checking database and cache round-trips."""
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
    except DatabaseError as exc:  # pragma: no cover
        return JsonResponse({"status": "error", "detail": str(exc)}, status=503)

    try:
        cache.set("healthz", "ok", timeout=5)
        cache_ok = cache.get("healthz") == "ok"
    except Exception:  # noqa: BLE001
        cache_ok = False

    if not cache_ok:
        return JsonResponse({"status": "error", "detail": "Cache check failed"}, status=503)
    return JsonResponse({"status": "ok"})


def readyz(_request: HttpRequest) -> JsonResponse:
    """Report readiness by ensuring database connectivity is available."""
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
    except DatabaseError as exc:  # pragma: no cover
        return JsonResponse({"status": "not-ready", "detail": str(exc)}, status=503)
    return JsonResponse({"status": "ready"})


urlpatterns: list[object] = [
    path("admin/", admin.site.urls),
    path("api/v1/", include("fuelsense.core.urls")),
    path("", include("django_prometheus.urls")),
    path("healthz", healthz),
    path("readyz", readyz),
]
