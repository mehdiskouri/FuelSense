"""FuelSense URL configuration."""

from django.core.cache import cache
from django.contrib import admin
from django.db import connection
from django.http import JsonResponse
from django.urls import include, path


def healthz(_request):
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        cache.set("healthz", "ok", timeout=5)
        cache_ok = cache.get("healthz") == "ok"
    except Exception as exc:  # pragma: no cover
        return JsonResponse({"status": "error", "detail": str(exc)}, status=503)
    if not cache_ok:
        return JsonResponse({"status": "error", "detail": "Cache check failed"}, status=503)
    return JsonResponse({"status": "ok"})


def readyz(_request):
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
    except Exception as exc:  # pragma: no cover
        return JsonResponse({"status": "not-ready", "detail": str(exc)}, status=503)
    return JsonResponse({"status": "ready"})


urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/", include("fuelsense.core.urls")),
    path("", include("django_prometheus.urls")),
    path("healthz", healthz),
    path("readyz", readyz),
]
