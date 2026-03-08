"""FuelSense URL configuration."""

from django.contrib import admin
from django.http import JsonResponse
from django.urls import include, path


def healthz(_request):
    return JsonResponse({"status": "ok"})


def readyz(_request):
    return JsonResponse({"status": "ready"})


urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/", include("fuelsense.core.urls")),
    path("healthz", healthz),
    path("readyz", readyz),
]
