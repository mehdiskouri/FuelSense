"""Core API URL router placeholders."""

from django.http import JsonResponse
from django.urls import path


def api_root(_request):
    return JsonResponse({"service": "fuelsense-core", "version": "v1"})


urlpatterns = [
    path("", api_root),
]
