"""Production settings for FuelSense."""

from __future__ import annotations

import os

from . import base as base_settings
from .base import *  # noqa: F403

DEBUG = False
SECRET_KEY = os.getenv("SECRET_KEY", getattr(base_settings, "SECRET_KEY", ""))
ALLOWED_HOSTS = [host.strip() for host in os.getenv("ALLOWED_HOSTS", "").split(",") if host.strip()]
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SECURE_SSL_REDIRECT = os.getenv("SECURE_SSL_REDIRECT", "true").lower() == "true"
SECURE_HSTS_SECONDS = 31536000
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
STATIC_ROOT = "/app/staticfiles"
