"""Production settings for FuelSense."""

from __future__ import annotations

import os

from .base import *  # noqa: F403

DEBUG = False
ALLOWED_HOSTS = [host.strip() for host in os.getenv("ALLOWED_HOSTS", "").split(",") if host.strip()]
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
