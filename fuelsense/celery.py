"""Celery app setup for FuelSense."""

from __future__ import annotations

import os

from celery import Celery
from kombu import Queue

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fuelsense.settings.development")

app = Celery("fuelsense")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.conf.task_queues = (
    Queue("default"),
    Queue("training"),
    Queue("planning"),
)
app.autodiscover_tasks(["fuelsense.core"])

# Ensure core task module is imported so shared_task decorators register immediately.
import fuelsense.core.tasks  # noqa: E402,F401
