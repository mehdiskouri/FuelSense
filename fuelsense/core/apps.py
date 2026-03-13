"""App configuration for fuelsense.core."""

from importlib import import_module

from django.apps import AppConfig


class CoreConfig(AppConfig):
    """Django app config for core domain models and startup hooks."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "fuelsense.core"

    def ready(self) -> None:
        """Import signal handlers when the app registry becomes ready."""
        # Import signal handlers on app startup.
        import_module("fuelsense.core.signals")
