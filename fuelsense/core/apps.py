"""App configuration for fuelsense.core."""

from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "fuelsense.core"

    def ready(self) -> None:
        # Import signal handlers on app startup.
        from fuelsense.core import signals  # noqa: F401
