"""Django's command-line utility for administrative tasks."""

import importlib
import os
import sys


def main() -> None:
    """Run Django administrative commands with default local settings."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fuelsense.settings.development")
    try:
        execute_from_command_line = importlib.import_module("django.core.management").execute_from_command_line
    except ImportError as exc:
        msg = "Couldn't import Django. Is it installed and on PYTHONPATH?"
        raise ImportError(msg) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
