"""Management command placeholder for synthetic data generation."""

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Generate synthetic FuelSense data (Phase 0 placeholder)."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--facilities", type=int, default=50)
        parser.add_argument("--days", type=int, default=365)
        parser.add_argument("--seed", type=int, default=42)

    def handle(self, *args, **options) -> None:
        self.stdout.write(
            self.style.WARNING(
                "Phase 0 placeholder: synthetic data generation is implemented in Phase 1."
            )
        )
