"""Management command for synthetic data generation."""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandParser

from fuelsense.synthetic.generator import generate_synthetic_data


class Command(BaseCommand):
    """Create synthetic FuelSense records for local/dev datasets."""

    help = "Generate synthetic FuelSense dataset for development and testing."

    @staticmethod
    def _coerce_int(value: object, default: int) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float | str):
            return int(value)
        return default

    def add_arguments(self, parser: CommandParser) -> None:
        """Register command-line arguments for generation size and seed."""
        parser.add_argument("--facilities", type=int, default=50)
        parser.add_argument("--days", type=int, default=365)
        parser.add_argument("--seed", type=int, default=42)

    def handle(self, *_args: object, **options: object) -> None:
        """Generate synthetic data and print a short run summary."""
        facilities_raw = options.get("facilities", 50)
        days_raw = options.get("days", 365)
        seed_raw = options.get("seed", 42)

        facilities = self._coerce_int(facilities_raw, 50)
        days = self._coerce_int(days_raw, 365)
        seed = self._coerce_int(seed_raw, 42)

        self.stdout.write(f"Generating synthetic data: facilities={facilities}, days={days}, seed={seed}")
        summary = generate_synthetic_data(facilities=facilities, days=days, seed=seed)
        self.stdout.write(self.style.SUCCESS("Synthetic data generation complete."))
        self.stdout.write(str(summary))
