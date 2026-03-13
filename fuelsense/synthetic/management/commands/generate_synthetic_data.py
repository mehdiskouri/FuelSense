"""Management command for synthetic data generation."""

from __future__ import annotations

from typing import cast

from django.core.management.base import BaseCommand, CommandParser

from fuelsense.synthetic.generator import generate_synthetic_data


class Command(BaseCommand):
    """Create synthetic FuelSense records for local/dev datasets."""

    help = "Generate synthetic FuelSense dataset for development and testing."

    def add_arguments(self, parser: CommandParser) -> None:
        """Register command-line arguments for generation size and seed."""
        parser.add_argument("--facilities", type=int, default=50)
        parser.add_argument("--days", type=int, default=365)
        parser.add_argument("--seed", type=int, default=42)

    def handle(self, *_args: object, **options: object) -> None:
        """Generate synthetic data and print a short run summary."""
        options_map = cast("dict[str, object]", options)
        facilities = int(options_map["facilities"])
        days = int(options_map["days"])
        seed = int(options_map["seed"])

        self.stdout.write(f"Generating synthetic data: facilities={facilities}, days={days}, seed={seed}")
        summary = generate_synthetic_data(facilities=facilities, days=days, seed=seed)
        self.stdout.write(self.style.SUCCESS("Synthetic data generation complete."))
        self.stdout.write(str(summary))
