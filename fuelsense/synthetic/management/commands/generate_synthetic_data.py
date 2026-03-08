"""Management command for synthetic data generation."""

from django.core.management.base import BaseCommand

from fuelsense.synthetic.generator import generate_synthetic_data


class Command(BaseCommand):
    help = "Generate synthetic FuelSense dataset for development and testing."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--facilities", type=int, default=50)
        parser.add_argument("--days", type=int, default=365)
        parser.add_argument("--seed", type=int, default=42)

    def handle(self, *args, **options) -> None:
        facilities = int(options["facilities"])
        days = int(options["days"])
        seed = int(options["seed"])

        self.stdout.write(f"Generating synthetic data: facilities={facilities}, days={days}, seed={seed}")
        summary = generate_synthetic_data(facilities=facilities, days=days, seed=seed)
        self.stdout.write(self.style.SUCCESS("Synthetic data generation complete."))
        self.stdout.write(str(summary))
