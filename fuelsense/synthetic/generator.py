"""Synthetic data orchestration entry points."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any

import numpy as np
from django.db import transaction
from django.utils import timezone

from fuelsense.core.models import (
    Depot,
    DepotFacilityAssignment,
    Facility,
    FuelType,
    InventoryLog,
    Vehicle,
)
from fuelsense.synthetic.anomalies import AnomalyInjector
from fuelsense.synthetic.consumption import generate_consumption
from fuelsense.synthetic.weather import generate_weather_series


@dataclass
class GenerationSummary:
    fuel_types: int
    depots: int
    facilities: int
    vehicles: int
    inventory_logs: int
    anomaly_events: int


class SyntheticDataGenerator:
    def __init__(self, facilities: int = 50, days: int = 365, seed: int = 42) -> None:
        self.facility_count = facilities
        self.days = days
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.injector = AnomalyInjector(trigger_probability=0.05)

    def _ensure_fuel_types(self) -> list[FuelType]:
        specs = [
            ("LNG", "m3", 430.0, "2.1"),
            ("Diesel", "liters", 832.0, "3"),
            ("HFO", "tonnes", 980.0, "9"),
            ("Natural Gas", "MWh", 0.8, "2.1"),
        ]
        fuels: list[FuelType] = []
        for name, unit, density, hazmat in specs:
            fuel, _ = FuelType.objects.get_or_create(
                name=name,
                defaults={
                    "unit": unit,
                    "density_kg_per_unit": density,
                    "hazmat_class": hazmat,
                },
            )
            fuels.append(fuel)
        return fuels

    def _create_depots(self, fuels: list[FuelType]) -> list[Depot]:
        depot_coords = [
            (24.7136, 46.6753),
            (21.4858, 39.1925),
            (26.4207, 50.0888),
            (18.2465, 42.5117),
            (28.3834, 36.5662),
        ]
        depots: list[Depot] = []
        for idx, (lat, lng) in enumerate(depot_coords, start=1):
            depots.append(
                Depot.objects.create(
                    name=f"Depot {idx}",
                    latitude=lat,
                    longitude=lng,
                    fuel_type=fuels[(idx - 1) % len(fuels)],
                    fuel_inventory=float(self.rng.uniform(20_000, 60_000)),
                )
            )
        return depots

    def _create_network(self, depots: list[Depot], fuels: list[FuelType]) -> tuple[list[Facility], int]:
        facilities: list[Facility] = []
        vehicle_count = 0

        for depot in depots:
            for vehicle_idx in range(10):
                Vehicle.objects.create(
                    depot=depot,
                    registration=f"{depot.name.replace(' ', '')}-{vehicle_idx:02d}",
                    capacity=float(self.rng.uniform(5000, 12000)),
                    cost_per_km=float(self.rng.uniform(1.5, 4.0)),
                    is_available=True,
                )
                vehicle_count += 1

        for idx in range(self.facility_count):
            depot = depots[idx % len(depots)]
            fuel = fuels[idx % len(fuels)]
            cap = float(self.rng.uniform(2500, 9000))
            min_safe = cap * float(self.rng.uniform(0.1, 0.25))
            reorder = cap * float(self.rng.uniform(0.2, 0.4))
            inv = cap * float(self.rng.uniform(0.5, 0.95))
            facility = Facility.objects.create(
                name=f"Facility {idx + 1}",
                facility_type=str(self.rng.choice(list(Facility.FacilityType.values))),
                fuel_type=fuel,
                latitude=float(depot.latitude + self.rng.normal(0.0, 0.8)),
                longitude=float(depot.longitude + self.rng.normal(0.0, 0.8)),
                storage_capacity=cap,
                current_inventory=inv,
                min_safe_inventory=min_safe,
                dynamic_reorder_point=reorder,
                delivery_window_start=time(hour=6),
                delivery_window_end=time(hour=14),
                delivery_days=[0, 1, 2, 3, 4],
                is_active=True,
            )
            DepotFacilityAssignment.objects.create(depot=depot, facility=facility)
            facilities.append(facility)

        return facilities, vehicle_count

    def _generate_inventory_logs(self, facilities: list[Facility]) -> tuple[int, int]:
        start = timezone.make_aware(datetime.combine((timezone.now() - timedelta(days=self.days)).date(), time.min))
        log_count = 0
        anomaly_count = 0

        for facility in facilities:
            base_load = float(self.rng.uniform(100, 500))
            weather = generate_weather_series(self.days, facility.latitude, self.rng)
            inventory = facility.current_inventory

            for day_idx in range(self.days):
                day_ts = start + timedelta(days=day_idx)
                temp, wind, solar = weather[day_idx]
                consumption = generate_consumption(base_load, day_idx, float(temp), day_ts.weekday(), self.rng)
                adjusted, anomaly = self.injector.apply(str(facility.id), day_idx, consumption, base_load, self.rng)
                if anomaly is not None:
                    anomaly_count += 1

                adjusted_value = 0.0 if np.isnan(adjusted) else float(max(adjusted, 0.0))
                inventory -= adjusted_value

                if inventory <= facility.min_safe_inventory:
                    refill_amount = facility.storage_capacity * float(self.rng.uniform(0.45, 0.75))
                    inventory = min(facility.storage_capacity, inventory + refill_amount)

                InventoryLog.objects.create(
                    facility=facility,
                    timestamp=day_ts,
                    inventory_level=inventory,
                    consumption=adjusted_value,
                    temperature=float(temp),
                    wind_speed=float(wind),
                    solar_irradiance=float(solar),
                )
                log_count += 1

            facility.current_inventory = inventory
            facility.save(update_fields=["current_inventory"])

        return log_count, anomaly_count

    @transaction.atomic
    def generate(self) -> GenerationSummary:
        InventoryLog.objects.all().delete()
        DepotFacilityAssignment.objects.all().delete()
        Vehicle.objects.all().delete()
        Facility.objects.all().delete()
        Depot.objects.all().delete()

        fuels = self._ensure_fuel_types()
        depots = self._create_depots(fuels)
        facilities, vehicle_count = self._create_network(depots, fuels)
        log_count, anomaly_count = self._generate_inventory_logs(facilities)

        return GenerationSummary(
            fuel_types=len(fuels),
            depots=len(depots),
            facilities=len(facilities),
            vehicles=vehicle_count,
            inventory_logs=log_count,
            anomaly_events=anomaly_count,
        )


def generate_synthetic_data(facilities: int, days: int, seed: int) -> dict[str, Any]:
    summary = SyntheticDataGenerator(facilities=facilities, days=days, seed=seed).generate()
    return {
        "fuel_types": summary.fuel_types,
        "depots": summary.depots,
        "facilities": summary.facilities,
        "vehicles": summary.vehicles,
        "inventory_logs": summary.inventory_logs,
        "anomaly_events": summary.anomaly_events,
    }
