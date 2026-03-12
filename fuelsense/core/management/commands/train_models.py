from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from celery.result import AsyncResult
from django.core.management.base import BaseCommand, CommandError
from django.db import DatabaseError

from fuelsense.core.models import Facility, ModelRegistry
from fuelsense.core.tasks import retrain_model


@dataclass(frozen=True)
class TrainTarget:
    label: str
    facility_id: int | None
    model_type: str


class Command(BaseCommand):
    help = "Train and register demand/anomaly models via Celery tasks."

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--models",
            choices=["demand", "anomaly", "all"],
            default="all",
            help="Model families to train.",
        )
        parser.add_argument(
            "--facility-id",
            action="append",
            type=int,
            default=[],
            help="Single demand facility id. Can be provided multiple times.",
        )
        parser.add_argument(
            "--facility-ids",
            nargs="+",
            type=int,
            default=[],
            help="Demand facility ids list.",
        )
        parser.add_argument(
            "--all-active",
            action="store_true",
            help="Train demand models for all active facilities.",
        )
        parser.add_argument(
            "--include-global-demand",
            action="store_true",
            help="Also train a global demand model (facility_id=None).",
        )
        parser.add_argument("--dry-run", action="store_true", help="Print resolved training targets without dispatch.")
        parser.add_argument("--wait", action="store_true", help="Wait for all enqueued tasks and print outcomes.")
        parser.add_argument(
            "--wait-timeout",
            type=int,
            default=1800,
            help="Per-task wait timeout in seconds when --wait is enabled.",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        targets = self._resolve_targets(options)
        if not targets:
            self.stdout.write(self.style.WARNING("No training targets resolved."))
            return

        if options["dry_run"]:
            self.stdout.write(self.style.WARNING("DRY-RUN: no tasks dispatched."))
            for target in targets:
                self.stdout.write(f"target={target.label} model_type={target.model_type}")
            self.stdout.write(f"Resolved {len(targets)} training target(s).")
            return

        if os.environ.get("FUELSENSE_ENABLE_TRAINING_TASKS", "0") != "1":
            self.stdout.write(
                self.style.WARNING(
                    "Training tasks are disabled. Set FUELSENSE_ENABLE_TRAINING_TASKS=1 to enable dispatch.",
                ),
            )
            return

        async_jobs: list[tuple[TrainTarget, AsyncResult]] = []
        for target in targets:
            async_result = retrain_model.delay(target.facility_id, target.model_type)
            async_jobs.append((target, async_result))
            self.stdout.write(f"queued target={target.label} task_id={async_result.id}")

        self.stdout.write(f"Enqueued {len(async_jobs)} training task(s).")

        if not options["wait"]:
            return

        summary: dict[str, int] = {
            "promoted": 0,
            "rejected": 0,
            "failed": 0,
            "skipped": 0,
            "disabled": 0,
            "unknown": 0,
        }
        for target, async_result in async_jobs:
            try:
                payload = async_result.get(timeout=int(options["wait_timeout"]))
            except Exception as exc:
                summary["failed"] += 1
                self.stdout.write(self.style.ERROR(f"failed target={target.label} error={exc}"))
                continue

            status = self._status_from_payload(payload)
            summary[status] += 1
            self.stdout.write(f"completed target={target.label} status={status}")

        self.stdout.write(
            "Summary: "
            + ", ".join(
                f"{key}={value}" for key, value in summary.items() if value > 0 or key in {"promoted", "failed"}
            ),
        )

        if summary["failed"] > 0:
            raise CommandError("One or more training tasks failed.")

    def _resolve_targets(self, options: dict[str, Any]) -> list[TrainTarget]:
        selected = str(options["models"])
        include_demand = selected in {"demand", "all"}
        include_anomaly = selected in {"anomaly", "all"}

        provided_ids = {int(fid) for fid in list(options["facility_id"]) + list(options["facility_ids"])}
        explicit_selectors = bool(provided_ids) or bool(options["all_active"])

        targets: list[TrainTarget] = []
        if include_demand:
            demand_facility_ids: set[int] = set(provided_ids)
            if options["all_active"] or not provided_ids:
                try:
                    active_ids = set(Facility.objects.filter(is_active=True).values_list("id", flat=True))
                    demand_facility_ids.update(int(fid) for fid in active_ids)
                except DatabaseError as exc:
                    self.stdout.write(
                        self.style.WARNING(
                            f"Unable to resolve active demand facilities from database: {exc}. "
                            "Proceeding without auto-selected demand targets.",
                        ),
                    )

            for facility_id in sorted(demand_facility_ids):
                targets.append(
                    TrainTarget(
                        label=f"demand:facility:{facility_id}",
                        facility_id=facility_id,
                        model_type=ModelRegistry.ModelType.DEMAND_FORECAST,
                    ),
                )

            if options["include_global_demand"]:
                targets.append(
                    TrainTarget(
                        label="demand:global",
                        facility_id=None,
                        model_type=ModelRegistry.ModelType.DEMAND_FORECAST,
                    ),
                )

        if include_anomaly:
            if explicit_selectors:
                self.stdout.write(
                    self.style.WARNING("Facility selectors are ignored for anomaly training (global-only model)."),
                )
            targets.append(
                TrainTarget(
                    label="anomaly:global",
                    facility_id=None,
                    model_type=ModelRegistry.ModelType.ANOMALY_DETECTOR,
                ),
            )

        return targets

    @staticmethod
    def _status_from_payload(payload: object) -> str:
        if not isinstance(payload, dict):
            return "unknown"
        status = str(payload.get("status", "unknown"))
        if status in {"promoted", "rejected", "failed", "skipped", "disabled"}:
            return status
        return "unknown"
