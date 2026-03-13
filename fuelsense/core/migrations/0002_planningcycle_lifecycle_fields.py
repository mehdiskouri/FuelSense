"""Add lifecycle status fields and indexes to planning cycles."""

from typing import ClassVar

from django.db import migrations, models


class Migration(migrations.Migration):
    """Extend planning cycle lifecycle metadata and query indexes."""

    dependencies: ClassVar[list[object]] = [
        ("core", "0001_initial"),
    ]

    operations: ClassVar[list[object]] = [
        migrations.AddField(
            model_name="planningcycle",
            name="completed_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="planningcycle",
            name="started_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="planningcycle",
            name="status",
            field=models.CharField(
                choices=[
                    ("QUEUED", "Queued"),
                    ("RUNNING", "Running"),
                    ("COMPLETED", "Completed"),
                    ("FAILED", "Failed"),
                ],
                default="COMPLETED",
                max_length=20,
            ),
        ),
        migrations.AddIndex(
            model_name="planningcycle",
            index=models.Index(fields=["status", "trigger_type"], name="core_planni_status_bda9e4_idx"),
        ),
        migrations.AddIndex(
            model_name="planningcycle",
            index=models.Index(fields=["-triggered_at", "status"], name="core_planni_trigger_9eb09b_idx"),
        ),
    ]
