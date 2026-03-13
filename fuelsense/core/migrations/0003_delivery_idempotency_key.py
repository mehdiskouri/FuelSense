"""Add idempotency key field for delivery deduplication."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Sequence

from django.db import migrations, models


class Migration(migrations.Migration):
    """Attach idempotency key to delivery records."""

    dependencies: ClassVar[Sequence[Any]] = [
        ("core", "0002_planningcycle_lifecycle_fields"),
    ]

    operations: ClassVar[Sequence[Any]] = [
        migrations.AddField(
            model_name="delivery",
            name="idempotency_key",
            field=models.CharField(blank=True, db_index=True, max_length=64, null=True, unique=True),
        ),
    ]
