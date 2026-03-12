from __future__ import annotations

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0002_planningcycle_lifecycle_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="delivery",
            name="idempotency_key",
            field=models.CharField(blank=True, db_index=True, max_length=64, null=True, unique=True),
        ),
    ]
