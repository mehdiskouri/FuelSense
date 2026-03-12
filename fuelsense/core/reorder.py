"""Shared reorder-threshold semantics for fallback-safe planning behavior."""

from __future__ import annotations

from typing import Any

from django.db.models import Case, F, FloatField, QuerySet, When

EFFECTIVE_REORDER_ALIAS = "_effective_reorder_point"


def is_reliable_reorder_point(value: float | None) -> bool:
    """Return whether a dynamic reorder threshold can be safely used."""
    return value is not None and float(value) > 0.0


def get_effective_reorder_point(dynamic_reorder_point: float | None, min_safe_inventory: float) -> float:
    """Resolve the effective reorder threshold with fallback semantics."""
    if is_reliable_reorder_point(dynamic_reorder_point):
        return float(dynamic_reorder_point)
    return float(min_safe_inventory)


def effective_reorder_point_expression() -> Case:
    """Build a queryset expression for effective reorder point selection."""
    return Case(
        When(dynamic_reorder_point__gt=0.0, then=F("dynamic_reorder_point")),
        default=F("min_safe_inventory"),
        output_field=FloatField(),
    )


def with_effective_reorder_point(queryset: QuerySet[Any]) -> QuerySet[Any]:
    """Annotate a queryset with an effective reorder threshold alias."""
    return queryset.annotate(**{EFFECTIVE_REORDER_ALIAS: effective_reorder_point_expression()})


def filter_below_reorder(queryset: QuerySet[Any]) -> QuerySet[Any]:
    """Filter facilities whose inventory is below effective reorder threshold."""
    return with_effective_reorder_point(queryset).filter(current_inventory__lte=F(EFFECTIVE_REORDER_ALIAS))
