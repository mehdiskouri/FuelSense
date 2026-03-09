"""Feature engineering helpers for forecasting and drift workflows."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast

import numpy as np
from django.utils import timezone

from fuelsense.core.models import DeliveryItem, Facility, Forecast, InventoryLog, ModelRegistry


@dataclass(frozen=True)
class _SeriesRow:
    timestamp: Any
    consumption: float
    temperature: float
    wind_speed: float
    solar_irradiance: float


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _forward_fill(rows: list[_SeriesRow]) -> list[_SeriesRow]:
    last_temp = 0.0
    last_wind = 0.0
    last_solar = 0.0
    out: list[_SeriesRow] = []
    for row in rows:
        temp = row.temperature if not math.isnan(row.temperature) else last_temp
        wind = row.wind_speed if not math.isnan(row.wind_speed) else last_wind
        solar = row.solar_irradiance if not math.isnan(row.solar_irradiance) else last_solar
        last_temp, last_wind, last_solar = temp, wind, solar
        out.append(
            _SeriesRow(
                timestamp=row.timestamp,
                consumption=row.consumption,
                temperature=temp,
                wind_speed=wind,
                solar_irradiance=solar,
            )
        )
    return out


def _row_to_features(row: _SeriesRow) -> list[float]:
    dow = row.timestamp.weekday()
    angle = 2.0 * math.pi * (dow / 7.0)
    return [
        row.consumption,
        row.temperature,
        row.wind_speed,
        row.solar_irradiance,
        math.sin(angle),
        math.cos(angle),
    ]


def _inventory_series_for_facility(facility_id: int) -> list[_SeriesRow]:
    logs = list(
        InventoryLog.objects.filter(facility_id=facility_id)
        .order_by("timestamp")
        .values("timestamp", "consumption", "temperature", "wind_speed", "solar_irradiance")
    )
    rows = [
        _SeriesRow(
            timestamp=item["timestamp"],
            consumption=_safe_float(item["consumption"]),
            temperature=_safe_float(item["temperature"], default=float("nan")),
            wind_speed=_safe_float(item["wind_speed"], default=float("nan")),
            solar_irradiance=_safe_float(item["solar_irradiance"], default=float("nan")),
        )
        for item in logs
    ]
    return _forward_fill(rows)


def _window_or_pad(features: list[list[float]], lookback: int = 90) -> np.ndarray:
    if len(features) >= lookback:
        return np.asarray(features[-lookback:], dtype=np.float32)
    if not features:
        return np.zeros((lookback, 6), dtype=np.float32)

    pad_len = lookback - len(features)
    pad_row = features[0]
    padded = [pad_row[:] for _ in range(pad_len)] + features
    return np.asarray(padded, dtype=np.float32)


def build_lookback_matrix(facility_ids: Iterable[int]) -> np.ndarray:
    """Build model input tensor with shape [n_facilities, 90, 6]."""
    matrices: list[np.ndarray] = []
    for facility_id in facility_ids:
        rows = _inventory_series_for_facility(int(facility_id))
        features = [_row_to_features(row) for row in rows]
        matrices.append(_window_or_pad(features, lookback=90))

    if not matrices:
        return np.zeros((0, 90, 6), dtype=np.float32)
    return np.stack(matrices, axis=0)


def extract_training_data(facility_id: int | None) -> dict[str, np.ndarray]:
    """Extract temporal train/val/test arrays for a facility or global dataset."""
    if facility_id is None:
        logs_qs = InventoryLog.objects.all().order_by("timestamp")
    else:
        logs_qs = InventoryLog.objects.filter(facility_id=facility_id).order_by("timestamp")

    rows = [
        _SeriesRow(
            timestamp=item["timestamp"],
            consumption=_safe_float(item["consumption"]),
            temperature=_safe_float(item["temperature"], default=float("nan")),
            wind_speed=_safe_float(item["wind_speed"], default=float("nan")),
            solar_irradiance=_safe_float(item["solar_irradiance"], default=float("nan")),
        )
        for item in logs_qs.values("timestamp", "consumption", "temperature", "wind_speed", "solar_irradiance")
    ]
    rows = _forward_fill(rows)

    features = np.asarray([_row_to_features(row) for row in rows], dtype=np.float32)
    targets = np.asarray([row.consumption for row in rows], dtype=np.float32)

    n = int(features.shape[0])
    test_size = min(14, n)
    val_size = min(14, max(n - test_size, 0))
    train_end = max(n - test_size - val_size, 0)
    val_end = train_end + val_size

    return {
        "train_data": features[:train_end],
        "train_targets": targets[:train_end],
        "val_data": features[train_end:val_end],
        "val_targets": targets[train_end:val_end],
        "test_data": features[val_end:],
        "test_targets": targets[val_end:],
    }


def _extract_prediction_series(predictions_json: Any) -> list[float]:
    if not isinstance(predictions_json, list):
        return []
    items = cast(list[object], predictions_json)
    out: list[float] = []
    for raw_item in items:
        item = cast(dict[str, Any], raw_item) if isinstance(raw_item, dict) else None
        if not isinstance(item, dict):
            continue
        if "p50" in item:
            out.append(_safe_float(item.get("p50")))
        elif "prediction" in item:
            out.append(_safe_float(item.get("prediction")))
    return out


def build_drift_data() -> list[dict[str, Any]]:
    """Assemble per-facility drift payloads for active forecast models."""
    payloads: list[dict[str, Any]] = []
    recent_window_start = timezone.now() - timedelta(days=7)

    active_models = ModelRegistry.objects.filter(
        model_type=ModelRegistry.ModelType.DEMAND_FORECAST,
        is_active=True,
    ).select_related("facility")

    for model in active_models:
        model_facility_id = cast(int | None, getattr(model, "facility_id", None))
        baseline_rmse = cast(float | None, getattr(model, "validation_rmse", None))

        if model_facility_id is None:
            continue

        recent_actuals = list(
            InventoryLog.objects.filter(facility_id=model_facility_id, timestamp__gte=recent_window_start)
            .order_by("timestamp")
            .values_list("consumption", flat=True)
        )

        latest_forecast = (
            Forecast.objects.filter(facility_id=model_facility_id)
            .order_by("-created_at")
            .only("predictions_json")
            .first()
        )
        recent_predictions = _extract_prediction_series(getattr(latest_forecast, "predictions_json", []))

        payloads.append(
            {
                "facility_id": int(model_facility_id),
                "model_registry_id": int(cast(int, model.pk)),
                "baseline_rmse": _safe_float(baseline_rmse),
                "recent_actuals": [float(x) for x in recent_actuals],
                "recent_predictions": recent_predictions,
            }
        )

    return payloads


def active_facility_ids() -> list[int]:
    return list(Facility.objects.filter(is_active=True).values_list("id", flat=True))


def build_anomaly_features(facility_id: int) -> dict[str, float] | None:
    now = timezone.now()
    logs_qs = InventoryLog.objects.filter(facility_id=facility_id).order_by("-timestamp")
    recent_logs = list(
        logs_qs[:7].values("timestamp", "consumption", "temperature", "inventory_level")
    )
    if len(recent_logs) < 2:
        return None

    latest_log = recent_logs[0]
    prev_log = recent_logs[1]
    actual = _safe_float(latest_log.get("consumption"))
    prev_consumption = _safe_float(prev_log.get("consumption"), default=0.0)

    latest_forecast = (
        Forecast.objects.filter(facility_id=facility_id)
        .order_by("-created_at")
        .only("predictions_json")
        .first()
    )
    if latest_forecast is None:
        return None

    p50_series = _extract_prediction_series(getattr(latest_forecast, "predictions_json", []))
    if not p50_series:
        return None
    predicted = float(p50_series[0])

    consumption_history = [_safe_float(row.get("consumption")) for row in recent_logs]
    rolling_std = float(np.std(np.asarray(consumption_history, dtype=np.float32)))
    z_score = (actual - predicted) / max(rolling_std, 1e-6)

    abs_z_history: list[float] = []
    for row in recent_logs[:3]:
        row_actual = _safe_float(row.get("consumption"))
        row_z = (row_actual - predicted) / max(rolling_std, 1e-6)
        abs_z_history.append(abs(row_z))
    z_score_rolling_3d = float(np.mean(np.asarray(abs_z_history, dtype=np.float32)))

    consumption_delta_pct = (actual - prev_consumption) / max(abs(prev_consumption), 1e-6)

    temps = [_safe_float(row.get("temperature"), default=0.0) for row in recent_logs]
    temp_mean = float(np.mean(np.asarray(temps, dtype=np.float32)))
    temperature_residual = _safe_float(latest_log.get("temperature"), default=0.0) - temp_mean

    latest_ts = latest_log.get("timestamp")
    day_of_week = float(latest_ts.weekday()) if latest_ts is not None else 0.0

    latest_delivery_item = (
        DeliveryItem.objects.filter(facility_id=facility_id)
        .order_by("-actual_arrival", "-planned_arrival")
        .values("actual_arrival", "planned_arrival")
        .first()
    )
    if latest_delivery_item is None:
        hours_since_delivery = 9999.0
    else:
        arrival = latest_delivery_item.get("actual_arrival") or latest_delivery_item.get("planned_arrival")
        if arrival is None:
            hours_since_delivery = 9999.0
        else:
            hours_since_delivery = max((now - arrival).total_seconds() / 3600.0, 0.0)

    facility_row = (
        Facility.objects.filter(id=facility_id)
        .values("storage_capacity", "current_inventory")
        .first()
    )
    if facility_row is None:
        return None
    inventory_level_pct = _safe_float(facility_row.get("current_inventory")) / max(
        _safe_float(facility_row.get("storage_capacity")),
        1e-6,
    )

    return {
        "z_score": float(z_score),
        "z_score_rolling_3d": float(z_score_rolling_3d),
        "consumption_delta_pct": float(consumption_delta_pct),
        "temperature_residual": float(temperature_residual),
        "day_of_week": float(day_of_week),
        "hours_since_delivery": float(hours_since_delivery),
        "inventory_level_pct": float(inventory_level_pct),
    }
