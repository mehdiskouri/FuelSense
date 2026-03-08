"""Model drift monitoring utilities for demand forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import fmean
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DriftConfig:
    drift_threshold: float = 1.5
    window_days: int = 7
    min_samples: int = 5


class DriftMonitor:
    DRIFT_THRESHOLD = 1.5
    WINDOW_DAYS = 7
    MIN_SAMPLES = 5

    def __init__(self, config: DriftConfig | None = None) -> None:
        cfg = config or DriftConfig(
            drift_threshold=self.DRIFT_THRESHOLD,
            window_days=self.WINDOW_DAYS,
            min_samples=self.MIN_SAMPLES,
        )
        self.config = cfg

    def check_drift(
        self,
        facility_id: int,
        baseline_rmse: float,
        recent_actuals: list[float],
        recent_predictions: list[float],
    ) -> dict[str, Any]:
        """Compute RMSE drift ratio and diagnostics for one facility."""
        sample_count = min(len(recent_actuals), len(recent_predictions))
        if sample_count < self.config.min_samples:
            return {
                "facility_id": facility_id,
                "needs_retrain": False,
                "reason": "insufficient_samples",
                "sample_count": sample_count,
                "drift_ratio": 1.0,
                "residual_mean": 0.0,
                "residual_std": 0.0,
                "residual_autocorrelation": 0.0,
                "max_absolute_error": 0.0,
            }

        actual = np.asarray(recent_actuals[:sample_count], dtype=np.float64)
        pred = np.asarray(recent_predictions[:sample_count], dtype=np.float64)
        residuals = actual - pred
        mse = float(np.mean(np.square(residuals)))
        recent_rmse = sqrt(max(mse, 0.0))

        baseline = baseline_rmse if baseline_rmse > 0 else 1e-6
        drift_ratio = recent_rmse / baseline

        residual_std = float(np.std(residuals))
        if sample_count > 1:
            left = residuals[:-1]
            right = residuals[1:]
            if float(np.std(left)) > 0 and float(np.std(right)) > 0:
                autocorr = float(np.corrcoef(left, right)[0, 1])
            else:
                autocorr = 0.0
        else:
            autocorr = 0.0

        return {
            "facility_id": facility_id,
            "needs_retrain": bool(drift_ratio > self.config.drift_threshold),
            "reason": "drift_exceeded" if drift_ratio > self.config.drift_threshold else "stable",
            "sample_count": sample_count,
            "drift_ratio": float(drift_ratio),
            "recent_rmse": float(recent_rmse),
            "baseline_rmse": float(baseline_rmse),
            "residual_mean": float(np.mean(residuals)),
            "residual_std": residual_std,
            "residual_autocorrelation": autocorr,
            "max_absolute_error": float(np.max(np.abs(residuals))),
        }

    def check_all_facilities(self, facility_drift_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate drift for all facilities and return retrain recommendations."""
        results: list[dict[str, Any]] = []
        retrain_ids: list[int] = []

        for item in facility_drift_data:
            result = self.check_drift(
                facility_id=int(item.get("facility_id", 0)),
                baseline_rmse=float(item.get("baseline_rmse", 0.0)),
                recent_actuals=[float(x) for x in item.get("recent_actuals", [])],
                recent_predictions=[float(x) for x in item.get("recent_predictions", [])],
            )
            results.append(result)
            if bool(result.get("needs_retrain")):
                retrain_ids.append(int(result["facility_id"]))

        ratios = [float(item.get("drift_ratio", 1.0)) for item in results] or [1.0]

        return {
            "facilities_checked": len(results),
            "retrain_facility_ids": retrain_ids,
            "drift_ratio_mean": float(fmean(ratios)),
            "drift_ratio_max": float(max(ratios)),
            "drift_ratio_p90": float(np.percentile(np.asarray(ratios, dtype=np.float64), 90)),
            "results": results,
        }
