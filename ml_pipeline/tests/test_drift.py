"""Unit tests for drift monitor per-facility and aggregate checks."""

from __future__ import annotations

from ml_pipeline.drift import DriftMonitor

EXPECTED_FACILITY_COUNT = 2


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_check_drift_detects_shifted_distribution() -> None:
    """Large distribution shift should set `needs_retrain` to true."""
    monitor = DriftMonitor()
    result = monitor.check_drift(
        facility_id=1,
        baseline_rmse=1.0,
        recent_actuals=[100, 102, 101, 104, 103, 105],
        recent_predictions=[80, 81, 82, 83, 84, 85],
    )
    _check(result["needs_retrain"] is True, "Shifted distributions should trigger retraining")


def test_check_drift_stable_series() -> None:
    """Stable actual-vs-prediction series should not trigger retraining."""
    monitor = DriftMonitor()
    result = monitor.check_drift(
        facility_id=2,
        baseline_rmse=5.0,
        recent_actuals=[100, 102, 101, 104, 103, 105],
        recent_predictions=[100, 101, 100, 103, 104, 105],
    )
    _check(result["needs_retrain"] is False, "Stable series should not trigger retraining")


def test_check_drift_insufficient_samples() -> None:
    """Very short series should return the insufficient-samples reason."""
    monitor = DriftMonitor()
    result = monitor.check_drift(
        facility_id=3,
        baseline_rmse=1.0,
        recent_actuals=[1, 2],
        recent_predictions=[1, 2],
    )
    _check(result["reason"] == "insufficient_samples", "Short windows should report insufficient_samples")


def test_check_drift_invalid_baseline() -> None:
    """Non-positive baseline should be treated as invalid and non-triggering."""
    monitor = DriftMonitor()
    result = monitor.check_drift(
        facility_id=4,
        baseline_rmse=0.0,
        recent_actuals=[10, 11, 12, 13, 14, 15],
        recent_predictions=[9, 10, 11, 12, 13, 14],
    )
    _check(result["needs_retrain"] is False, "Invalid baseline should not trigger retraining")
    _check(result["reason"] == "baseline_invalid", "Invalid baseline should be reported")


def test_check_all_facilities_summary() -> None:
    """Aggregate check should report evaluated and retrain-target facility ids."""
    monitor = DriftMonitor()
    summary = monitor.check_all_facilities(
        [
            {
                "facility_id": 1,
                "baseline_rmse": 1.0,
                "recent_actuals": [10, 11, 12, 13, 14, 15],
                "recent_predictions": [2, 3, 4, 5, 6, 7],
            },
            {
                "facility_id": 2,
                "baseline_rmse": 5.0,
                "recent_actuals": [10, 11, 12, 13, 14, 15],
                "recent_predictions": [10, 11, 12, 13, 14, 15],
            },
        ],
    )
    _check(summary["facilities_checked"] == EXPECTED_FACILITY_COUNT, "Two facilities should be evaluated")
    _check(1 in summary["retrain_facility_ids"], "Facility 1 should be flagged for retraining")
