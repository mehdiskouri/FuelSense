from __future__ import annotations

from ml_pipeline.drift import DriftMonitor


def test_check_drift_detects_shifted_distribution() -> None:
    monitor = DriftMonitor()
    result = monitor.check_drift(
        facility_id=1,
        baseline_rmse=1.0,
        recent_actuals=[100, 102, 101, 104, 103, 105],
        recent_predictions=[80, 81, 82, 83, 84, 85],
    )
    assert result["needs_retrain"] is True


def test_check_drift_stable_series() -> None:
    monitor = DriftMonitor()
    result = monitor.check_drift(
        facility_id=2,
        baseline_rmse=5.0,
        recent_actuals=[100, 102, 101, 104, 103, 105],
        recent_predictions=[100, 101, 100, 103, 104, 105],
    )
    assert result["needs_retrain"] is False


def test_check_drift_insufficient_samples() -> None:
    monitor = DriftMonitor()
    result = monitor.check_drift(
        facility_id=3,
        baseline_rmse=1.0,
        recent_actuals=[1, 2],
        recent_predictions=[1, 2],
    )
    assert result["reason"] == "insufficient_samples"


def test_check_all_facilities_summary() -> None:
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
        ]
    )
    assert summary["facilities_checked"] == 2
    assert 1 in summary["retrain_facility_ids"]
