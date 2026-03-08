from __future__ import annotations

import pytest

from fuelsense.core import tasks
from fuelsense.core.tests.factories import FacilityFactory


@pytest.mark.django_db
def test_tasks_execute_and_return_shapes():
    facility = FacilityFactory()
    assert "facility_count" in tasks.daily_tick()
    assert "active_facilities" in tasks.ingest_hourly()
    assert tasks.ingest_facility_data(facility.id)["facility_id"] == facility.id
    assert "recent_logs" in tasks.ingestion_complete()
    assert "facility_count" in tasks.run_batch_forecasts([facility.id])
    assert "facility_count" in tasks.run_batch_anomaly_detection([facility.id])
    assert "models" in tasks.check_all_drift()
    assert tasks.retrain_model(facility.id, "DEMAND_FORECAST")["model_type"] == "DEMAND_FORECAST"
    assert "queued" in tasks.run_planning_cycle()
    assert tasks.trigger_emergency_delivery(facility.id)["exists"] is True
