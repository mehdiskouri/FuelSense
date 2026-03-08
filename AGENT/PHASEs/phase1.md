
---

## Plan: Phase 1 — Django Domain Layer, Admin, Synthetic Data & REST API

**TL;DR:** Implement the complete Django domain layer — 12 models with migrations, custom admin frontend with charts/actions/dashboard, deterministic synthetic data generator, full DRF REST API (15+ endpoints), Redis caching, Celery beat schedule, Prometheus business metrics, and tests at >95% coverage. All stubs from Phase 0 get replaced with real implementations.

---

### Sub-phase 1A — Settings & Dependency Updates

*Depends on: Phase 0 complete. Blocks all other sub-phases.*

**Steps**

1. **Add missing Python dependencies to base.txt:**
   - `matplotlib` — for admin inline charts (InventoryLog 30-day chart, Forecast overlay, DeliveryAdmin route map)
   - `django-filter` — for DRF queryset filtering on API endpoints (fuel_type, is_active, status, anomaly_type, etc.)
   - Pin versions on all existing entries per Architecture spec (Django>=5.1, djangorestframework>=3.15, celery>=5.4, etc.) — currently unpinned

2. **Update base.py:**
   - Add `"rest_framework.authtoken"` and `"django_filters"` to `INSTALLED_APPS`
   - Add `"DEFAULT_FILTER_BACKENDS": ["django_filters.rest_framework.DjangoFilterBackend"]` to `REST_FRAMEWORK`
   - Add `CACHES` config: Redis backend via `django.core.cache.backends.redis.RedisCache` (Django 4.0+ native), URL from `REDIS_URL` env var (default `redis://localhost:6379/0`)
   - Add `CELERY_BEAT_SCHEDULE` dict with all 5 periodic tasks: daily_tick (02:00), ingest hourly, forecasts (02:00), anomaly detection (03:00), drift check (04:00), planning cycle (05:00)
   - Add `LOGGING` config: structured logging for Django, Celery, and app-level loggers

3. **Update development.py:**
   - `EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"`
   - Override `CACHES` to use local Redis for dev

4. **Update production.py:**
   - `SECURE_SSL_REDIRECT = True`, `SECURE_HSTS_SECONDS = 31536000`
   - `STATIC_ROOT = "/app/staticfiles"`
   - Env-based `ALLOWED_HOSTS`, `SECRET_KEY`, database and Redis from Kubernetes Secrets

**Relevant files**
- base.txt — add matplotlib, django-filter; pin versions
- base.py — INSTALLED_APPS, REST_FRAMEWORK, CACHES, CELERY_BEAT_SCHEDULE, LOGGING
- development.py — dev overrides
- production.py — security hardening

---

### Sub-phase 1B — 12 Domain Models + Migrations

*Depends on: 1A. Blocks 1C–1H.*

**Steps**

1. **Implement all 12 models in models.py** — replace the empty stub with full model declarations per Architecture + PRD specs:

   - **`FuelType`**: name (CharField 50), unit (CharField 20), density_kg_per_unit (FloatField), hazmat_class (CharField 10, blank=True). `__str__` returns name.
   - **`Facility`**: name (CharField 200), facility_type (CharField, choices: POWER_PLANT/INDUSTRIAL/STORAGE), fuel_type (FK→FuelType, PROTECT), latitude/longitude (FloatField), storage_capacity (FloatField), current_inventory (FloatField), min_safe_inventory (FloatField), dynamic_reorder_point (FloatField, null=True), delivery_window_start/end (TimeField), delivery_days (JSONField, default=list), is_active (BooleanField, default=True). **Meta indexes**: `(fuel_type, is_active)`, `(current_inventory)`. Property `reorder_status` → "OK"/"WARNING"/"CRITICAL" computed from current_inventory vs dynamic_reorder_point vs min_safe_inventory.
   - **`Depot`**: name (CharField 200), latitude/longitude (FloatField), fuel_type (FK→FuelType, PROTECT), fuel_inventory (FloatField), facilities (M2M→Facility through `DepotFacilityAssignment`).
   - **`DepotFacilityAssignment`**: depot (FK→Depot, CASCADE), facility (FK→Facility, CASCADE). Through model for M2M.
   - **`Vehicle`**: depot (FK→Depot, CASCADE, related_name="vehicles"), registration (CharField 50, unique), capacity (FloatField), cost_per_km (FloatField), is_available (BooleanField, default=True).
   - **`InventoryLog`**: facility (FK→Facility, CASCADE, related_name="inventory_logs"), timestamp (DateTimeField), inventory_level (FloatField), consumption (FloatField), temperature (FloatField, null=True), wind_speed (FloatField, null=True), solar_irradiance (FloatField, null=True). **Meta**: index on `(facility, -timestamp)`, unique_together `(facility, timestamp)`.
   - **`Delivery`**: depot (FK→Depot, CASCADE), vehicle (FK→Vehicle, CASCADE), planned_date (DateField), status (CharField, choices: PLANNED/IN_TRANSIT/DELIVERED/FAILED), total_distance_km (FloatField, null=True), total_cost (FloatField, null=True), route_json (JSONField, null=True), solver_time_ms (FloatField, null=True), created_by_planning_cycle (FK→PlanningCycle, SET_NULL, null=True).
   - **`DeliveryItem`**: delivery (FK→Delivery, CASCADE, related_name="items"), facility (FK→Facility, CASCADE), quantity (FloatField), planned_arrival (DateTimeField), actual_arrival (DateTimeField, null=True), sequence (PositiveIntegerField).
   - **`Forecast`**: facility (FK→Facility, CASCADE, related_name="forecasts"), model_version (CharField 100), created_at (DateTimeField, auto_now_add), horizon_days (IntegerField, default=14), predictions_json (JSONField), rmse (FloatField, null=True).
   - **`AnomalyAlert`**: facility (FK→Facility, CASCADE, related_name="alerts"), timestamp (DateTimeField), anomaly_type (CharField, choices: LEAK/THEFT/EQUIPMENT_DEGRADATION/DEMAND_SHIFT/SENSOR_FAULT), score (FloatField), actual_consumption (FloatField), predicted_consumption (FloatField), is_acknowledged (BooleanField, default=False), acknowledged_by (FK→User, SET_NULL, null=True), notes (TextField, blank=True).
   - **`PlanningCycle`**: triggered_at (DateTimeField, auto_now_add), trigger_type (CharField, choices: SCHEDULED/MANUAL/EMERGENCY), facilities_in_queue (IntegerField), deliveries_created (IntegerField), total_distance_km (FloatField), total_cost (FloatField), solver_time_ms (FloatField), baseline_cost (FloatField, null=True), cost_reduction_pct (FloatField, null=True).
   - **`ModelRegistry`**: model_type (CharField, choices: DEMAND_FORECAST/ANOMALY_DETECTOR), facility (FK→Facility, CASCADE, null=True), mlflow_run_id (CharField 100), version (IntegerField), is_active (BooleanField, default=False), trained_at (DateTimeField), training_rmse (FloatField, null=True), validation_rmse (FloatField, null=True), drift_ratio (FloatField, default=1.0), last_drift_check (DateTimeField, null=True).

2. **Generate and apply migrations** — `python manage.py makemigrations core` + `python manage.py migrate`. Verify zero warnings.

**Relevant files**
- models.py — all 12 models with fields, indexes, constraints, choice enums, `__str__`, `reorder_status` property
- fuelsense/core/migrations/ — generated initial migration

---

### Sub-phase 1C — Synthetic Data Generator

*Depends on: 1B. Parallel with 1D, 1E, 1F, 1G.*

**Steps**

1. **consumption.py** — Implement consumption model:
   - `generate_consumption(base_load, day, temperature, day_of_week, rng)` → float
   - Formula: `C(t) = base_load + A_seasonal * sin(2π * t / 365) + temp_response(T) + weekday_factor(dow) + noise`
   - Seasonal amplitude: `0.2 * base_load`
   - `temp_response(T)`: piecewise linear — heating if T<15°C (`(15-T)*0.5*base_load/15`), cooling if T>25°C (`(T-25)*0.3*base_load/15`)
   - Weekday factor: 1.0 weekdays, 0.85 weekends
   - Noise: Gaussian σ = `0.05 * base_load`

2. **weather.py** — Implement weather via AR(1):
   - `generate_weather_series(days, latitude, rng)` → array of (temperature, wind_speed, solar_irradiance) per day
   - Temperature: seasonal mean based on latitude + AR(1) with ρ=0.85
   - Wind speed: mean + AR(1) with ρ=0.6
   - Solar irradiance: day-of-year dependent + cloud cover noise

3. **anomalies.py** — Implement anomaly injection:
   - `AnomalyInjector` class managing active anomalies per facility
   - 5% daily probability per facility to trigger one of 5 types
   - LEAK: +15–30% for 3–7 days. THEFT: +20–50% single day. EQUIPMENT_DEGRADATION: +2%/day for 10–20 days. DEMAND_SHIFT: permanent ±5–15% base_load. SENSOR_FAULT: 0 or NaN for 1–3 days.
   - Returns both modified consumption and anomaly labels (type, start_day, duration)

4. **generator.py** — Orchestration:
   - `SyntheticDataGenerator(facilities=50, days=365, seed=42)`
   - Create 3–4 `FuelType` records (LNG, Diesel, HFO, Natural Gas)
   - Create 5 `Depot` records with geographic distribution
   - Create 50 `Facility` records with base_load ∈ U[100, 500], linked to depots via `DepotFacilityAssignment`, assigned fuel types
   - Create 10 `Vehicle` records per depot (50 total)
   - Per facility per day: generate weather → apply consumption formula → apply anomaly injection → track inventory level (storage_capacity - cumulative consumption + deliveries) → create `InventoryLog` record
   - Deterministic seeding via `numpy.random.default_rng(seed)` — identical output on repeat runs

5. **Update management command** — Replace placeholder with real invocation of `SyntheticDataGenerator`, print progress, report total records created.

**Relevant files**
- consumption.py — `generate_consumption()` with formula components
- weather.py — `generate_weather_series()` with AR(1) processes
- anomalies.py — `AnomalyInjector` with 5 anomaly types + label tracking
- generator.py — `SyntheticDataGenerator` orchestrating 50 facilities × 365 days
- generate_synthetic_data.py — CLI with --facilities, --days, --seed

---

### Sub-phase 1D — DRF REST API (15+ Endpoints)

*Depends on: 1B. Parallel with 1C, 1E, 1F, 1G.*

**Steps**

1. **serializers.py** — Implement all serializers:
   - `FuelTypeSerializer` — all fields
   - `FacilityListSerializer` — id, name, facility_type, fuel_type (nested), lat/lng, current_inventory, dynamic_reorder_point, reorder_status (read-only computed), is_active
   - `FacilityDetailSerializer` — extends list with storage_capacity, min_safe_inventory, delivery_window_start/end, delivery_days, latest_forecast (nested), latest_inventory_log (nested)
   - `InventoryLogSerializer` — timestamp, inventory_level, consumption, temperature, wind_speed, solar_irradiance
   - `ForecastSerializer` — model_version, created_at, horizon_days, predictions_json, rmse
   - `AnomalyAlertSerializer` — all fields, acknowledged_by as username read-only
   - `AnomalyAlertAcknowledgeSerializer` — write: alert_id, notes
   - `DeliveryListSerializer` — id, depot, vehicle (nested), planned_date, status, total_distance_km, total_cost
   - `DeliveryDetailSerializer` — extends list with route_json, solver_time_ms, related items (nested `DeliveryItemSerializer`)
   - `DeliveryItemSerializer` — facility (nested), quantity, planned_arrival, actual_arrival, sequence
   - `DeliveryStatusUpdateSerializer` — write: status (validated against allowed transitions), notes
   - `PlanningCycleSerializer` — all fields read-only
   - `PlanningTriggerSerializer` — write: trigger_type (MANUAL/EMERGENCY), facility_ids (optional list)
   - `ModelRegistrySerializer` — all fields
   - `DashboardKPISerializer` — avg_delivery_cost_last_30d, forecast_accuracy_mape, anomaly_detection_rate, unacknowledged_anomalies_count, facilities_below_reorder, deliveries_in_transit
   - `DriftHeatmapSerializer` — facility_id, facility_name, drift_ratio, model_version

2. **views.py** — Implement all ViewSets:
   - `FacilityViewSet(ModelViewSet)` — list (filter: fuel_type, is_active, facility_type), retrieve (detail with reorder_status), plus `@action` routes:
     - `GET /facilities/{id}/inventory/` — lookback query (default 90 days via `?days=90`)
     - `GET /facilities/{id}/forecasts/` — latest Forecast record
     - `GET /facilities/{id}/alerts/` — paginated, filtered by anomaly_type, is_acknowledged
     - `POST /facilities/{id}/acknowledge/` — mark AnomalyAlert acknowledged, set acknowledged_by=request.user
   - `DeliveryViewSet(ModelViewSet)` — list (filter: status, vehicle, depot), retrieve (detail with items), plus:
     - `POST /deliveries/{id}/update-status/` — status transition with validation
   - `PlanningViewSet(ViewSet)` — two endpoints:
     - `POST /planning/trigger/` — dispatch `run_planning_cycle` or `trigger_emergency_delivery` Celery task, return `{planning_cycle_id, status: "queued"}`
     - `GET /planning/history/` — paginated `PlanningCycle` list, ordered by `-triggered_at`
   - `ModelRegistryViewSet(ModelViewSet)` — list (filter: model_type, is_active, facility), plus:
     - `POST /models/{id}/promote/` — set is_active=True, deactivate others of same model_type+facility
     - `POST /models/{id}/retrain/` — dispatch `retrain_model` Celery task
   - `DashboardViewSet(ViewSet)` — two endpoints:
     - `GET /dashboard/kpis/` — aggregated metrics, cached in Redis (5-min TTL)
     - `GET /dashboard/drift-heatmap/` — per-facility drift ratios array

3. **urls.py** — Replace stub with DRF router:
   - `DefaultRouter` with prefix bindings: `facilities`, `deliveries`, `planning`, `models`, `dashboard`
   - Wire `rest_framework.authtoken.views.obtain_auth_token` at `auth/token/`

4. **Update urls.py:**
   - Add `django_prometheus.urls` for `/metrics` endpoint
   - Ensure `/healthz`, `/readyz` do real DB + Redis connectivity checks (replace plain stubs)

**Relevant files**
- serializers.py — 16+ serializers with validation
- views.py — 5 ViewSets with custom actions
- urls.py — DRF router + auth token endpoint
- urls.py — health probes with real DB/Redis checks, Prometheus /metrics

---

### Sub-phase 1E — Custom Django Admin

*Depends on: 1B. Parallel with 1C, 1D, 1F, 1G.*

**Steps**

1. **admin.py** — Replace stub with full admin registrations:

   - **`FacilityAdmin`**: `list_display` = (name, facility_type, fuel_type, current_inventory, dynamic_reorder_point, reorder_status, is_active). `list_filter` = (fuel_type, facility_type, is_active). Inline: `InventoryLogInline` (tabular, last 10 records). Custom change_view override: render matplotlib chart (last 30 days consumption + inventory_level), forecast overlay if available, anomaly alerts sidebar. Custom action: "Trigger Emergency Delivery" → dispatches `trigger_emergency_delivery.delay(facility_id)`.

   - **`DeliveryAdmin`**: `list_display` = (id, depot, vehicle, planned_date, status, total_cost, total_distance_km). `list_filter` = (status, planned_date). Inline: `DeliveryItemInline` (tabular). Custom change_view: route map image (matplotlib scatter of depot + facility lat/lng with line path), status timeline, cost breakdown (total_cost, baseline_cost, cost_reduction_pct, solver_time_ms).

   - **Custom Dashboard Admin View**: Register as admin custom page at `/admin/fuelsense/dashboard/`. Template renders:
     - Facilities below reorder point (count + table)
     - Active IN_TRANSIT deliveries (count + table)
     - Today's anomaly alerts (paginated by type)
     - Model drift heatmap (table with color-coded drift_ratio cells: green <1.0, yellow 1.0–1.5, red >1.5)
     - KPI cards: avg delivery cost (30d), forecast MAPE, anomaly detection rate, unacknowledged anomalies count

   - **`PlanningCycleAdmin`**: `list_display` = (triggered_at, trigger_type, facilities_in_queue, deliveries_created, total_cost, baseline_cost, cost_reduction_pct). Read-only. Change view shows side-by-side optimized vs baseline cost comparison.

   - **`ModelRegistryAdmin`**: `list_display` = (model_type, facility, version, is_active, trained_at, drift_ratio). `list_filter` = (model_type, is_active). Actions: "Promote to Active" (deactivates others of same type+facility, logs via admin log), "Rollback to Previous" (finds previous version, promotes it).

   - **Register all remaining models**: FuelType, Depot, DepotFacilityAssignment, Vehicle, InventoryLog, DeliveryItem, Forecast, AnomalyAlert — basic ModelAdmin registrations.

2. **Admin templates** — Create `fuelsense/core/templates/admin/core/` with:
   - `facility_chart.html` — template fragment for matplotlib chart embed (base64 PNG in `<img>` tag)
   - `delivery_route_map.html` — template fragment for route visualization
   - `dashboard.html` — full dashboard page extending `admin/base_site.html`

**Relevant files**
- admin.py — FacilityAdmin (inline chart, emergency action), DeliveryAdmin (route map, timeline), Dashboard view (KPIs, heatmap), PlanningCycleAdmin, ModelRegistryAdmin (promote/rollback)
- `fuelsense/core/templates/admin/core/` — dashboard.html, facility_chart.html, delivery_route_map.html (new directory+files)

---

### Sub-phase 1F — Redis Caching Layer + Prometheus Business Metrics

*Depends on: 1B. Parallel with 1C, 1D, 1E, 1G.*

**Steps**

1. **Redis caching** — Implement in fuelsense/core/cache.py (new file):
   - `get_or_set_facility_inventory(facility_id)` → uses `cache:facility:{id}:latest_inventory`, TTL 3600s
   - `get_or_set_reorder_status(facility_id)` → `cache:facility:{id}:reorder_status`, TTL 900s (OK/WARNING/CRITICAL)
   - `get_or_set_dashboard_kpis()` → `cache:dashboard:kpis`, TTL 300s (5 min)
   - Cache invalidation via Django signals: `post_save` on InventoryLog → invalidate `latest_inventory`; `post_save` on Forecast → invalidate forecast cache; `post_save` on AnomalyAlert → invalidate `anomaly_alerts:today`

2. **Django signal handlers** — Implement in fuelsense/core/signals.py (new file):
   - `on_inventory_log_saved` — invalidate facility inventory + reorder status cache
   - `on_forecast_saved` — invalidate facility forecast cache
   - `on_anomaly_alert_saved` — invalidate today's alerts cache
   - Register signals in fuelsense/core/apps.py `ready()` method

3. **Prometheus business metrics** — Implement in fuelsense/core/metrics.py (new file):
   - `active_anomaly_alerts` (Gauge) — unacknowledged count
   - `facilities_below_reorder` (Gauge) — count where inventory ≤ reorder_point
   - `forecast_mape_pct` (Gauge) — recent MAPE across all facilities
   - `celery_task_duration_seconds` (Histogram, labels: task_name, queue)
   - `celery_task_failures_total` (Counter, labels: task_name, queue)
   - `celery_task_success_total` (Counter, labels: task_name, queue)
   - `retraining_triggered_total` (Counter, labels: facility_id, model_type)
   - Periodic gauge refresh via Celery signal hooks or scheduled task

**Relevant files**
- `fuelsense/core/cache.py` — cache helpers with key patterns + TTL (new file)
- `fuelsense/core/signals.py` — Django signal handlers for cache invalidation (new file)
- `fuelsense/core/metrics.py` — Prometheus business metrics (new file)
- fuelsense/core/apps.py — wire signal handlers in `ready()`

---

### Sub-phase 1G — Celery Beat Schedule + Task Stubs

*Depends on: 1B. Parallel with 1C, 1D, 1E, 1F.*

**Steps**

1. **Update tasks.py** — Replace empty stub with typed task declarations that reference models:
   - `daily_tick()` — orchestrator task (stub body: logs "daily_tick invoked", queries active facility IDs, chains remaining tasks). Bound to `default` queue.
   - `ingest_facility_data(facility_id: int)` — stub: logs facility_id, creates dummy InventoryLog. Default queue.
   - `ingestion_complete()` — stub: logs completion. Default queue.
   - `run_batch_forecasts(facility_ids: list[int])` — stub: logs count, would call forecaster service. Default queue.
   - `run_batch_anomaly_detection(facility_ids: list[int])` — stub: logs count, would call anomaly service. Default queue.
   - `check_all_drift()` — stub: logs "drift check", queries ModelRegistry. Training queue.
   - `retrain_model(facility_id: int | None, model_type: str)` — stub: logs params. Training queue.
   - `run_planning_cycle()` — stub: logs "planning cycle", queries facilities below reorder. Planning queue.
   - `trigger_emergency_delivery(facility_id: int)` — stub: logs facility_id. Planning queue.

   All stubs import models and do basic queries to validate model integration. Full logic (httpx calls to ML services, chord/chain orchestration) lands in Phases 3–5.

2. **Verify celery.py** — already configured with 3 queues and autodiscover. No changes needed unless beat schedule must be added here instead of settings.

**Relevant files**
- tasks.py — 9 task declarations with model imports, queue routing, stub bodies
- base.py — CELERY_BEAT_SCHEDULE config (from 1A)

---

### Sub-phase 1H — Tests (>95% Coverage)

*Depends on: 1B–1G complete. Final sub-phase.*

**Steps**

1. **Factory-Boy factories** — Create fuelsense/core/tests/factories.py (new file):
   - `FuelTypeFactory`, `FacilityFactory`, `DepotFactory`, `DepotFacilityAssignmentFactory`, `VehicleFactory`, `InventoryLogFactory`, `DeliveryFactory`, `DeliveryItemFactory`, `ForecastFactory`, `AnomalyAlertFactory`, `PlanningCycleFactory`, `ModelRegistryFactory`
   - All factories with sensible defaults, using `factory.Faker` and `factory.SubFactory`

2. **conftest.py** — Shared fixtures:
   - `api_client` — DRF `APIClient` with authenticated user (token)
   - `admin_client` — Django test client logged in as superuser
   - `sample_facilities` — 5 facilities across 2 depots with fuel types
   - `sample_inventory_logs` — 30 days of logs for one facility

3. **test_models.py** — Model unit tests:
   - All model creation via factories (verify save + retrieve)
   - `Facility.reorder_status` property: OK when inventory > reorder_point, WARNING when near, CRITICAL when < min_safe_inventory
   - InventoryLog unique constraint enforcement (duplicate facility+timestamp raises IntegrityError)
   - All ForeignKey cascading behavior (CASCADE, PROTECT, SET_NULL)
   - Choice field validation (invalid status rejected)
   - Index existence via `_meta.indexes` inspection

4. **test_api.py** — API integration tests:
   - All 15+ endpoints: correct HTTP status codes (200 for GET, 200/201 for POST, 404 for missing, 400 for invalid)
   - `GET /facilities/` with filtering (fuel_type, is_active) — verify filtered results
   - `GET /facilities/{id}/inventory/` — verify date range filtering
   - `POST /facilities/{id}/acknowledge/` — verify is_acknowledged=True, acknowledged_by set
   - `POST /deliveries/{id}/update-status/` — verify state transition (PLANNED→IN_TRANSIT, IN_TRANSIT→DELIVERED)
   - `POST /planning/trigger/` — verify task dispatched (mock Celery)
   - `POST /models/{id}/promote/` — verify is_active toggle, others deactivated
   - `GET /dashboard/kpis/` — verify response shape and caching
   - `GET /dashboard/drift-heatmap/` — verify per-facility array
   - Authentication: unauthenticated requests → 401/403
   - Pagination: verify page_size=50, next/previous links

5. **test_admin.py** — Admin tests:
   - All admin list views render (200 status) for all registered models
   - FacilityAdmin change view renders with chart (mock matplotlib output)
   - DeliveryAdmin change view renders with route map
   - Dashboard custom page renders at `/admin/fuelsense/dashboard/` with KPI cards
   - "Trigger Emergency Delivery" action queues Celery task (mock)
   - ModelRegistryAdmin "Promote" action: toggles is_active, deactivates siblings
   - ModelRegistryAdmin "Rollback" action: previous version becomes active

6. **test_tasks.py** — Task tests:
   - Each task callable without error (stub behavior — logs, queries models)
   - `daily_tick` queries active facilities
   - `trigger_emergency_delivery` with valid facility_id succeeds
   - Queue routing: verify each task routes to correct queue

7. **Synthetic data tests** — Create fuelsense/synthetic/tests/ (new directory):
   - `test_consumption.py` — verify formula: seasonal component varies sinusoidally, weekday factor applied, noise σ ≈ 0.05*base_load
   - `test_weather.py` — AR(1) autocorrelation check (lag-1 correlation ≈ ρ), temperature seasonal variation
   - `test_anomalies.py` — each anomaly type produces expected modification (LEAK → 15-30% increase, THEFT → 20-50% spike, etc.)
   - `test_generator.py` — deterministic seeding (same seed → identical output), correct record counts (50 facilities, 18250 InventoryLog records, ~5% anomaly rate)
   - `test_management_command.py` — `call_command("generate_synthetic_data", "--facilities", "5", "--days", "10", "--seed", "42")` creates expected records

8. **Cache + metrics tests** — in test_api.py or separate file:
   - Dashboard KPIs response cached after first call (verify Redis key exists)
   - Signal-driven cache invalidation (save InventoryLog → invalidate inventory cache)

**Relevant files**
- `fuelsense/core/tests/factories.py` — 12 factory-boy factories (new file)
- conftest.py — shared fixtures (api_client, admin_client, sample data)
- test_models.py — model unit tests
- test_api.py — API integration tests (all 15+ endpoints)
- test_admin.py — admin view tests (chart rendering, actions, dashboard)
- test_tasks.py — task stub tests
- `fuelsense/synthetic/tests/` — test_consumption.py, test_weather.py, test_anomalies.py, test_generator.py, test_management_command.py (new files)

---

### Verification

1. `python manage.py makemigrations --check` — no pending migrations
2. `python manage.py migrate` — zero warnings
3. `python manage.py check` — system check passes
4. `python manage.py generate_synthetic_data --facilities 50 --days 365 --seed 42` — produces deterministic output (run twice, compare counts)
5. `pytest fuelsense/ --cov --cov-fail-under=95` — all tests pass, coverage > 95%
6. `make lint` — ruff passes on all new code
7. `make typecheck` — mypy passes (strict mode for core, relaxed for synthetic math)
8. All 15+ API endpoints return correct HTTP status codes (verified by test_api.py)
9. Admin dashboard renders at `/admin/fuelsense/dashboard/` — facilities below reorder, anomaly alerts, KPI cards, drift heatmap all populated after synthetic data generation
10. FacilityAdmin shows 30-day inline chart + forecast overlay after synthetic data
11. DeliveryAdmin shows route map after planning cycle creates deliveries
12. ModelRegistryAdmin promote/rollback actions execute with audit trail in admin log
13. Redis cache keys populated: `cache:facility:*:latest_inventory`, `cache:dashboard:kpis` (verify via `redis-cli KEYS 'cache:*'`)
14. Celery worker starts: `celery -A fuelsense worker -Q default` — no import errors, tasks discovered
15. CI `test-django` job activated (replace stub) and green

---

### Decisions

- **matplotlib for admin charts** (not basemap): Route maps rendered via matplotlib scatter/line plots with lat/lng coordinates. `basemap` is deprecated — plain matplotlib with marker annotations provides the route visualization without the dependency burden. If high-fidelity map backgrounds are needed later, `cartopy` can be added.
- **django-filter for API filtering**: Standard DRF filter backend instead of manual queryset filtering.
- **Coverage target > 95%**: Per user requirement. The PRD mentions 85% in one place but the roadmap and user explicitly require 95%.
- **Task stubs in Phase 1, full logic in Phases 3–5**: Tasks reference models and do basic queries, but httpx calls to ML services and chord/chain orchestration are not wired until the ML services exist. This avoids import errors and allows task tests to validate model integration.
- **Admin chart rendering**: Server-side matplotlib → base64 PNG embedded in HTML template. No JavaScript charting library needed — keeps admin frontend pure Django.
- **Predictions JSON format**: Architecture uses `{day, p10, p50, p90}` (quantile), PRD uses `{day, value, lower, upper}`. **Go with Architecture's quantile format** (`p10`, `p50`, `p90`) since that's what the TCN model outputs. The PRD's `value` = `p50`, `lower` = `p10`, `upper` = `p90`.

### Further Considerations

1. **PostgreSQL + Redis local availability**: Phase 1 tests require a running PostgreSQL and Redis. The Makefile `test-django` target should document this prerequisite, or tests should use SQLite fallback for unit tests and mark integration tests requiring PG/Redis with `@pytest.mark.integration`. **Recommendation**: Use pytest-django's `--reuse-db` flag + local PG for all tests (native machine has PG available), mark Redis-dependent tests with `@pytest.mark.redis`.

2. **Synthetic data volume for tests**: Full generation (50 × 365 = 18,250 rows) is slow for unit tests. **Recommendation**: Test with `--facilities 3 --days 10` in test suite (fast), full 50 × 365 only in the management command integration test or `@pytest.mark.slow`.

3. **Admin template directory**: Phase 0 didn't create `fuelsense/core/templates/`. This directory and `DIRS` setting in `TEMPLATES` config need to be created in 1E. Alternatively, use `APP_DIRS=True` (already set) and place templates under `fuelsense/core/templates/admin/core/`.