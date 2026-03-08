

---

## Plan: FuelSense — Comprehensive Implementation Roadmap

**TL;DR:** 10-phase rigorous implementation plan covering every component specified in the Architecture and PRD documents. Dual CPU/GPU compute is a first-class requirement throughout. Forecast strategy covers both global and facility-specialized models. Django admin frontend, Prometheus/Grafana observability frontend, mTLS via service mesh, Kubernetes NetworkPolicies, and all 13 services are fully in scope. Test coverage target is > 95%. CI/CD builds progressively from Phase 0 and includes both `ci.yml` and `gpu-tests.yml`. No items omitted.

---

### Phase 0 — Engineering Foundation, Toolchain & CI Skeleton

*Depends on: nothing. Prerequisite for all subsequent phases.*

**Steps**

1. **Repository scaffolding** — Create the complete project directory structure per Architecture §11 (every directory, every `__init__.py`): `fuelsense/` (Django project with `settings/`, `core/`, `synthetic/`), `fuelsense_common/`, `forecaster/` (with `backends/`, `tests/`), `anomaly/` (with `tests/`), `optimizer/` (with `backends/`, `tests/`), `ml_pipeline/` (with `tests/`), `charts/fuelsense/` (with full `templates/` tree — 23 template files), `infra/` (with `grafana/dashboards/`), `.github/workflows/`, `requirements/`.

2. **Dependency pinning** — Create all 7 requirements files:
   - `requirements/base.txt` — Django 5.1+, DRF 3.15+, Celery 5.4+, django-celery-beat, httpx, psycopg2-binary, redis, gunicorn, prometheus-client, django-prometheus, factory-boy
   - `requirements/forecaster-cpu.txt` — PyTorch 2.2+ (CPU), FastAPI 0.110+, uvicorn, pydantic, numpy, prometheus-client
   - `requirements/forecaster-gpu.txt` — PyTorch 2.2+ (CUDA), FastAPI 0.110+, uvicorn, pydantic, numpy, prometheus-client, pynvml
   - `requirements/anomaly.txt` — scikit-learn 1.4+, FastAPI, uvicorn, joblib, numpy, prometheus-client
   - `requirements/optimizer-cpu.txt` — Google OR-Tools 9.9+, FastAPI, uvicorn, numpy, prometheus-client
   - `requirements/optimizer-gpu.txt` — OR-Tools 9.9+, PyTorch 2.2+ (CUDA), FastAPI, uvicorn, numpy, prometheus-client
   - `requirements/dev.txt` — pytest, pytest-django, pytest-cov, pytest-asyncio, factory-boy, ruff, mypy, httpx (test client), hypothesis (property tests)

3. **Toolchain lockdown** — Configure `pyproject.toml`: ruff (lint + format), mypy (strict), pytest as runner, coverage settings. Pin Python 3.12. Add pre-commit hooks: ruff-check, ruff-format, mypy, pytest smoke.

4. **CI skeleton** — Create `.github/workflows/ci.yml` with all 5 job stubs (test-django, test-ml-services, build-images, deploy-staging, smoke-test) per Architecture §12. Create `.github/workflows/gpu-tests.yml` with `workflow_dispatch` trigger for GPU-specific tests. Wire only lint + type-check initially — tests and builds activate as code lands.

5. **Environment configuration** — Create `.env.example` with all `FUELSENSE_*` variables from Architecture §3.3: `FUELSENSE_DEVICE`, `FUELSENSE_CUDA_MEMORY_FRACTION`, `FUELSENSE_CUDA_ALLOW_TF32`, `FUELSENSE_CUDA_BENCHMARK`, all batch size variants (CPU/GPU for forecast + training), `FUELSENSE_OPTIMIZER_BACKEND`, `FUELSENSE_OPTIMIZER_TIME_LIMIT_MS`. Add `.env` to `.gitignore`. Database URLs, Redis URLs, Celery broker, MLflow URI, secret key.

6. **Makefile** — Implement all 17 targets from Architecture §11.1: `help`, `dev`, `dev-gpu`, `down`, `test`, `test-django`, `test-ml`, `test-optimizer`, `lint`, `migrate`, `seed`, `train-all`, `k8s-local`, `k8s-local-gpu`, `k8s-down`, `bench-forecaster`, `bench-optimizer`.

7. **Branch strategy** — Define branch protection: require passing CI, require 1 approval, no force-push to main.

**Relevant files**
- Full directory tree per Architecture §11 (every package, every template placeholder)
- `pyproject.toml` — ruff, mypy, pytest, coverage config
- `.github/workflows/ci.yml` — 5-job CI pipeline skeleton
- `.github/workflows/gpu-tests.yml` — GPU test workflow with manual trigger
- `Makefile` — all 17 development workflow targets
- `.env.example` — all environment variables with defaults and descriptions
- `requirements/*.txt` — 7 pinned dependency files

**Gate**
- `make lint` passes on empty project
- CI triggers on push, reports green for lint/type stage
- `docker compose config` validates syntax
- All 17 Makefile targets defined (stubs where code not yet present)
- `.env.example` documents every `FUELSENSE_*` variable

---

### Phase 1 — Django Domain Model, Admin Frontend & Synthetic Data

*Depends on: Phase 0. Blocking for all subsequent phases.*

**Steps**

1. **Django settings split** — Implement `fuelsense/settings/base.py`, `development.py`, `production.py`. Configure: PostgreSQL via `DATABASE_URL`, Redis via `REDIS_URL`, Celery broker/result backend, static files config, logging config, `INSTALLED_APPS` (django-celery-beat, django-prometheus, rest_framework), middleware stack, `AUTH_USER_MODEL`.

2. **All 12 Django models** — Implement in `fuelsense/core/models.py` exactly as specified in PRD §4.1:
   - `FuelType` (name, unit, density_kg_per_unit, hazmat_class)
   - `Facility` (name, facility_type choices, fuel_type FK, lat/lng, storage_capacity, current_inventory, min_safe_inventory, dynamic_reorder_point, delivery_window_start/end, delivery_days JSONField, is_active; indexes on [fuel_type, is_active] and [current_inventory])
   - `Depot` (name, lat/lng, fuel_type FK, fuel_inventory, M2M to Facility through `DepotFacilityAssignment`)
   - `DepotFacilityAssignment` (through model)
   - `Vehicle` (depot FK, registration unique, capacity, cost_per_km, is_available)
   - `InventoryLog` (facility FK, timestamp, inventory_level, consumption, temperature, wind_speed, solar_irradiance; index on [facility, -timestamp], unique_together [facility, timestamp])
   - `Delivery` (depot FK, vehicle FK, planned_date, status choices [PLANNED/IN_TRANSIT/DELIVERED/FAILED], total_distance_km, total_cost, route_json, solver_time_ms, created_by_planning_cycle FK)
   - `DeliveryItem` (delivery FK, facility FK, quantity, planned_arrival, actual_arrival, sequence)
   - `Forecast` (facility FK, model_version, created_at auto, horizon_days default 14, predictions_json, rmse)
   - `AnomalyAlert` (facility FK, timestamp, anomaly_type choices, score, actual_consumption, predicted_consumption, is_acknowledged, acknowledged_by User FK, notes)
   - `PlanningCycle` (triggered_at auto, trigger_type choices [SCHEDULED/MANUAL/EMERGENCY], facilities_in_queue, deliveries_created, total_distance_km, total_cost, solver_time_ms, baseline_cost, cost_reduction_pct)
   - `ModelRegistry` (model_type choices [DEMAND_FORECAST/ANOMALY_DETECTOR], facility FK nullable, mlflow_run_id, version, is_active, trained_at, training_rmse, validation_rmse, drift_ratio default 1.0, last_drift_check)
   - Run `makemigrations` + `migrate`.

3. **Custom Django Admin frontend** — Implement `fuelsense/core/admin.py` with full admin customization per PRD §4.1:
   - **`FacilityAdmin`**: inline `InventoryLog` chart (last 30 days consumption + inventory rendered via matplotlib), forecast overlay on chart, anomaly alert list (inline or sidebar), custom action "Trigger Emergency Delivery" that dispatches `trigger_emergency_delivery` Celery task.
   - **`DeliveryAdmin`**: route map rendering (static image generated server-side via matplotlib + basemap showing depot → stop → stop → depot), status timeline visualization, cost breakdown panel.
   - **Dashboard custom admin view** (registered as custom admin page, not a model): facilities below reorder point list, active deliveries in transit count and table, today's anomaly alerts feed, model drift status heatmap (facility × drift_ratio color-coded), KPI cards (avg delivery cost, forecast accuracy/MAPE, anomaly detection rate, route cost reduction %).
   - **`PlanningCycleAdmin`**: side-by-side comparison of optimized vs baseline cost, with percentage savings highlighted, facilities_in_queue, solver_time_ms display.
   - **`ModelRegistryAdmin`**: promote action (sets is_active=True, deactivates previous version for same facility+model_type), rollback action (revert to previous version), confirmation dialog, audit trail via admin log.
   - Register all models (FuelType, Depot, Vehicle, InventoryLog, DeliveryItem, Forecast, AnomalyAlert).

4. **Synthetic data generator** — Build `fuelsense/synthetic/` package with 4 modules per PRD §6:
   - `consumption.py` — Facility consumption model: `C(t) = base_load + seasonal_component(t) + temperature_response(T(t)) + weekday_factor(t) + noise(t) + anomaly_injection(t)`. Seasonal: sinusoidal period=365d. Temp response: piecewise linear (heating if T<15°C, cooling if T>25°C). Weekday: 0.85 on weekends. Noise: Gaussian σ=0.05×base_load.
   - `weather.py` — Synthetic weather via AR(1) process: temperature (ρ=0.8), wind speed (ρ=0.6), solar irradiance (day-of-year dependent). Location-based seasonal norms from latitude.
   - `anomalies.py` — 5% daily probability per facility of one of: LEAK (15-30% increase, 3-7 days), THEFT (20-50% single-day spike), EQUIPMENT_DEGRADATION (2% per day cumulative, 10-20 days), DEMAND_SHIFT (permanent step change in base_load), SENSOR_FAULT (0 or NaN, 1-3 days). Labels stored separately for anomaly classifier training.
   - `generator.py` — Orchestration: create 50 facilities across 5 depots with 10 vehicles each, generate 365 days of inventory logs, track deliveries, produce labeled anomaly dataset. **Deterministic seeding** for test reproducibility.
   - Management command: `fuelsense/synthetic/management/commands/generate_synthetic_data.py` with flags `--facilities 50 --days 365 --seed 42`.

5. **DRF REST API** — Implement all 15 endpoints per PRD §4.1 plus the additional dashboard endpoint:
   - `fuelsense/core/serializers.py` — Serializers for all models with nested relations, validation rules.
   - `fuelsense/core/views.py` — `FacilityViewSet` (list with filtering/pagination, detail with reorder status, nested /inventory/, /forecasts/, /alerts/ routes, /acknowledge/ action), `DeliveryViewSet` (list with status filtering, detail with route+items, /update-status/ action), `PlanningViewSet` (/trigger/ to manually dispatch planning cycle, /history/ with metrics), `ModelViewSet` (list with drift status, /promote/ action, /retrain/ action dispatching Celery task), `DashboardViewSet` (/kpis/ aggregated metrics, /drift-heatmap/ per-facility drift ratios).
   - `fuelsense/core/urls.py` — Router configuration for all 15+ endpoints under `/api/v1/`.
   - `fuelsense/urls.py` — Wire API urls, admin urls, health probes (/healthz, /readyz).
   - **Authentication**: Django session auth for admin, DRF `TokenAuthentication` for API consumers.

6. **Redis caching layer** — Implement caching per PRD §4.4:
   - `cache:facility:{id}:latest_inventory` → float
   - `cache:facility:{id}:reorder_status` → "OK" | "WARNING" | "CRITICAL"
   - `cache:dashboard:kpis` → JSON, refresh every 5 minutes
   - Cache invalidation on model save signals.

7. **Celery bootstrap** — Configure `fuelsense/celery.py`: Redis broker, 3 queues (`default`, `training`, `planning`), `django-celery-beat` for periodic schedule (ingest hourly, forecasts daily 02:00, anomaly 03:00, drift 04:00, planning 05:00). Define task stubs for all 14 tasks.

8. **Prometheus metrics for Django** — Integrate django-prometheus middleware. Implement custom metrics: `http_request_duration_seconds` (histogram by endpoint), `celery_task_duration_seconds` (histogram by task), `celery_task_failures_total` (counter), `active_anomaly_alerts` (gauge), `facilities_below_reorder` (gauge).

9. **Tests** — Factory-boy factories for all 12 models. Unit tests for model methods, custom querysets, reorder point computation, serializer validation. Integration tests for all 15 API endpoints (request/response cycles). Admin tests: all custom views render, all actions execute, dashboard KPIs display, FacilityAdmin inline chart renders, DeliveryAdmin route map renders, ModelRegistryAdmin promote/rollback works, PlanningCycleAdmin comparison view works. **Coverage target > 95%.** Enable `test-django` CI job.

**Relevant files**
- `fuelsense/settings/base.py`, `development.py`, `production.py`
- `fuelsense/core/models.py` — all 12 domain models with all indexes and constraints
- `fuelsense/core/admin.py` — FacilityAdmin (inline charts, emergency action), DeliveryAdmin (route map, timeline), Dashboard view (KPIs, heatmap), PlanningCycleAdmin (comparison), ModelRegistryAdmin (promote/rollback)
- `fuelsense/core/serializers.py` + `views.py` + `urls.py` — full DRF layer, 15+ endpoints
- `fuelsense/synthetic/` — consumption.py, weather.py, anomalies.py, generator.py + management command
- `fuelsense/celery.py` — app config with 3 queues + beat schedule
- `fuelsense/core/tests/` — test_models.py, test_api.py, test_admin.py, test_tasks.py (> 95% coverage)

**Gate**
- `python manage.py migrate` succeeds zero warnings
- `python manage.py generate_synthetic_data --facilities 50 --days 365 --seed 42` produces deterministic output (identical data on repeat runs with same seed)
- All 15 API endpoints return correct HTTP status codes with valid response shapes
- Admin dashboard renders: facility inventory charts, reorder point list, anomaly alerts, drift heatmap, KPI cards
- FacilityAdmin inline InventoryLog chart shows last 30 days
- DeliveryAdmin route map renders via matplotlib+basemap
- ModelRegistryAdmin promote/rollback executes with audit trail
- PlanningCycleAdmin shows optimized vs baseline cost
- Redis cache populated on data changes
- `pytest fuelsense/ --cov` > 95%
- CI `test-django` job green

---

### Phase 2 — Compute Abstraction, Backend Registry & Shared Contracts

*Depends on: Phase 0. Parallel with late Phase 1. Blocking for Phases 3–5.*

**Steps**

1. **`fuelsense_common/compute.py`** — Implement exactly as Architecture §3.1:
   - `DeviceType` enum: `CPU = "cpu"`, `CUDA = "cuda"`
   - `resolve_device()`: priority chain (1: `FUELSENSE_DEVICE` env var, 2: CUDA availability via `torch.cuda.is_available()`, 3: CPU fallback). Log warnings on CUDA request without availability.
   - `ComputeBackend` protocol (`@runtime_checkable`): `device: DeviceType`, `warmup() -> None`, `health_check() -> dict`.

2. **`fuelsense_common/registry.py`** — Implement exactly as Architecture §3.2:
   - `_registry: Dict[str, Dict[DeviceType, Type[ComputeBackend]]]`
   - `register_backend(name: str, device: DeviceType)` decorator
   - `get_backend(name: str, device: DeviceType)` factory with CPU fallback when requested device unavailable. `RuntimeError` if no backend registered for name at all.

3. **`fuelsense_common/schemas.py`** — Define shared Pydantic request/response models for all 3 ML service contracts:
   - Forecaster: `ForecastRequest` (facility_id, lookback [90×6] with min/max_length validation), `QuantilePrediction` (day, p10, p50, p90), `ForecastResponse` (facility_id, forecast list, model_version, inference_time_ms, device), `BatchForecastRequest` (requests list), `BatchForecastResponse` (responses list, total_inference_time_ms, device)
   - Anomaly: detect request (facility_id, actual_consumption, predicted_consumption, features dict), detect response (is_anomaly, z_score, anomaly_type, confidence, if_score, stage)
   - Optimizer: optimize request (depot, vehicles, stops with time windows and service times, max_route_duration, objective_weights), optimize response (routes with stops/ETAs/costs, totals, vehicles_used, solver_time_ms, baseline_cost, cost_reduction_pct, status)
   - `HealthResponse` (status, device dict)

4. **Service template pattern** — Establish the canonical FastAPI service pattern: lifespan context manager for backend init, `/health` endpoint returning `HealthResponse`, `/metrics` endpoint exposing Prometheus metrics via `generate_latest()`, structured logging. This pattern is the blueprint applied identically in `forecaster/service.py`, `anomaly/service.py`, `optimizer/service.py`.

5. **Tests** — Test `resolve_device()` with mocked env vars and mocked torch availability (all branches: explicit cpu, explicit cuda with available GPU, explicit cuda without GPU fallback, auto-detect with GPU, auto-detect without torch). Test registry: decoration, lookup, CPU fallback, missing backend RuntimeError. Test all Pydantic schema validation (reject malformed, accept valid, boundary values). **Coverage > 95%.**

**Relevant files**
- `fuelsense_common/compute.py` — `DeviceType`, `resolve_device()`, `ComputeBackend` protocol
- `fuelsense_common/registry.py` — `register_backend()`, `get_backend()`, `_registry`
- `fuelsense_common/schemas.py` — All shared Pydantic request/response models for all 3 services

**Gate**
- `resolve_device()` correctly handles all 5 branches (explicit cpu, explicit cuda available, explicit cuda fallback, auto cuda, auto cpu)
- `register_backend` + `get_backend` round-trip for mock backends on both CPU and CUDA
- `get_backend` falls back to CPU when CUDA requested but only CPU registered
- `get_backend` raises `RuntimeError` when no backend registered for name
- All Pydantic schemas validate and reject correctly
- Tests > 95% coverage on `fuelsense_common/`
- CI green for common tests

---

### Phase 3 — Demand Forecaster: Dual CPU/GPU Implementation + Training Pipeline + MLflow

*Depends on: Phases 1–2. Partial parallelism with Phases 4–5 once `model.py` done.*

**Steps**

1. **TCN model architecture** — Implement `forecaster/model.py` exactly as Architecture §4.1:
   - `CausalConv1d`: Conv1d with causal padding `(kernel_size - 1) * dilation`, trim future leakage in forward pass.
   - `TCNBlock`: `CausalConv1d → BatchNorm1d → ReLU → Dropout(0.2) → CausalConv1d → BatchNorm1d → ReLU → Dropout(0.2) → residual`. Residual connection with 1×1 conv if in_ch != out_ch.
   - `DemandTCN`: N_FEATURES=6, LOOKBACK=90, HORIZON=14, N_QUANTILES=3. 3 TCNBlock layers with dilations [1, 2, 4], hidden_channels=32. Output head: `Linear(32→128) → ReLU → Dropout → Linear(128→14×3)`. Receptive field = 28 days. Input `[batch, 90, 6]` permuted to `[batch, 6, 90]` for TCN, last timestep → output head → reshape to `[batch, 14, 3]`.
   - `QuantileLoss`: pinball loss at quantiles [0.1, 0.5, 0.9]. Formula: `q * max(y - ŷ, 0) + (1-q) * max(ŷ - y, 0)`.

2. **CPU backend** — Implement `forecaster/backends/cpu_backend.py` exactly as Architecture §4.2:
   - `@register_backend("demand_forecaster", DeviceType.CPU)` class `CPUForecaster`
   - `__init__`: `torch.set_num_threads(4)`, `torch.set_float32_matmul_precision("medium")`
   - `warmup()`: dummy forward pass `[1, 90, 6]`
   - `health_check()`: device, threads, mkl_available, model_loaded
   - `load_model(state_dict_path)`: load state_dict with `map_location="cpu"`, eval mode, warmup
   - `predict(lookback: np.ndarray)`: `[n, 90, 6]` → torch tensor → `model(x)` → numpy
   - `train(...)`: full training loop — epochs=100, batch_size=32, AdamW(lr=1e-3, weight_decay=1e-4), CosineAnnealingLR, gradient clipping norm=1.0, early stopping patience=10, returns history + best_state_dict

3. **GPU backend** — Implement `forecaster/backends/gpu_backend.py` exactly as Architecture §4.3:
   - `@register_backend("demand_forecaster", DeviceType.CUDA)` class `CUDAForecaster`
   - `__init__`: `torch.device("cuda:0")`, `torch.cuda.Stream`, `cudnn.benchmark=True`, `cuda.matmul.allow_tf32=True`, `cudnn.allow_tf32=True`, `torch.cuda.set_per_process_memory_fraction` from env var
   - `warmup()`: dummy forward pass on GPU with `amp.autocast()`, `torch.cuda.synchronize()`
   - `health_check()`: device, gpu_name, gpu_memory_free/total_gb, gpu_utilization (via pynvml), model_loaded, cudnn_benchmark, tf32_enabled
   - `load_model(state_dict_path)`: load to CUDA device, `torch.compile(mode="reduce-overhead")` with fallback, warmup
   - `predict(lookback)`: pinned memory → `non_blocking` H2D on CUDA stream → AMP autocast forward → D2H → stream.synchronize
   - `train(...)`: batch_size=512, pinned memory DataLoader (num_workers=2, persistent_workers=True), AMP autocast + GradScaler, `optimizer.zero_grad(set_to_none=True)`, gradient clipping, early stopping, best state to CPU clone

4. **FastAPI service** — Implement `forecaster/service.py` exactly as Architecture §4.4:
   - Lifespan: `resolve_device()` → `get_backend("demand_forecaster", device)` → `load_model()` from `MODEL_PATH` env → set `DEVICE_INFO` gauge
   - `POST /predict` — single facility inference with `INFERENCE_LATENCY` histogram observation
   - `POST /predict/batch` — batch inference, single forward pass for all facilities, per-facility responses
   - `GET /health` — `HealthResponse` with backend `health_check()`
   - `GET /metrics` — Prometheus `generate_latest()`
   - Prometheus metrics: `forecaster_inference_latency_ms` (histogram, buckets=[0.5,1,2,5,10,25,50,100]), `forecaster_model_version` (gauge), `forecaster_device_gpu` (gauge)

5. **Feature engineering** — Implement `fuelsense/core/features.py`:
   - `build_lookback_matrix(facility_ids)`: query `InventoryLog` for 90-day windows, construct `[n, 90, 6]` arrays with (consumption, temperature, wind_speed, solar_irradiance, dow_sin, dow_cos)
   - `extract_training_data(facility_id)`: extract train/val/test splits (temporal: last 14d=test, previous 14d=val, rest=train), return dict with data + targets arrays
   - `build_drift_data()`: for all active models, query recent actuals + predictions, return list of dicts for DriftMonitor

6. **MLflow training pipeline** — Implement `ml_pipeline/training.py` exactly as Architecture §7.2:
   - `ForecastTrainer` class with `EXPERIMENT_NAME = "demand-forecaster"`, `__init__` sets tracking URI and experiment
   - `train_and_register(facility_id, train_data, train_targets, val_data, val_targets, test_data, test_targets)`:
     - `mlflow.start_run(run_name=f"facility_{facility_id or 'global'}")`
     - `mlflow.log_params()`: facility_id, device, lookback_days, horizon_days, n_features, hidden_channels, kernel_size, dilations, dropout, lr, weight_decay, batch_size, train/val/test sample counts
     - Per-epoch: `mlflow.log_metrics()` with step: train_loss, val_loss, learning_rate
     - Final: `mlflow.log_metrics()`: test_rmse, test_mape, best_val_loss, epochs_trained
     - `mlflow.pytorch.log_model()` with `registered_model_name=f"demand-forecaster-{facility_id or 'global'}"`
     - `mlflow.set_tags()`: device, facility_id, data_hash (SHA-256 of training data, first 12 chars)
   - **Both global model AND facility-specialized models**: global model trained on all facilities combined, per-facility models trained on individual facility data. Both registered in MLflow with separate model names. Promotion logic compares per-facility model against global model for each facility.

7. **Drift detection** — Implement `ml_pipeline/drift.py` exactly as Architecture §7.3:
   - `DriftMonitor` with `DRIFT_THRESHOLD=1.5`, `WINDOW_DAYS=7`, `MIN_SAMPLES=5`
   - `check_drift(facility_id, baseline_rmse, recent_actuals, recent_predictions)`: compute RMSE ratio, return `needs_retrain` bool + diagnostics (residual_mean, residual_std, residual_autocorrelation, max_absolute_error)
   - `check_all_facilities(facility_drift_data)`: iterate all facilities, return summary with retrain_facility_ids list, drift_ratio_mean/max/p90, per_facility results

8. **Celery tasks** — Wire in `fuelsense/core/tasks.py`:
   - `run_batch_forecasts(facility_ids)`: build lookback matrix → httpx POST to `http://demand-forecaster:8001/predict/batch` → create `Forecast` records → update `Facility.dynamic_reorder_point` (3-day lead time p90 × 1.1 safety margin)
   - `check_all_drift()`: build drift data → `DriftMonitor.check_all_facilities()` → dispatch `retrain_model.delay(fid, "DEMAND_FORECAST")` for each facility needing retrain
   - `retrain_model(facility_id, model_type)`: extract training data → `ForecastTrainer.train_and_register()` → compare against current active model → auto-promote if improved (deactivate old, create new `ModelRegistry` entry with incremented version) or log rejection

9. **Tests** — Comprehensive test suite:
   - `forecaster/tests/test_model.py`: forward pass shape `[B, 90, 6] → [B, 14, 3]` for arbitrary B, CausalConv1d no future leakage, QuantileLoss correctness, quantile ordering (p10 ≤ p50 ≤ p90 after training)
   - `forecaster/tests/test_cpu_backend.py`: load_model round-trip, predict shape, train convergence on small synthetic data, warmup, health_check fields
   - `forecaster/tests/test_gpu_backend.py`: same tests with CUDA device placement, AMP verification, pinned memory, torch.compile (skip if no GPU in CI, run via `gpu-tests.yml`)
   - `forecaster/tests/test_service.py`: contract tests for /predict and /predict/batch (valid request→valid response, malformed request→422, model not loaded→503), /health, /metrics
   - `ml_pipeline/tests/test_training.py`: full train_and_register flow on small data, verify MLflow run created with correct params/metrics/tags, verify model artifact logged, verify registered model name pattern
   - `ml_pipeline/tests/test_drift.py`: drift detected when distribution shifted (ratio > 1.5 → needs_retrain=True), no drift on stable data (ratio ~1.0), insufficient data handling (<5 samples), diagnostics computed correctly
   - **Targets: MAPE < 8% on synthetic test set. Coverage > 95%.**
   - Enable both `test-ml-services` (forecaster path) and `gpu-tests.yml` CI jobs

**Relevant files**
- `forecaster/model.py` — `CausalConv1d`, `TCNBlock`, `DemandTCN`, `QuantileLoss`
- `forecaster/backends/cpu_backend.py` — `CPUForecaster` (register_backend, train, predict, warmup, health_check)
- `forecaster/backends/gpu_backend.py` — `CUDAForecaster` (AMP, CUDA streams, pinned memory, torch.compile, pynvml)
- `forecaster/service.py` — FastAPI with /predict, /predict/batch, /health, /metrics + Prometheus instruments
- `fuelsense/core/features.py` — `build_lookback_matrix()`, `extract_training_data()`, `build_drift_data()`
- `ml_pipeline/training.py` — `ForecastTrainer` (global + per-facility models, MLflow integration)
- `ml_pipeline/drift.py` — `DriftMonitor` (RMSE ratio, diagnostics, facility sweep)
- `fuelsense/core/tasks.py` — `run_batch_forecasts`, `check_all_drift`, `retrain_model`
- `forecaster/tests/` — test_model.py, test_cpu_backend.py, test_gpu_backend.py, test_service.py
- `ml_pipeline/tests/` — test_training.py, test_drift.py

**Gate**
- TCN forward pass: `[B, 90, 6] → [B, 14, 3]` for B in {1, 16, 50, 256}
- CPU training loop converges on synthetic data (val loss monotonically decreasing over first 20 epochs)
- GPU training loop converges with AMP (val loss comparable to CPU), pinned memory transfers verified, torch.compile succeeds or gracefully falls back
- MAPE < 8% on held-out synthetic test set (both global and per-facility models)
- Per-facility model outperforms global model on high-variance facilities
- MLflow run logged with all params, per-epoch metrics, test metrics, model artifact, tags
- Registered model names: `demand-forecaster-global` and `demand-forecaster-{facility_id}` present in MLflow
- Drift ratio > 1.5 correctly triggers retrain when distribution artificially shifted
- Drift ratio ≈ 1.0 on stable data, no false retrain triggers
- Auto-promotion: new model replaces old when test_rmse improves, old model deactivated
- `/predict` and `/predict/batch` return valid responses with correct shapes
- `/predict` latency: ~2ms CPU, ~0.3ms GPU (single facility)
- `/predict/batch` latency: ~15ms CPU, ~0.5ms GPU (50 facilities)
- Tests > 95% coverage on forecaster + ml_pipeline
- CI `test-ml-services` green, `gpu-tests.yml` green on GPU runner

---

### Phase 4 — Anomaly Detector: Two-Stage Detection + Type Classification

*Depends on: Phases 1–2. Parallel with Phases 3 and 5.*

**Steps**

1. **Anomaly detector core** — Implement `anomaly/detector.py` exactly as Architecture §6:
   - `AnomalyType` enum: LEAK, THEFT, EQUIPMENT_DEGRADATION, DEMAND_SHIFT, SENSOR_FAULT, UNKNOWN
   - `AnomalyDetector` class with `Z_THRESHOLD=3.0`, `N_ESTIMATORS=200`, `CONTAMINATION=0.05`
   - Feature names list: z_score, z_score_rolling_3d, consumption_delta_pct, temperature_residual, day_of_week, hours_since_delivery, inventory_level_pct
   - `load(forest_path, classifier_path)`: load Isolation Forest and type classifier via joblib
   - `detect(actual, predicted, rolling_std, features)`:
     - Stage 1: compute z-score, if |z| < 3.0 → not anomaly (stage=1)
     - Stage 2: build 7-feature vector → Isolation Forest `decision_function` + `predict` → if IF says normal → report anomaly with low confidence (0.4, type=UNKNOWN, stage=1)
     - Stage 2 confirmed: type classifier `predict` + `predict_proba` → return anomaly_type, confidence (max class probability), if_score, stage=2

2. **Anomaly training pipeline** — Train Isolation Forest and type classifier on labeled synthetic anomaly data. IsolationForest on all anomaly feature vectors. RandomForestClassifier (or equivalent) for type classification on labeled samples. Save via joblib. Log to MLflow under `anomaly-detector` experiment.

3. **FastAPI service** — Implement `anomaly/service.py`:
   - Lifespan: load models from disk (forest + classifier paths from env vars)
   - `POST /detect` — single observation detection per Architecture §6 contract
   - `GET /health` — status + model_loaded check
   - `GET /metrics` — Prometheus metrics (detection latency histogram, anomaly count counter by type)

4. **Anomaly feature engineering** — Add `build_anomaly_features(facility_id)` to `fuelsense/core/features.py`:
   - Query yesterday's `InventoryLog` actual consumption
   - Query latest `Forecast` for predicted consumption
   - Compute z_score from rolling_std of residuals
   - Compute z_score_rolling_3d, consumption_delta_pct, temperature_residual, day_of_week, hours_since_delivery (from latest DeliveryItem), inventory_level_pct
   - Return None if insufficient data

5. **Celery task** — Wire `run_batch_anomaly_detection(facility_ids)` in `fuelsense/core/tasks.py`: for each facility → build features → httpx POST to `http://anomaly-detector:8002/detect` → create `AnomalyAlert` record if `is_anomaly=True`

6. **Tests** — Comprehensive:
   - `anomaly/tests/test_detector.py`: Stage 1 thresholding (z < 3 → no anomaly, z > 3 → proceeds to stage 2), Isolation Forest classification, type classifier output shape, feature vector construction, all 6 anomaly types returned correctly, UNKNOWN type for statistical-only flags
   - `anomaly/tests/test_service.py`: contract tests for /detect (valid→response, malformed→422), /health, /metrics
   - **Targets: precision > 0.80, recall > 0.70 on labeled synthetic anomalies. All 5 anomaly types correctly classified. Coverage > 95%.**

**Relevant files**
- `anomaly/detector.py` — `AnomalyType`, `AnomalyDetector` (two-stage: z-score → IsolationForest → type classifier)
- `anomaly/service.py` — FastAPI with /detect, /health, /metrics
- `fuelsense/core/features.py` — `build_anomaly_features()`
- `fuelsense/core/tasks.py` — `run_batch_anomaly_detection`
- `anomaly/tests/test_detector.py`, `anomaly/tests/test_service.py`

**Gate**
- Stage 1: |z| < 3.0 → `is_anomaly=False`, |z| > 3.0 → proceeds to stage 2
- Stage 2: IF confirms → type classification with confidence; IF rejects → UNKNOWN with 0.4 confidence
- Precision > 0.80, recall > 0.70 on labeled synthetic anomaly dataset
- All 5 anomaly types (LEAK, THEFT, EQUIPMENT_DEGRADATION, DEMAND_SHIFT, SENSOR_FAULT) classified correctly on representative test cases
- `AnomalyAlert` records created in DB with correct type, score, actual/predicted values
- `/detect` latency < 0.5ms per observation
- Tests > 95% coverage
- CI green for anomaly tests

---

### Phase 5 — Route Optimizer: Dual CPU/GPU + Planning Orchestration + Daily Tick

*Depends on: Phases 1–2. Integrates outputs from Phases 3–4. Completes the full system pipeline.*

**Steps**

1. **OR-Tools CPU backend** — Implement `optimizer/backends/cpu_backend.py` exactly as Architecture §5.1:
   - `@register_backend("route_optimizer", DeviceType.CPU)` class `ORToolsOptimizer`
   - `__init__`: time_limit_ms from `FUELSENSE_OPTIMIZER_TIME_LIMIT_MS` (default 10000), logging
   - `warmup()`: no-op (OR-Tools has no JIT)
   - `health_check()`: device=cpu, solver=or-tools, time_limit_ms
   - `solve(distance_matrix, demands, vehicle_capacities, vehicle_costs_per_km, time_windows, service_times, max_route_duration)`:
     - Build `RoutingIndexManager(n_nodes, n_vehicles, 0)` and `RoutingModel`
     - Register `distance_callback` (distance matrix lookup, scaled to int), `demand_callback` (per-stop demand), `time_callback` (travel time + service time)
     - Add capacity dimension with per-vehicle limits
     - Add time dimension with 30-min slack, per-stop time windows
     - Search: `PATH_CHEAPEST_ARC` first solution, `GUIDED_LOCAL_SEARCH` metaheuristic
     - Extract routes: per-vehicle stops with facility_index, demand, arrival_min, sequence
     - Return: status (optimal/feasible/infeasible), routes, total_distance, total_cost, vehicles_used, solver_time_ms
   - `_compute_baseline(distance_matrix, demands, vehicle_capacities, vehicle_costs_per_km)`: naive round-trip per facility from depot, return total cost
   - Response includes `baseline_cost` and `cost_reduction_pct`

2. **GPU backend** — Implement `optimizer/backends/gpu_backend.py` exactly as Architecture §5.2:
   - `@register_backend("route_optimizer", DeviceType.CUDA)` class `CUDARouteOptimizer`
   - `N_PARALLEL=64`, `MAX_ITERATIONS=1000`
   - `__init__`: `torch.device("cuda:0")`, time_limit_ms from env
   - `warmup()`: solve tiny 6-node problem to JIT compile kernels, synchronize
   - `health_check()`: device=cuda, gpu_name, gpu_memory_free_gb, n_parallel, time_limit_ms
   - `_nearest_neighbor_init(dist_matrix, n_solutions)`: generate N diverse starting solutions via randomized nearest-neighbor with Gumbel noise on distances
   - `_evaluate_2opt_batch(dist_matrix, routes)`: GPU kernel evaluating all O(n²) 2-opt swap deltas for N parallel routes. Tensor operations: gather current/new edge costs, compute deltas `[N, n, n]`, mask upper triangle (i < j only)
   - `solve(...)`: move distance matrix to GPU → init N solutions → iterative improvement loop (evaluate all swaps → find best improving move per solution → apply 2-opt reversal → update global best → check convergence/time limit) → synchronize → convert to route format → compute baseline → return results with iterations count and parallel_instances count

3. **Optimizer FastAPI service** — Implement `optimizer/service.py`:
   - Lifespan: `resolve_device()` → `get_backend("route_optimizer", device)` → warmup
   - `POST /optimize` — accept request per PRD §4.2 optimizer contract (depot, vehicles, stops with time windows, objective weights), return optimized routes with baseline comparison
   - `GET /health` — backend health_check
   - `GET /metrics` — Prometheus: `solver_time_ms` (histogram), `cost_reduction_pct` (histogram), `vehicles_used` (histogram), `planning_cycles_total` (counter)

4. **Route builder** — Implement `fuelsense/core/routing.py`:
   - `build_optimizer_request(depot, facilities)`: compute Haversine distance matrix from lat/lng, extract demands from latest forecasts (dynamic reorder delta), map facility delivery_window_start/end to minutes, set service_time_min=30, construct full optimizer request payload with depot, vehicles, stops, objective_weights

5. **Planning cycle task** — Wire `run_planning_cycle()` in `fuelsense/core/tasks.py`:
   - Query facilities where `current_inventory <= dynamic_reorder_point` and `is_active=True`
   - Group by depot via `DepotFacilityAssignment`
   - Per depot: `build_optimizer_request()` → httpx POST to `http://route-optimizer:8003/optimize` → create `PlanningCycle` audit record → create `Delivery` + `DeliveryItem` records for each route and stop

6. **Daily tick orchestration** — Implement `daily_tick()` exactly as Architecture §8:
   - Celery chain: `chord([ingest_facility_data.si(fid) for fid in facility_ids], ingestion_complete.si())` → `run_batch_forecasts.si(facility_ids)` → `run_batch_anomaly_detection.si(facility_ids)` → `check_all_drift.si()` → `run_planning_cycle.si()`
   - All 14 task implementations complete: daily_tick, ingest_facility_data, ingestion_complete, run_batch_forecasts, run_batch_anomaly_detection, check_all_drift, retrain_model, run_planning_cycle

7. **Emergency delivery** — Implement `trigger_emergency_delivery(facility_id)` task: single-facility route optimization bypassing normal planning cycle, dispatched on planning queue, creates PlanningCycle with trigger_type=EMERGENCY

8. **Benchmarks** — Implement `optimizer/benchmark.py` and `forecaster/benchmark.py` for `make bench-forecaster` and `make bench-optimizer` targets. Measure and record CPU vs GPU performance per Architecture §13 benchmark table.

9. **Tests** — Comprehensive:
   - `optimizer/tests/test_cpu_backend.py`: known small instances (5-10 stops) with hand-verified optimal, capacity constraints respected, time windows respected
   - `optimizer/tests/test_gpu_backend.py`: parallel init diversity, 2-opt improvement over initial solution, convergence detection, GPU tensor shapes (skip if no GPU, run via gpu-tests.yml)
   - `optimizer/tests/test_service.py`: contract tests for /optimize (valid→routes, infeasible→status, malformed→422)
   - Property tests (hypothesis): for any valid input — all demands satisfied, no vehicle exceeds capacity, all time windows respected or penalty recorded, no duplicate stops, depot is start/end
   - Benchmark tests: 50-stop < 10s (CPU), cost_reduction > 0%
   - Integration: full daily_tick → verify Forecast, AnomalyAlert, Delivery records created, database state consistent at each step
   - **Coverage > 95%.**

**Relevant files**
- `optimizer/backends/cpu_backend.py` — `ORToolsOptimizer` (solve with all constraints, _compute_baseline)
- `optimizer/backends/gpu_backend.py` — `CUDARouteOptimizer` (parallel 2-opt, _nearest_neighbor_init, _evaluate_2opt_batch, N_PARALLEL=64)
- `optimizer/service.py` — FastAPI with /optimize, /health, /metrics + Prometheus instruments
- `fuelsense/core/routing.py` — `build_optimizer_request()` (Haversine distance, payload construction)
- `fuelsense/core/tasks.py` — `run_planning_cycle`, `daily_tick`, `trigger_emergency_delivery`, all remaining tasks
- `forecaster/benchmark.py`, `optimizer/benchmark.py` — CPU vs GPU benchmarking scripts
- `optimizer/tests/` — test_cpu_backend.py, test_gpu_backend.py, test_service.py

**Gate**
- OR-Tools: valid routes for all test cases, capacity respected, time windows honored, solver returns optimal or feasible
- GPU optimizer: parallel init produces 64 diverse solutions, 2-opt improves over initial cost, tensors on correct device
- Cost reduction > 20% vs naive baseline averaged across 30 simulated planning cycles
- 50-stop CPU < 10s, 100-stop CPU < 30s
- GPU crossover verified: GPU faster than CPU at 100+ stops per Architecture §13 table
- Full daily_tick chain: ingest → forecast → anomaly → drift → planning completes end-to-end for 50 facilities in < 60s (CPU), < 15s (GPU)
- PlanningCycle audit records: correct facilities_in_queue, deliveries_created, total_distance, total_cost, solver_time_ms, baseline_cost, cost_reduction_pct
- Emergency delivery: single-facility route created with trigger_type=EMERGENCY
- Benchmark results recorded and match Architecture §13 targets (within 2x tolerance)
- All property tests pass (demands satisfied, capacity respected, no duplicates)
- Tests > 95% coverage
- CI green for all optimizer + integration tests

---

### Phase 6 — Containerization: All Dockerfiles, Compose Stacks & Runtime Profiles

*Depends on: Phases 3–5 (all services implemented). Blocking for Phase 7.*

**Steps**

1. **6 Dockerfiles** — Implement exactly as Architecture §9.1:
   - `Dockerfile.django` — Multi-stage: `python:3.12-slim` base with libpq-dev+gcc, `pip install base.txt`, app stage with COPY + `collectstatic --noinput`, EXPOSE 8000, CMD gunicorn with 4 workers, 120s timeout
   - `Dockerfile.forecaster` — `python:3.12-slim`, install forecaster-cpu.txt, COPY forecaster/ + fuelsense_common/, EXPOSE 8001, CMD uvicorn
   - `Dockerfile.forecaster-gpu` — `nvidia/cuda:12.4-runtime-ubuntu22.04`, install python3+pip, install forecaster-gpu.txt, COPY forecaster/ + fuelsense_common/, EXPOSE 8001, CMD uvicorn
   - `Dockerfile.anomaly` — `python:3.12-slim`, install anomaly.txt, COPY anomaly/ + fuelsense_common/, EXPOSE 8002, CMD uvicorn
   - `Dockerfile.optimizer` — `python:3.12-slim`, install optimizer-cpu.txt, COPY optimizer/ + fuelsense_common/, EXPOSE 8003, CMD uvicorn
   - `Dockerfile.optimizer-gpu` — `nvidia/cuda:12.4-runtime-ubuntu22.04`, install optimizer-gpu.txt, COPY optimizer/ + fuelsense_common/, EXPOSE 8003, CMD uvicorn

2. **docker-compose.yml** — All 13 services per Architecture §9.2:
   - `postgres` (postgres:16, volume pgdata, port 5432)
   - `redis` (redis:7-alpine, port 6379)
   - `django` (Dockerfile.django, port 8000, depends on postgres+redis, dev volume mount, runserver for dev)
   - `celery-default` (Dockerfile.django, `celery -A fuelsense worker -Q default -c 4`, depends on postgres+redis)
   - `celery-training` (Dockerfile.django, `celery -A fuelsense worker -Q training -c 2`)
   - `celery-planning` (Dockerfile.django, `celery -A fuelsense worker -Q planning -c 1`)
   - `celery-beat` (Dockerfile.django, `celery -A fuelsense beat`)
   - `forecaster` (Dockerfile.forecaster, port 8001, FUELSENSE_DEVICE=cpu, volume for models)
   - `anomaly` (Dockerfile.anomaly, port 8002, volume for models)
   - `optimizer` (Dockerfile.optimizer, port 8003)
   - `mlflow` (ghcr.io/mlflow/mlflow:v2.11.0, port 5000, PG backend store, volume mlartifacts)
   - `prometheus` (prom/prometheus:v2.50.0, port 9090, volume prometheus.yml)
   - `grafana` (grafana/grafana:10.3.0, port 3000, volume dashboards)
   - Volumes: pgdata, mlartifacts

3. **docker-compose.gpu.yml** — GPU override per Architecture §9.3:
   - `forecaster`: Dockerfile.forecaster-gpu, FUELSENSE_DEVICE=cuda, NVIDIA runtime reservation (driver: nvidia, count: 1, capabilities: [gpu])
   - `optimizer`: Dockerfile.optimizer-gpu, FUELSENSE_OPTIMIZER_BACKEND=cuda, NVIDIA runtime reservation

4. **Health probes** — All services expose health endpoints: Django `/healthz` + `/readyz`, ML services `/health`. Add startup delays and health check configs to compose (interval, timeout, retries, start_period for model loading).

5. **Runtime tuning** — Configure all `FUELSENSE_*` env vars in `.env`: device, CUDA memory fraction, TF32, cudnn.benchmark, batch sizes (CPU: forecast 16, training 32; GPU: forecast 256, training 512), optimizer backend, optimizer time limit. Verify CPU thread binding and GPU memory limits under compose.

6. **No secrets in images** — Verify with `docker history` that no .env or credential data is baked into any image. All secrets via env_file or Kubernetes Secrets.

**Relevant files**
- `Dockerfile.django`, `Dockerfile.forecaster`, `Dockerfile.forecaster-gpu`, `Dockerfile.anomaly`, `Dockerfile.optimizer`, `Dockerfile.optimizer-gpu`
- `docker-compose.yml` — 13-service CPU stack
- `docker-compose.gpu.yml` — GPU override for forecaster + optimizer

**Gate**
- `docker compose up -d --build` starts all 13 services without errors
- `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build` starts GPU stack (on GPU-equipped host)
- All health endpoints return 200 within 90 seconds of startup
- `make seed` → `make test` passes inside containers
- Image sizes: Django < 500MB, ML CPU services < 1.5GB, ML GPU services < 4GB
- `docker history` shows no secrets baked into any image
- GPU forecaster runs inference on CUDA device (verified via /health response showing device=cuda)
- GPU optimizer runs solve on CUDA device

---

### Phase 7 — Kubernetes, Helm Chart & Service Mesh (mTLS)

*Depends on: Phase 6. Blocking for Phase 8.*

**Steps**

1. **Helm chart scaffolding** — Create `charts/fuelsense/` with `Chart.yaml`, `values.yaml` (CPU default per Architecture §10.2), `values-gpu.yaml` (GPU override per Architecture §10.3), `_helpers.tpl` (template macros).

2. **All 23 Helm templates** — per Architecture §10.1:
   - `namespace.yaml`, `configmap.yaml` (all non-secret config), `secrets.yaml` (DB password, Django secret key, API tokens)
   - **Django**: `django/deployment.yaml` (2 replicas, 250m CPU/512Mi req, 500m/1Gi limits, env_from configmap+secrets, probes /healthz liveness + /readyz readiness), `django/service.yaml` (ClusterIP:8000), `django/ingress.yaml` (fuelsense.local → django:8000, Nginx class), `django/hpa.yaml`
   - **Celery**: `celery/worker-default.yaml` (2 replicas, -Q default -c 4, 250m/512Mi), `celery/worker-training.yaml` (1 replica, -Q training -c 2, 1CPU/2Gi req 2CPU/4Gi limit), `celery/worker-planning.yaml` (1 replica, -Q planning -c 1, 500m/1Gi), `celery/beat.yaml` (1 replica)
   - **ML services**: `ml-services/forecaster.yaml` (2 replicas, 250m/512Mi, FUELSENSE_DEVICE=cpu, liveness+readiness on /health), `ml-services/forecaster-hpa.yaml` (min=1, max=4, targetCPU=70%), `ml-services/anomaly.yaml` (1 replica, 100m/256Mi), `ml-services/optimizer.yaml` (1 replica, 500m/1Gi req 2CPU/2Gi limit)
   - **Data**: `data/postgresql.yaml` (StatefulSet, 10Gi PVC, postgres:16), `data/redis.yaml` (Deployment, redis:7), `data/mlflow.yaml` (Deployment, port 5000, 20Gi PVC)
   - **Monitoring**: `monitoring/prometheus.yaml` (Deployment, 9090, retention 15d, 10Gi PVC, scrape config), `monitoring/grafana.yaml` (Deployment, 3000, dashboard provisioning), `monitoring/servicemonitor.yaml` (ServiceMonitor for Prometheus Operator scraping of all services)
   - **Tests**: `tests/smoke-test.yaml` (helm test pod: curl all health endpoints)

3. **values-gpu.yaml** — Full GPU overlay per Architecture §10.3:
   - `compute.device: cuda`
   - `forecaster`: GPU image, resources with `nvidia.com/gpu: 1` requests+limits, FUELSENSE_DEVICE=cuda, FUELSENSE_CUDA_MEMORY_FRACTION=0.5, tolerations for nvidia.com/gpu, nodeSelector accelerator=nvidia
   - `optimizer`: GPU image, resources with `nvidia.com/gpu: 1`, tolerations
   - `celery.training`: elevated resources with `nvidia.com/gpu: 1`, tolerations

4. **Resource policies** — All resource requests and limits per Architecture §10.2 values.yaml. Forecaster HPA: minReplicas=1, maxReplicas=4, targetCPU=70%.

5. **Service mesh (mTLS)** — Install and configure Istio or Linkerd in the cluster per Architecture §14:
   - Enable automatic sidecar injection for `fuelsense` namespace
   - mTLS strict mode: all pod-to-pod communication encrypted (Django → ML services, Celery → ML services, Django → PostgreSQL, all → Redis)
   - Certificate rotation managed by mesh control plane
   - Verify mTLS active via mesh dashboard or `istioctl` / `linkerd check`

6. **Kubernetes NetworkPolicies** — Per Architecture §14:
   - Default deny-all ingress for `fuelsense` namespace
   - Allow: Django pods → ML services (forecaster:8001, anomaly:8002, optimizer:8003)
   - Allow: Celery worker pods → ML services
   - Allow: Django + Celery → PostgreSQL:5432
   - Allow: Django + Celery → Redis:6379
   - Allow: MLflow → PostgreSQL:5432
   - Allow: Ingress controller → Django:8000
   - Allow: Prometheus → all services `/metrics` endpoints
   - ML services not reachable from outside cluster (ClusterIP only, no Ingress)

7. **Kind local cluster** — Create `infra/kind-config.yaml`. `make k8s-local` deploys full CPU stack. `make k8s-local-gpu` deploys with GPU values. Run `helm test fuelsense` (smoke test pod curls all health endpoints).

**Relevant files**
- `charts/fuelsense/Chart.yaml`, `values.yaml`, `values-gpu.yaml`
- `charts/fuelsense/templates/` — all 23 template files across django/, celery/, ml-services/, data/, monitoring/, tests/
- `charts/fuelsense/templates/_helpers.tpl`
- `infra/kind-config.yaml`
- NetworkPolicy manifests (within templates or separate)
- Service mesh configuration (Istio/Linkerd install + namespace annotation)

**Gate**
- `helm install fuelsense charts/fuelsense/` brings up entire stack from zero on Kind
- All pods reach Ready state within 120 seconds
- `helm test fuelsense` passes (smoke test hits all health endpoints)
- `helm upgrade` performs rolling update with zero downtime
- `helm template` renders valid YAML (validate with `kubeconform`)
- `values-gpu.yaml`: forecaster/optimizer/training-worker pods request nvidia.com/gpu, tolerations and nodeSelector applied
- mTLS active: inter-service traffic encrypted, verified via service mesh dashboard or CLI
- NetworkPolicies enforced: ML services not accessible from outside cluster, only Django+Celery can reach them
- PostgreSQL only reachable from Django, Celery, MLflow pods
- Prometheus can scrape /metrics from all services
- `make k8s-local` and `make k8s-down` succeed cleanly

---

### Phase 8 — Full CI/CD Pipeline with Progressive Gates

*Depends on: Phase 7. Parallel with Phase 9.*

**Steps**

1. **Complete `.github/workflows/ci.yml`** — All 5 jobs per Architecture §12:
   - **Job 1: test-django** — ubuntu-latest, services: postgres:16 (test DB) + redis:7-alpine. Steps: checkout → setup Python 3.12 → `pip install base.txt + dev.txt` → `pytest fuelsense/ --cov --cov-report=xml` (env: DATABASE_URL, REDIS_URL). **Fail if coverage < 95%.**
   - **Job 2: test-ml-services** — ubuntu-latest. Steps: checkout → setup Python 3.12 → install forecaster-cpu.txt + dev.txt → `pytest forecaster/tests/` → install anomaly.txt → `pytest anomaly/tests/` → install optimizer-cpu.txt → `pytest optimizer/tests/` → install dev.txt → `pytest ml_pipeline/tests/`. **Fail if coverage < 95%.**
   - **Job 3: build-images** (needs: test-django + test-ml-services, only on main) — Matrix build 4 images: docker/build-push-action with context=., push to ghcr.io, tag=$GITHUB_SHA. Matrix: (Dockerfile.django→django, Dockerfile.forecaster→forecaster, Dockerfile.anomaly→anomaly, Dockerfile.optimizer→optimizer). Login via GITHUB_TOKEN.
   - **Job 4: deploy-staging** (needs: build-images, only on main) — `helm upgrade --install fuelsense charts/fuelsense/ --set global.imageTag=$GITHUB_SHA --wait --timeout 300s`
   - **Job 5: smoke-test** (needs: deploy-staging) — wait 30s → curl -f all 5 health endpoints (Django /healthz, API /facilities/, forecaster:8001/health, anomaly:8002/health, optimizer:8003/health)

2. **GPU test workflow** — Complete `.github/workflows/gpu-tests.yml`:
   - Trigger: `workflow_dispatch` (manual)
   - Runner: self-hosted GPU runner
   - Steps: checkout → install forecaster-gpu.txt + optimizer-gpu.txt + dev.txt → `pytest forecaster/tests/test_gpu_backend.py -v` → `pytest optimizer/tests/test_gpu_backend.py -v` → benchmark scripts (CPU vs GPU comparison)
   - Report benchmark results as job summary

3. **Branch protection** — Configure main branch: require test-django + test-ml-services jobs passing, require 1 approval, no force push, no bypass.

4. **Coverage enforcement** — Configure pytest-cov with `--cov-fail-under=95` in CI for all test jobs. Coverage reports uploaded as artifacts.

5. **Image scanning** — Add container image vulnerability scanning step in build-images job (e.g., trivy or grype). Fail on critical/high severity findings.

**Relevant files**
- `.github/workflows/ci.yml` — 5-job progressive pipeline (test → build → deploy → smoke)
- `.github/workflows/gpu-tests.yml` — manual GPU test + benchmark workflow

**Gate**
- CI pipeline passes end-to-end on push to main (all 5 jobs green)
- Coverage ≥ 95% enforced on both test-django and test-ml-services jobs
- gpu-tests.yml runs successfully on GPU runner (manual trigger)
- Branch protection enforces passing CI before merge
- Docker images pushed to ghcr.io with 