# ADR-0001: FuelSense Architecture Decisions

**Status:** Accepted
**Date:** 2026-03-13
**Authors:** Mehdi Skouri

---

## Summary

This document records the key architectural decisions made in FuelSense, an energy logistics optimization platform. Each decision is documented with its context, the choice made, consequences, rejected alternatives, and pointers to implementation evidence. A summary decision matrix is provided at the end.

---

## Decision Matrix

| # | Decision | Primary Driver | Risk | Mitigated By |
|---|---|---|---|---|
| 1 | ML microservice split | Independent scaling and dependency isolation | HTTP latency, operational complexity | Health checks, fallback paths, HPA |
| 2 | Three specialized Celery queues | Workload isolation across latency classes | Four deployment units required | Helm chart templates, independent scaling |
| 3 | Graceful degradation for remote services | Resilience in unreliable environments | Suboptimal results during fallback | Monitoring, strict-mode override flag |
| 4 | Conditional ML imports with fallback resolvers | Lean base container image | Late import errors at task execution | Suppressed module-level + resolver functions, separate requirements |
| 5 | Pluggable compute backend abstraction | Hardware portability (CPU ↔ CUDA) | GPU code paths not in standard coverage | Backend parity tests, hardware-specific CI |
| 6 | Chord-based daily pipeline orchestration | Strict stage ordering with parallel ingest | Ingestion synchronization bottleneck | Parallel per-facility ingest, batch sizing |
| 7 | Idempotent delivery materialization | Safe task retries without duplicate deliveries | SHA256 hashing overhead per route | Indexed unique constraint, `get_or_create` |
| 8 | Dynamic reorder points with cascading fallback | Demand adaptation when forecaster available or not | Forecast service dependency | Three-tier fallback chain |

---

## ADR-1: ML Microservice Split

### Context

FuelSense combines a Django-based API and orchestration layer with three distinct ML workloads: demand forecasting (PyTorch), anomaly detection (scikit-learn), and route optimization (Google OR-Tools). These workloads have fundamentally different dependency trees (PyTorch alone adds ~2 GiB to an image), resource profiles (GPU for inference vs. CPU for routing), and scaling characteristics (forecasting is bursty at batch time; the API serves interactive traffic).

### Decision

Separate each ML capability into its own containerized FastAPI microservice, each with its own Dockerfile and requirements file:

- **Forecaster** (`Dockerfile.forecaster`, port 8001) — PyTorch TCN inference and training
- **Anomaly Detector** (`Dockerfile.anomaly`, port 8002) — scikit-learn Isolation Forest
- **Route Optimizer** (`Dockerfile.optimizer`, port 8003) — OR-Tools vehicle routing

Django orchestrates these services via HTTP calls from Celery tasks. Each service exposes a `/health` endpoint for container health checks and a `/metrics` endpoint for Prometheus scraping.

### Consequences

**Benefits:**
- Independent scaling — the forecaster has HPA (1 → 4 replicas at 70% CPU) while the anomaly detector runs as a single replica
- Independent deployment — updating the forecaster model doesn't require redeploying the API
- Isolated failure domains — a forecaster OOM doesn't crash the optimizer
- Clean dependency boundaries — no torch in the base Django image

**Costs:**
- HTTP latency overhead per service call (mitigated by batch endpoints)
- Operational complexity of managing 4+ container images
- Need for health checks and fallback logic at every service boundary

### Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| **Monolith** (all ML in the Django process) | Dependency conflicts (torch + ortools in one image), inability to scale ML independently, single-process memory pressure |
| **Full event-driven** (Kafka/NATS between services) | Overengineered for batch-oriented ML workloads that run on a daily schedule; Celery provides sufficient orchestration |
| **Serverless functions** (AWS Lambda / Cloud Run) | Cold start latency incompatible with batch inference timing requirements; torch model loading takes seconds |

### Evidence

- `docker-compose.yml`: 13 services with separate Dockerfiles per ML service
- `Dockerfile.forecaster`, `Dockerfile.anomaly`, `Dockerfile.optimizer`: independent build contexts
- `requirements/forecaster-cpu.txt`, `requirements/anomaly.txt`, `requirements/optimizer-cpu.txt`: isolated dependency trees
- `charts/fuelsense/values.yaml`: HPA enabled on forecaster with `minReplicas: 1`, `maxReplicas: 4`
- `fuelsense/core/tasks.py`: HTTP POST calls to each service with independent fallback handling

---

## ADR-2: Three Specialized Celery Queues

### Context

FuelSense's background workloads span three fundamentally different latency and resource classes:

1. **Low-latency operational tasks**: data ingestion, anomaly detection (seconds)
2. **I/O-bound service tasks**: forecasting batch calls, planning optimization (seconds to minutes, mostly waiting on HTTP)
3. **CPU/GPU-intensive training tasks**: model retraining, drift analysis (minutes to hours)

Running all tasks on a single queue risks training jobs starving time-sensitive ingestion or planning operations.

### Decision

Three named Celery queues, each served by an independent worker pool:

| Queue | Worker | Tasks | Resource Profile |
|---|---|---|---|
| `default` | `celery-default` (concurrency 4) | `ingest_facility_data`, `daily_tick`, `run_batch_anomaly_detection` | Low CPU, low memory |
| `planning` | `celery-planning` (concurrency 4) | `run_batch_forecasts`, `run_planning_cycle`, `run_emergency_planning_cycle` | I/O-bound, moderate memory |
| `training` | `celery-training` (concurrency 2) | `check_all_drift`, `retrain_model` | High CPU/GPU, high memory |

A fourth deployment unit, `celery-beat`, handles schedule coordination.

### Consequences

**Benefits:**
- Training cannot starve operational pipelines — queues are physically isolated
- Independent scaling — Kubernetes can allocate GPU nodes only to training workers
- Resource limits tuned per workload class (training: 1–2 CPU / 2–4 GiB; default: 250–500m CPU / 512 MiB–1 GiB)

**Costs:**
- Four Celery deployment units in Docker Compose and Helm (3 workers + 1 beat)
- Task routing must be explicitly declared in task decorators or configuration
- Monitoring spans multiple worker processes

### Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| **Single queue with priority** | Celery priority support is broker-dependent and doesn't provide resource isolation |
| **Separate Celery apps per service** | Increases configuration complexity without proportional benefit; single app with queue routing is simpler |

### Evidence

- `fuelsense/celery.py`: `task_routes` mapping tasks to queues, `task_default_queue = "default"`
- `fuelsense/settings/base.py`: `CELERY_BEAT_SCHEDULE` with explicit queue assignments per task
- `docker-compose.yml`: `celery-default`, `celery-training`, `celery-planning`, `celery-beat` services
- `charts/fuelsense/values.yaml`: differentiated resource limits per celery worker type

---

## ADR-3: Graceful Degradation for Remote Services

### Context

FuelSense operates in energy logistics environments where network reliability between the Django orchestrator and ML microservices cannot be guaranteed. A hard dependency on any single ML service would halt the entire daily pipeline — including ingestion and planning — even if only one service is temporarily unavailable.

### Decision

Every remote service call implements a built-in fallback path, controllable via environment variables:

| Service | Fallback Behavior | Control Variable |
|---|---|---|
| Forecaster | Return zeroed predictions; use heuristic reorder point (7-day avg × lead_time × 1.1) | `FUELSENSE_ENABLE_REMOTE_FORECAST` |
| Anomaly Detector | Return `is_anomaly=False` for all facilities; skip alert creation | `FUELSENSE_ENABLE_REMOTE_ANOMALY` |
| Route Optimizer | Return infeasible response (empty routes); no deliveries created | `FUELSENSE_ENABLE_REMOTE_OPTIMIZER` |

Operators who require ML accuracy over availability can set `FUELSENSE_REQUIRE_REMOTE_SERVICES=1` to make failures halt the pipeline with an explicit error.

### Consequences

**Benefits:**
- The daily pipeline always completes — facilities still get ingested, reorder points still get updated (via heuristic), planning still runs (even if no optimized routes are produced)
- Operators choose their tolerance: degraded-but-running vs. fail-fast
- Fallback paths are independently testable

**Costs:**
- Suboptimal operations during degraded mode (zeroed forecasts, no anomaly detection, no optimized routing)
- Fallback paths require their own testing and monitoring to avoid silent failures in production
- Risk of prolonged degraded operation going unnoticed without proper alerting

### Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| **Circuit breaker pattern** | Adds complexity for batch-oriented workloads that run once daily; the pipeline doesn't make repeated rapid calls that benefit from circuit breaking |
| **Retry-only with exponential backoff** | Delays the entire downstream pipeline while waiting; a single slow service blocks planning |
| **Queue-based async with eventual consistency** | Increases complexity and makes the pipeline harder to reason about for a batch workflow |

### Evidence

- `fuelsense/core/tasks.py`: `run_batch_forecasts` — `if not enable_remote_forecast:` fallback branch; `run_batch_anomaly_detection` — HTTP POST wrapped in try/except with `is_anomaly=False` fallback; `_run_optimizer` — returns infeasible response on failure
- `.env.example`: `FUELSENSE_ENABLE_REMOTE_FORECAST`, `FUELSENSE_ENABLE_REMOTE_ANOMALY`, `FUELSENSE_ENABLE_REMOTE_OPTIMIZER`, `FUELSENSE_REQUIRE_REMOTE_SERVICES`

---

## ADR-4: Conditional ML Imports with Fallback Resolvers

### Context

The Django/Celery base image (`Dockerfile.django`) includes only `requirements/base.txt` dependencies to keep it lean (~200 MiB vs. ~2+ GiB with PyTorch). However, Celery tasks in `fuelsense/core/tasks.py` need to call ML pipeline modules (`ml_pipeline.training.ForecastTrainer`, `ml_pipeline.anomaly_training.AnomalyTrainer`, `ml_pipeline.drift.DriftMonitor`) for model retraining and drift checking.

**Incident:** In a staging deployment, a hard module-level import of `ml_pipeline.training` caused `ModuleNotFoundError: No module named 'torch'` in all celery workers, resulting in `CrashLoopBackOff` across the celery-default and celery-planning deployments — even though torch isn't needed for their tasks.

### Decision

ML pipeline imports use a two-tier conditional pattern:

1. **Module level**: `with suppress(ModuleNotFoundError)` attempts the import but defaults to `None` if the dependency is missing — the worker starts normally either way.
2. **Task execution**: Resolver functions (`_resolve_drift_monitor()`, `_resolve_forecast_training_types()`, `_resolve_anomaly_trainer()`) retry the import at call time if the module-level attempt returned `None`.

```python
# Module level — fails gracefully
DriftMonitor: Any = None
with suppress(ModuleNotFoundError):
    DriftMonitor = importlib.import_module("ml_pipeline.drift").DriftMonitor

# Resolver — retries at task execution if module-level was None
def _resolve_drift_monitor() -> type[object]:
    monitor_cls = DriftMonitor
    if monitor_cls is None:
        monitor_cls = importlib.import_module("ml_pipeline.drift").DriftMonitor
    return cast("type[object]", monitor_cls)
```

This ensures workers without ML dependencies start cleanly, and ML tasks on properly-equipped workers get the classes they need.

### Consequences

**Benefits:**
- Base Celery image stays lean — no torch, sklearn, or ortools needed
- Default and planning workers start and operate normally even without ML dependencies
- Training workers (which have the full dependency set) resolve ML classes either at module load or at task execution time
- `suppress(ModuleNotFoundError)` prevents hard crashes; resolver functions provide a second chance

**Costs:**
- Import errors for ML-dependent tasks surface at task execution rather than worker startup
- Developers must follow the two-tier pattern (suppressed module-level + resolver function) when adding new ML-dependent tasks
- Module-level variables default to `None`, so callers must go through resolver functions rather than using the module-level reference directly

### Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| **Include torch in base image** | Doubles image size for all workers; most workers never need torch |
| **Separate Celery app for training** | Increases configuration complexity; single app with queue routing and conditional imports is simpler |
| **Pure function-level imports only** | Would lose the benefit of pre-loading on workers that do have the dependencies; the two-tier approach gets the best of both worlds |

### Evidence

- `fuelsense/core/tasks.py`: Module-level `with suppress(ModuleNotFoundError)` blocks for `AnomalyTrainer`, `DriftMonitor`, `ForecastTrainer`, `TrainingDataset`; resolver functions `_resolve_drift_monitor()`, `_resolve_forecast_training_types()`, `_resolve_anomaly_trainer()`
- `requirements/base.txt`: no torch, sklearn, or ortools dependencies
- Repository operational note: torch `CrashLoopBackOff` incident in celery workers triggered the lazy import migration

---

## ADR-5: Pluggable Compute Backend Abstraction

### Context

The forecaster and optimizer need to run on both CPU-only development machines and GPU-accelerated production nodes. The business logic (forecast inference, route solving) should not change based on hardware — only the underlying compute implementation.

### Decision

A `ComputeBackend` runtime protocol with a decorator-based registry:

```python
# Protocol definition
class ComputeBackend(Protocol):
    device: DeviceType
    def warmup(self) -> None: ...
    def health_check(self) -> dict[str, object]: ...

# Registration
@register_backend("forecaster", DeviceType.CPU)
class CPUForecaster: ...

@register_backend("forecaster", DeviceType.CUDA)
class GPUForecaster: ...

# Resolution
backend = get_backend("forecaster", resolve_device())
```

`resolve_device()` checks `FUELSENSE_DEVICE` env var, falls back to auto-detection (CUDA availability via `torch.cuda.is_available()`), then defaults to CPU.

### Consequences

**Benefits:**
- Same service entrypoint works on any hardware — the Dockerfile changes, not the application code
- GPU backends added by decoration — no modification to existing code paths
- `FUELSENSE_DEVICE` env var provides deterministic override for testing and deployment
- CPU fallback is automatic if CUDA is requested but unavailable

**Costs:**
- GPU backend code excluded from standard test coverage (requires hardware)
- Backend parity must be verified separately (dedicated `test_backend_parity.py` tests)
- `importlib.import_module("torch")` in `resolve_device()` adds a level of indirection

### Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| **Compile-time selection** (separate codebases per device) | Duplicates business logic; backends differ only in compute primitives |
| **Abstract base class hierarchy** | Runtime protocol is more Pythonic, doesn't require inheritance, and works with third-party types |
| **Configuration-file-driven backend selection** | Environment variable is simpler and standard for container deployments |

### Evidence

- `fuelsense_common/compute.py`: `ComputeBackend` protocol, `DeviceType` enum, `resolve_device()` function
- `fuelsense_common/registry.py`: `@register_backend` decorator, `get_backend()` with CPU fallback
- `forecaster/backends/cpu_backend.py`, `forecaster/backends/gpu_backend.py`: registered implementations
- `optimizer/backends/cpu_backend.py`, `optimizer/backends/gpu_backend.py`: registered implementations
- `docker-compose.gpu.yml`: `FUELSENSE_DEVICE: cuda` with NVIDIA device reservation
- `charts/fuelsense/values-gpu.yaml`: `nodeSelector: accelerator: nvidia`
- `forecaster/tests/test_backend_parity.py`: parity verification between CPU and GPU outputs

---

## ADR-6: Chord-Based Daily Pipeline Orchestration

### Context

The daily ML pipeline has strict ordering constraints:

1. **Ingestion** must complete for all facilities before forecasting starts (forecasts depend on up-to-date inventory logs)
2. **Forecasting** must complete before anomaly detection (anomaly features use predicted vs. actual consumption)
3. **Anomaly detection** must complete before drift checking (drift uses recent prediction accuracy)
4. **Drift checking** must complete before planning (reorder points may change from re-forecasting)
5. **Planning** runs last (uses updated reorder points to select facilities needing delivery)

Within ingestion, individual facility data can be processed in parallel.

### Decision

Celery chord + chain pattern in `daily_tick`:

```
chord(
    [ingest_facility_data.si(fid) for fid in facility_ids],  # Parallel ingest
    ingestion_complete.si()                                    # Barrier callback
) | run_batch_forecasts.si(facility_ids)                       # Sequential chain
  | run_batch_anomaly_detection.si(facility_ids)
  | check_all_drift.si()
  | run_planning_cycle.si()
```

The chord provides a synchronization barrier after parallel ingestion. The chain provides strict sequential ordering for the remaining stages. `celery-beat` triggers this at 02:00 UTC daily.

### Consequences

**Benefits:**
- Guaranteed ordering — downstream stages always see completed upstream data
- Maximum parallelism where safe — N facilities ingested concurrently
- Native Celery primitive — no external orchestrator needed
- Each stage can be independently retried or run manually via API

**Costs:**
- Chord is a synchronization barrier — the slowest facility's ingestion gates the entire pipeline
- All-or-nothing per stage — a single facility ingestion failure can block the chord callback (mitigated by error handling in individual tasks)
- Chord results consume Redis memory proportional to facility count

### Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| **Sequential per-facility pipeline** | Too slow — each facility would wait for the previous to complete all stages |
| **Airflow / Prefect** | Additional infrastructure and operational burden for what Celery chord+chain handles natively |
| **Pub/sub event-driven** (facility-complete events trigger downstream) | Ordering guarantees harder to reason about; partial completion states are complex to manage |
| **Celery group (without chord)** | Groups don't provide a completion callback — no way to trigger the next stage |

### Evidence

- `fuelsense/core/tasks.py`: `daily_tick` function with `chord([ingest...], callback) | chain(forecast, anomaly, drift, planning)`
- `fuelsense/settings/base.py`: `CELERY_BEAT_SCHEDULE` with `daily-tick` entry at `crontab(minute=0, hour=2)` (02:00 UTC)
- `fuelsense/core/tests/test_tasks.py`: tests verifying orchestration flow

---

## ADR-7: Idempotent Delivery Materialization

### Context

Planning cycle tasks (`run_planning_cycle`, `run_emergency_planning_cycle`) create `Delivery` and `DeliveryItem` records from optimizer results. These tasks may retry on transient failures (network timeouts, database locks). Without idempotency protection, a retry would create duplicate deliveries for the same route, corrupting operational data.

### Decision

Each delivery is keyed by a SHA256 hash of its deterministic inputs:

```
idempotency_key = SHA256(cycle_id + depot_id + vehicle_id + route_json)
```

The `idempotency_key` field has a `unique` database constraint and is indexed. Delivery creation uses Django's `get_or_create` pattern — if the key already exists, the existing record is returned without modification.

### Consequences

**Benefits:**
- Tasks are safely retryable — retry produces no duplicates
- Deterministic key derivation — same inputs always produce the same key
- Database-enforced uniqueness — even concurrent retries are safe
- No distributed locking or coordination required

**Costs:**
- SHA256 computation per route (negligible in practice — sub-millisecond)
- `route_json` serialization must be stable (no floating-point ordering changes)
- Debugging requires key reverse-lookup (key is stored alongside the delivery record)

### Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| **Database-level upsert** (`ON CONFLICT DO NOTHING`) | Less portable across databases; `get_or_create` is the Django-native pattern |
| **Task-level deduplication** (Celery `task_id` as key) | Task IDs change on retry; doesn't capture the semantic identity of the delivery |
| **Distributed locks** (Redis-based) | Adds complexity and a single point of failure; database uniqueness is simpler and more reliable |

### Evidence

- `fuelsense/core/models.py`: `Delivery.idempotency_key` — `CharField(max_length=64, unique=True, db_index=True)`
- `fuelsense/core/tasks.py`: `_materialize_delivery_routes` — SHA256 key computation from `(cycle_id, depot_id, vehicle_id, route_json)`, followed by `Delivery.objects.get_or_create(idempotency_key=key, defaults={...})`

---

## ADR-8: Dynamic Reorder Points with Cascading Fallback

### Context

Static reorder thresholds (`min_safe_inventory`) don't adapt to seasonal demand volatility. A facility experiencing a winter demand spike needs a higher reorder point than the same facility in summer. However, the forecast-driven reorder point depends on the forecaster service being available — which, per ADR-3, cannot be guaranteed.

### Decision

A three-tier fallback chain for computing each facility's reorder point:

| Priority | Source | Calculation | When Used |
|---|---|---|---|
| 1 | **Forecast-based** | p90 quantile × lead_time_days × 1.1 safety margin | Forecaster available, fresh forecast exists |
| 2 | **Preserved prior** | Previous `dynamic_reorder_point` value | Forecaster unavailable but a prior forecast-based value exists and is considered reliable |
| 3 | **Local heuristic** | 7-day average consumption × lead_time_days × 1.1 | No forecast available and no prior dynamic value |

The result is stored in `Facility.dynamic_reorder_point`. The `Facility.effective_reorder_point` property transparently returns `dynamic_reorder_point` if positive, otherwise falls back to `min_safe_inventory`.

### Consequences

**Benefits:**
- Reorder points adapt automatically to demand shifts when the forecaster is operational
- Graceful degradation — the system always has a reorder threshold, even without ML
- Transparent to consumers — `effective_reorder_point` hides the fallback logic
- The safety margin (1.1×) provides buffer against forecast under-prediction

**Costs:**
- Stale `dynamic_reorder_point` values persist between forecast outages — could be days old
- The heuristic fallback (tier 3) uses a simple average that doesn't account for trends or seasonality
- Three code paths require testing and monitoring to ensure correctness

### Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| **Static thresholds only** | Doesn't adapt to demand volatility; leads to either excess inventory or stockouts |
| **Forecast-only (no fallback)** | Creates hard dependency on forecaster service; violates ADR-3 principles |
| **Exponential smoothing fallback** | More complex than necessary for a fallback path that should rarely be needed |

### Evidence

- `fuelsense/core/tasks.py`: `run_batch_forecasts` — three-tier reorder resolution with priority cascade
- `fuelsense/core/models.py`: `Facility.dynamic_reorder_point` (nullable FloatField), `Facility.effective_reorder_point` (property returning dynamic if > 0, else `min_safe_inventory`)
- `fuelsense/core/tests/test_reorder_semantics.py`: tests verifying fallback chain behavior

---

## Validation

All decisions in this document are traceable to the implementation as of the current codebase state. Each decision cites specific files and functions that implement the described behavior. The decision matrix at the top provides a quick-reference summary of drivers, risks, and mitigations.

These decisions should be revisited if:
- Facility counts grow beyond the chord barrier's scalability (ADR-6) — consider streaming ingestion
- GPU workloads become the primary bottleneck (ADR-5) — consider dedicated GPU scheduling infrastructure
- Service-to-service communication patterns shift from batch to real-time (ADR-1) — consider gRPC or event streaming
- Regulatory requirements mandate hard service dependencies (ADR-3) — switch to `FUELSENSE_REQUIRE_REMOTE_SERVICES=1` by default
