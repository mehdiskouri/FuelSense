# PRD: FuelSense — Energy Logistics Optimization & MLOps Platform

**Version:** 0.1-draft  
**Author:** Mehdi  
**Date:** March 2026  
**Status:** Architecture Phase

---

## 1. Vision & Objective

FuelSense is a full-stack energy logistics platform that optimizes fuel supply chain operations for power generation facilities. The system forecasts per-facility fuel demand, detects consumption anomalies indicating equipment degradation or theft, and solves vehicle routing problems to minimize delivery cost and latency under real-world constraints (time windows, vehicle capacity, road distance, storage limits).

The portfolio objective is threefold: demonstrate **Django mastery** (admin customization, model architecture, DRF API, Celery orchestration), **MLOps engineering** (PyTorch training pipelines, MLflow experiment tracking, drift detection, automated retraining, model versioning), and **Kubernetes deployment** (multi-service orchestration, Helm charts, horizontal pod autoscaling, CI/CD with GitHub Actions).

---

## 2. Core Concepts & Definitions

**Facility:** A power plant, industrial site, or storage terminal that consumes fuel. Each facility has a fuel type, storage capacity, minimum safe inventory level, and a consumption profile that varies by season, weather, and operational load.

**Depot:** A fuel distribution hub with vehicle fleet and fuel inventory. Depots serve a regional set of facilities. The system supports multiple depots operating concurrently.

**Delivery Window:** Each facility specifies acceptable delivery times (e.g., 06:00–14:00 weekdays only). Deliveries outside windows incur penalty costs or are rejected outright.

**Reorder Point (dynamic):** Unlike a static threshold, FuelSense computes a per-facility reorder point as: `R = forecast_demand(lead_time_days) + safety_stock(forecast_uncertainty)`. When current inventory crosses R, the facility enters the planning queue.

**Demand Forecast:** A per-facility time-series model predicting daily fuel consumption over a 14-day horizon. Inputs: historical consumption, weather forecast (temperature, wind, solar irradiance), day-of-week, facility metadata. Model: temporal convolutional network (TCN) implemented in PyTorch, chosen over RNNs for parallelizable training and stable gradients over the 14-day horizon.

**Consumption Anomaly Score:** A per-facility score computed daily as the normalized residual between actual consumption and the forecaster's prediction. Scores exceeding a calibrated threshold (default: 3σ) trigger alerts. An Isolation Forest (scikit-learn) trained on residual features provides a secondary anomaly classification with type labels: `LEAK`, `THEFT`, `EQUIPMENT_DEGRADATION`, `DEMAND_SHIFT`, `SENSOR_FAULT`.

**Vehicle Routing Problem with Time Windows (VRPTW):** Given a set of facilities needing delivery, vehicle fleet capacity, depot locations, travel time matrix, and delivery windows, find the minimum-cost set of routes. Solved via Google OR-Tools CP-SAT solver with configurable objective weights on total distance, time window violations, vehicle count, and fuel waste (partial loads).

**Model Drift:** Measured as the ratio of recent prediction RMSE (last 7 days) to baseline RMSE (validation set at training time). Drift ratio > 1.5 triggers automated retraining. Tracked per facility and globally.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     CLIENT LAYER                         │
│          Django Admin Dashboard / REST API                │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                 DJANGO APPLICATION                        │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐  │
│  │ Admin Views   │ │ DRF API      │ │ Celery Tasks    │  │
│  │ (operations)  │ │ (programmatic│ │ (async jobs)    │  │
│  │              │ │  access)     │ │                 │  │
│  └──────────────┘ └──────────────┘ └────────┬────────┘  │
└──────────┬──────────────────────────────────┼───────────┘
           │                                  │
     ┌─────▼─────┐                   ┌────────▼────────┐
     │ PostgreSQL │                   │     Redis       │
     │            │                   │ (cache + broker)│
     └───────────┘                   └─────────────────┘
                                              │
                    ┌─────────────────────────┼──────────────────┐
                    │              CELERY WORKERS                 │
                    │  ┌────────────┐ ┌──────────┐ ┌──────────┐  │
                    │  │ Data       │ │ Training │ │ Planning  │  │
                    │  │ Ingestion  │ │ Pipeline │ │ Cycle     │  │
                    │  └────────────┘ └──────────┘ └──────────┘  │
                    └─────────────────────────┬──────────────────┘
                                              │
              ┌───────────────────────────────┼──────────────────┐
              │           ML SERVICES (Kubernetes)               │
              │  ┌──────────────┐ ┌────────────┐ ┌───────────┐  │
              │  │ Demand       │ │ Anomaly    │ │ Route     │  │
              │  │ Forecaster   │ │ Detector   │ │ Optimizer │  │
              │  │ (PyTorch TCN)│ │ (sklearn)  │ │ (OR-Tools)│  │
              │  └──────────────┘ └────────────┘ └───────────┘  │
              └──────────────────────────────────────────────────┘
                                     │
                              ┌──────▼──────┐
                              │   MLflow    │
                              │   Server    │
                              └─────────────┘
```

---

## 4. Service Specifications

### 4.1 Django Application

**Framework:** Django 5.1+ with Django REST Framework 3.15+ for API, Celery 5.4+ for async task orchestration, django-celery-beat for periodic scheduling.

#### Django Models (core):

```python
class FuelType(models.Model):
    name = models.CharField(max_length=50)            # "LNG", "Diesel", "HFO", "Natural Gas"
    unit = models.CharField(max_length=20)            # "m3", "tonnes", "MWh"
    density_kg_per_unit = models.FloatField()
    hazmat_class = models.CharField(max_length=10, blank=True)

class Facility(models.Model):
    name = models.CharField(max_length=200)
    facility_type = models.CharField(choices=FACILITY_TYPES)  # POWER_PLANT, INDUSTRIAL, STORAGE
    fuel_type = models.ForeignKey(FuelType, on_delete=models.PROTECT)
    latitude = models.FloatField()
    longitude = models.FloatField()
    storage_capacity = models.FloatField()            # max fuel storage in fuel_type.unit
    current_inventory = models.FloatField()
    min_safe_inventory = models.FloatField()           # hard floor, emergency only
    dynamic_reorder_point = models.FloatField(null=True)  # computed by forecaster
    delivery_window_start = models.TimeField()
    delivery_window_end = models.TimeField()
    delivery_days = models.JSONField(default=list)     # [0,1,2,3,4] = weekdays
    is_active = models.BooleanField(default=True)

    class Meta:
        indexes = [
            models.Index(fields=["fuel_type", "is_active"]),
            models.Index(fields=["current_inventory"]),
        ]

class Depot(models.Model):
    name = models.CharField(max_length=200)
    latitude = models.FloatField()
    longitude = models.FloatField()
    fuel_type = models.ForeignKey(FuelType, on_delete=models.PROTECT)
    fuel_inventory = models.FloatField()
    facilities = models.ManyToManyField(Facility, through="DepotFacilityAssignment")

class Vehicle(models.Model):
    depot = models.ForeignKey(Depot, on_delete=models.CASCADE, related_name="vehicles")
    registration = models.CharField(max_length=50, unique=True)
    capacity = models.FloatField()                     # in fuel_type.unit
    cost_per_km = models.FloatField()
    is_available = models.BooleanField(default=True)

class InventoryLog(models.Model):
    """Time-series of inventory levels per facility. One row per day."""
    facility = models.ForeignKey(Facility, on_delete=models.CASCADE, related_name="inventory_logs")
    timestamp = models.DateTimeField()
    inventory_level = models.FloatField()
    consumption = models.FloatField()                  # consumed since last log
    temperature = models.FloatField(null=True)
    wind_speed = models.FloatField(null=True)
    solar_irradiance = models.FloatField(null=True)

    class Meta:
        indexes = [
            models.Index(fields=["facility", "-timestamp"]),
        ]
        unique_together = [["facility", "timestamp"]]

class Delivery(models.Model):
    depot = models.ForeignKey(Depot, on_delete=models.CASCADE)
    vehicle = models.ForeignKey(Vehicle, on_delete=models.CASCADE)
    planned_date = models.DateField()
    status = models.CharField(choices=DELIVERY_STATUSES)  # PLANNED, IN_TRANSIT, DELIVERED, FAILED
    total_distance_km = models.FloatField(null=True)
    total_cost = models.FloatField(null=True)
    route_json = models.JSONField(null=True)           # ordered list of stops with ETAs
    solver_time_ms = models.FloatField(null=True)
    created_by_planning_cycle = models.ForeignKey("PlanningCycle", null=True, on_delete=models.SET_NULL)

class DeliveryItem(models.Model):
    delivery = models.ForeignKey(Delivery, on_delete=models.CASCADE, related_name="items")
    facility = models.ForeignKey(Facility, on_delete=models.CASCADE)
    quantity = models.FloatField()
    planned_arrival = models.DateTimeField()
    actual_arrival = models.DateTimeField(null=True)
    sequence = models.PositiveIntegerField()           # stop order in route

class Forecast(models.Model):
    facility = models.ForeignKey(Facility, on_delete=models.CASCADE, related_name="forecasts")
    model_version = models.CharField(max_length=100)   # MLflow run_id
    created_at = models.DateTimeField(auto_now_add=True)
    horizon_days = models.IntegerField(default=14)
    predictions_json = models.JSONField()              # [{day: 1, value: 42.5, lower: 38, upper: 47}, ...]
    rmse = models.FloatField(null=True)                # evaluated after actuals arrive

class AnomalyAlert(models.Model):
    facility = models.ForeignKey(Facility, on_delete=models.CASCADE, related_name="alerts")
    timestamp = models.DateTimeField()
    anomaly_type = models.CharField(choices=ANOMALY_TYPES)
    score = models.FloatField()
    actual_consumption = models.FloatField()
    predicted_consumption = models.FloatField()
    is_acknowledged = models.BooleanField(default=False)
    acknowledged_by = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)
    notes = models.TextField(blank=True)

class PlanningCycle(models.Model):
    """Audit log for each delivery planning run."""
    triggered_at = models.DateTimeField(auto_now_add=True)
    trigger_type = models.CharField(choices=TRIGGER_TYPES)  # SCHEDULED, MANUAL, EMERGENCY
    facilities_in_queue = models.IntegerField()
    deliveries_created = models.IntegerField()
    total_distance_km = models.FloatField()
    total_cost = models.FloatField()
    solver_time_ms = models.FloatField()
    baseline_cost = models.FloatField(null=True)       # naive FIFO for comparison
    cost_reduction_pct = models.FloatField(null=True)  # vs baseline

class ModelRegistry(models.Model):
    """Tracks deployed ML models and their performance."""
    model_type = models.CharField(choices=MODEL_TYPES)  # DEMAND_FORECAST, ANOMALY_DETECTOR
    facility = models.ForeignKey(Facility, null=True, on_delete=models.CASCADE)  # null = global model
    mlflow_run_id = models.CharField(max_length=100)
    version = models.IntegerField()
    is_active = models.BooleanField(default=False)
    trained_at = models.DateTimeField()
    training_rmse = models.FloatField(null=True)
    validation_rmse = models.FloatField(null=True)
    drift_ratio = models.FloatField(default=1.0)
    last_drift_check = models.DateTimeField(null=True)
```

#### Custom Django Admin:

This is where Django expertise becomes visible. Not just default ModelAdmin, but:

- **FacilityAdmin** with inline InventoryLog chart (last 30 days), forecast overlay, anomaly alert list, and a custom action "Trigger Emergency Delivery" that bypasses the normal planning cycle.
- **DeliveryAdmin** with route map rendering (static image generated server-side via matplotlib + basemap), status timeline, and cost breakdown.
- **Dashboard view** (custom admin page, not a model) showing: facilities below reorder point, active deliveries in transit, today's anomaly alerts, model drift status heatmap, and KPI cards (avg delivery cost, forecast accuracy, anomaly detection rate).
- **PlanningCycleAdmin** with side-by-side comparison: optimized vs baseline cost, with percentage savings highlighted.
- **ModelRegistryAdmin** with promote/rollback actions — click to set a model version as active, with confirmation and audit trail.

#### REST API (DRF):

```
GET    /api/v1/facilities/                    # list with filtering, pagination
GET    /api/v1/facilities/{id}/               # detail with current inventory + reorder status
GET    /api/v1/facilities/{id}/inventory/     # time-series inventory logs
GET    /api/v1/facilities/{id}/forecasts/     # latest forecast with confidence intervals
GET    /api/v1/facilities/{id}/alerts/        # anomaly alerts, filterable by type/status
POST   /api/v1/facilities/{id}/acknowledge/   # acknowledge an alert

GET    /api/v1/deliveries/                    # list with status filter
GET    /api/v1/deliveries/{id}/               # detail with route and items
POST   /api/v1/deliveries/{id}/update-status/ # mark delivered/failed

POST   /api/v1/planning/trigger/              # manually trigger planning cycle
GET    /api/v1/planning/history/              # past planning cycles with metrics

GET    /api/v1/models/                        # registered models with drift status
POST   /api/v1/models/{id}/promote/           # promote a model version
POST   /api/v1/models/{id}/retrain/           # trigger async retraining

GET    /api/v1/dashboard/kpis/                # aggregated KPIs for dashboard
GET    /api/v1/dashboard/drift-heatmap/       # per-facility drift ratios
```

Authentication via Django's built-in session auth for admin, token auth (DRF TokenAuthentication) for API consumers.

### 4.2 ML Services

Each model runs as a standalone FastAPI microservice deployed in Kubernetes. Django communicates with them via internal HTTP calls through Celery tasks.

#### Demand Forecaster (PyTorch)

**Architecture:** Temporal Convolutional Network (TCN) with the following structure:

```
Input: [batch, 90, 6]
  - 90 days lookback
  - 6 channels: consumption, temperature, wind, solar, day_of_week_sin, day_of_week_cos

TCN Blocks (3 layers):
  - Layer 1: dilation=1, channels=32, kernel=3
  - Layer 2: dilation=2, channels=32, kernel=3
  - Layer 3: dilation=4, channels=32, kernel=3
  Each block: Conv1d → BatchNorm → ReLU → Dropout(0.2) → residual connection

Output head: Linear(32 → 14)
  - 14-day forecast, single value per day

Loss: Quantile loss at [0.1, 0.5, 0.9] for prediction intervals
Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
Scheduler: CosineAnnealingLR over 100 epochs
```

**Training pipeline:**
1. Extract training data from PostgreSQL via SQLAlchemy (read-only connection)
2. Per-facility train/val/test split: last 14 days = test, previous 14 = val, rest = train
3. Train with early stopping on validation quantile loss (patience=10)
4. Log params, metrics, and model artifact to MLflow
5. If val RMSE < current active model RMSE: auto-promote (with manual override flag)

**Serving endpoint:**
```
POST /predict
{
  "facility_id": 42,
  "lookback": [[consumption, temp, wind, solar, dow_sin, dow_cos], ...],  // 90 rows
}
→ {
  "forecast": [
    {"day": 1, "p10": 38.2, "p50": 42.5, "p90": 47.1},
    ...  // 14 days
  ],
  "model_version": "v12",
  "inference_time_ms": 4.2
}
```

#### Anomaly Detector (scikit-learn)

**Architecture:** Two-stage detection.

Stage 1 — residual scoring: compute `z = (actual - predicted) / rolling_std` per facility per day. Flag if `|z| > 3.0`.

Stage 2 — classification: Isolation Forest trained on a feature vector per flagged day:
```
Features:
  - z_score (normalized residual)
  - z_score_rolling_3d (3-day average of |z|)
  - consumption_delta_pct (% change from previous day)
  - temperature_residual (actual temp - seasonal norm)
  - day_of_week
  - hours_since_last_delivery
  - inventory_level_pct (current / capacity)
```

Output: anomaly type classification + confidence score. The model is trained on synthetic labeled anomalies injected into historical data (known patterns for each anomaly type).

**Serving endpoint:**
```
POST /detect
{
  "facility_id": 42,
  "actual_consumption": 55.2,
  "predicted_consumption": 42.5,
  "features": { ... }
}
→ {
  "is_anomaly": true,
  "score": 0.87,
  "anomaly_type": "EQUIPMENT_DEGRADATION",
  "confidence": 0.73
}
```

#### Route Optimizer (OR-Tools)

**Solver:** Google OR-Tools CVRPTW (Capacitated Vehicle Routing Problem with Time Windows).

**Input:**
```json
{
  "depot": {"id": 1, "lat": 35.78, "lng": -5.81},
  "vehicles": [
    {"id": 1, "capacity": 30.0, "cost_per_km": 2.5},
    {"id": 2, "capacity": 30.0, "cost_per_km": 2.5}
  ],
  "stops": [
    {
      "facility_id": 42,
      "lat": 35.72, "lng": -5.90,
      "demand": 18.5,
      "window_start": "06:00",
      "window_end": "14:00",
      "service_time_min": 30
    }
  ],
  "max_route_duration_min": 480,
  "objective_weights": {
    "total_distance": 1.0,
    "vehicle_count": 50.0,
    "time_window_penalty": 100.0
  }
}
```

**Output:**
```json
{
  "routes": [
    {
      "vehicle_id": 1,
      "stops": [
        {"facility_id": 42, "eta": "07:30", "quantity": 18.5, "sequence": 1},
        {"facility_id": 17, "eta": "09:15", "quantity": 24.0, "sequence": 2}
      ],
      "total_distance_km": 87.3,
      "total_cost": 218.25
    }
  ],
  "total_distance_km": 87.3,
  "total_cost": 218.25,
  "vehicles_used": 1,
  "solver_time_ms": 342,
  "baseline_cost": 312.50,
  "cost_reduction_pct": 30.2
}
```

**Baseline comparison:** Every optimization run also computes a naive baseline (nearest-facility-first greedy assignment, one vehicle per facility) and reports the percentage improvement. This is the key measurable metric for the resume.

### 4.3 Celery Task Orchestration

```python
# Periodic tasks (celery beat)

@app.task
def ingest_consumption_data():
    """Runs every hour. Pulls latest consumption readings for all active facilities.
    In production: API calls to SCADA/IoT endpoints.
    In demo: synthetic data generator with realistic patterns."""
    ...

@app.task
def run_daily_forecasts():
    """Runs daily at 02:00. For each active facility, calls the demand forecaster
    service, stores predictions, updates dynamic_reorder_point on Facility model."""
    ...

@app.task
def run_anomaly_detection():
    """Runs daily at 03:00. For each facility with yesterday's actuals, computes
    residual vs forecast, calls anomaly detector service if flagged, creates
    AnomalyAlert records."""
    ...

@app.task
def run_planning_cycle():
    """Runs daily at 05:00. Queries facilities where current_inventory <=
    dynamic_reorder_point, groups by depot, calls route optimizer for each depot,
    creates Delivery and DeliveryItem records."""
    ...

@app.task
def check_model_drift():
    """Runs daily at 04:00. For each active model, computes recent RMSE vs baseline.
    If drift_ratio > 1.5, triggers retrain_model task."""
    ...

# On-demand tasks

@app.task
def retrain_model(facility_id, model_type):
    """Extracts training data, trains model, logs to MLflow, updates ModelRegistry.
    Auto-promotes if validation RMSE improves."""
    ...

@app.task
def trigger_emergency_delivery(facility_id):
    """Bypasses normal planning cycle. Single-facility route optimization
    with highest priority. Called from admin action."""
    ...
```

### 4.4 Data Layer

**PostgreSQL:** All Django models above. Standard relational schema, Django migrations manage everything. Key performance considerations:
- `InventoryLog` is the largest table: ~50 facilities × 365 days/year = 18,250 rows/year. Trivial volume. Indexed on `(facility_id, -timestamp)` for fast lookback queries.
- `Forecast.predictions_json` stores the 14-day prediction array as JSON. Queried by facility, not by individual prediction day, so JSON is appropriate.
- Composite index on `AnomalyAlert(facility_id, timestamp, is_acknowledged)` for the admin dashboard query.

**Redis:**
```
cache:facility:{id}:latest_inventory    → float (fast reads for dashboard)
cache:facility:{id}:reorder_status      → "OK" | "WARNING" | "CRITICAL"
cache:dashboard:kpis                    → JSON (refreshed every 5 min)
celery broker queues                    → default, training, planning
```

**MLflow:**
- Tracking server with PostgreSQL backend store and local artifact store (or S3 in production)
- Experiments: `demand-forecaster`, `anomaly-detector`
- Each run logs: hyperparameters, training/validation metrics, model artifact (PyTorch state_dict or joblib pickle), training data hash, facility_id

---

## 5. Kubernetes Architecture

```yaml
# Namespace: fuelsense

# --- Core Application ---
django-app:
  type: Deployment
  replicas: 2
  image: fuelsense/django:latest
  ports: [8000]
  resources:
    requests: { cpu: 250m, memory: 512Mi }
    limits: { cpu: 500m, memory: 1Gi }
  env_from: [fuelsense-secrets, fuelsense-config]
  probes:
    liveness: /healthz
    readiness: /readyz

django-ingress:
  type: Ingress
  rules: [host: fuelsense.local → django-app:8000]

# --- Data Layer ---
postgresql:
  type: StatefulSet
  replicas: 1
  storage: 10Gi PVC
  image: postgres:16

redis:
  type: Deployment
  replicas: 1
  image: redis:7-alpine

# --- Task Workers ---
celery-worker-default:
  type: Deployment
  replicas: 2
  command: celery -A fuelsense worker -Q default -c 4

celery-worker-training:
  type: Deployment
  replicas: 1
  command: celery -A fuelsense worker -Q training -c 2
  resources:
    requests: { cpu: 1, memory: 2Gi }    # training is CPU-heavy
    limits: { cpu: 2, memory: 4Gi }

celery-beat:
  type: Deployment
  replicas: 1                              # exactly one scheduler
  command: celery -A fuelsense beat

# --- ML Services ---
demand-forecaster:
  type: Deployment
  replicas: 2
  image: fuelsense/forecaster:latest
  ports: [8001]
  hpa:
    min: 1, max: 4
    metric: cpu, target: 70%

anomaly-detector:
  type: Deployment
  replicas: 1
  image: fuelsense/anomaly:latest
  ports: [8002]

route-optimizer:
  type: Deployment
  replicas: 1
  image: fuelsense/optimizer:latest
  ports: [8003]
  resources:
    requests: { cpu: 500m, memory: 1Gi }  # OR-Tools is CPU-intensive

# --- MLOps ---
mlflow-server:
  type: Deployment
  replicas: 1
  image: fuelsense/mlflow:latest
  ports: [5000]
  volumes: [mlflow-artifacts: 20Gi PVC]
```

**Helm chart** packages all manifests with configurable values for resource limits, replica counts, image tags, and environment-specific secrets.

**CI/CD (GitHub Actions):**
```
on push to main:
  1. Run Django tests (pytest)
  2. Run ML service tests (pytest)
  3. Build Docker images (multi-stage, minimal)
  4. Push to container registry
  5. Deploy to Kind cluster (local) or staging K8s
  6. Run integration smoke tests
  7. Tag release if all pass
```

---

## 6. Synthetic Data Generator

Since this is a portfolio project, a realistic synthetic data generator replaces real SCADA/IoT feeds. The generator is a Django management command: `python manage.py generate_synthetic_data --facilities 50 --days 365`.

**Facility consumption model:**
```
C(t) = base_load
     + seasonal_component(t)           # sinusoidal, period=365d
     + temperature_response(T(t))      # piecewise linear (heating + cooling)
     + weekday_factor(t)               # 0.85 on weekends
     + noise(t)                        # Gaussian, σ = 0.05 × base_load
     + anomaly_injection(t)            # see below
```

**Weather model:** Synthetic temperature, wind, solar based on location latitude and seasonal profiles with realistic autocorrelation (AR(1) process).

**Anomaly injection:** Each facility has a 5% daily probability of one of:
- `LEAK`: consumption increases by 15-30% for 3-7 consecutive days
- `THEFT`: single-day spike of 20-50% above normal
- `EQUIPMENT_DEGRADATION`: gradual consumption increase of 2% per day over 10-20 days
- `DEMAND_SHIFT`: permanent step change in base_load (simulates new equipment or tenant)
- `SENSOR_FAULT`: consumption reads as 0 or NaN for 1-3 days

Labels are stored separately for training the anomaly classifier and evaluating detection accuracy.

---

## 7. Phased Delivery Plan

### Phase 1 — Django Foundation + Synthetic Data (days 1-2)

**Deliverables:** Django project scaffolded. All models migrated. Custom admin with facility dashboard, inline charts, and delivery management. Synthetic data generator producing 1 year of realistic consumption data for 50 facilities across 5 depots. REST API endpoints operational with DRF. Basic Celery setup with Redis broker.

**Success criteria:** Admin dashboard renders facility inventory charts, shows facilities below reorder point, and supports manual delivery creation. API returns paginated, filtered facility and delivery data.

### Phase 2 — ML Services + MLOps Pipeline (days 3-4)

**Deliverables:** PyTorch TCN demand forecaster trained on synthetic data, serving predictions via FastAPI microservice. Anomaly detector trained and serving. MLflow tracking all training runs. Celery tasks for daily forecast, anomaly detection, and drift checking operational. ModelRegistry admin with promote/rollback actions.

**Success criteria:** Demand forecaster achieves MAPE < 8% on held-out test set. Anomaly detector achieves precision > 0.80 and recall > 0.70 on injected anomalies. Drift detection correctly triggers retraining when test-set distribution is artificially shifted.

### Phase 3 — Route Optimization + Integration (days 5-6)

**Deliverables:** OR-Tools route optimizer service deployed. Full planning cycle operational: forecast → reorder detection → route optimization → delivery creation. Baseline comparison computed for every planning run. PlanningCycle admin shows optimized vs naive cost.

**Success criteria:** Route optimization achieves > 20% cost reduction vs naive baseline averaged across 30 days of simulated planning cycles. Full tick cycle (forecast + detect + plan) completes in < 60 seconds for 50 facilities.

### Phase 4 — Kubernetes + CI/CD + Polish (days 7-8)

**Deliverables:** All services containerized with multi-stage Docker builds. Helm chart packaging full deployment. GitHub Actions CI pipeline running tests, building images, deploying to Kind cluster. HPA configured for demand forecaster. README, architecture diagram, and demo walkthrough.

**Success criteria:** `helm install fuelsense ./charts/fuelsense` brings up entire stack from zero. CI pipeline passes end-to-end. Forecaster HPA scales from 1 to 3 replicas under synthetic load test.

---

## 8. Monitoring & Observability

**Prometheus metrics:**

Django: `http_request_duration_seconds` (histogram by endpoint), `celery_task_duration_seconds` (histogram by task), `celery_task_failures_total` (counter), `active_anomaly_alerts` (gauge), `facilities_below_reorder` (gauge).

ML services: `inference_latency_ms` (histogram by service), `model_version_active` (info gauge), `forecast_rmse_recent` (gauge per facility), `drift_ratio` (gauge per facility), `retraining_triggered_total` (counter).

Route optimizer: `solver_time_ms` (histogram), `cost_reduction_pct` (histogram), `vehicles_used` (histogram), `planning_cycles_total` (counter).

**Grafana dashboards:**

- **Operations:** Facility inventory heatmap, deliveries in transit, anomaly alert feed
- **ML Health:** Per-facility forecast accuracy over time, drift ratio heatmap, retraining history
- **Logistics:** Route optimization savings trend, vehicle utilization, delivery on-time rate
- **System:** Service latencies, Celery queue depths, pod scaling events

---

## 9. Testing Strategy

**Django:**
- Unit tests: Model methods, custom querysets, reorder point computation, serializer validation
- Integration tests: Full API request/response cycles with factory-generated data (factory_boy)
- Admin tests: Custom views render correctly, actions execute and produce expected side effects
- Coverage target: > 85% on Django app code

**ML Services:**
- Unit tests: TCN forward pass shape correctness, loss computation, feature engineering functions
- Model tests: Train on small synthetic dataset, assert MAPE below threshold, assert prediction shape and bounds
- Contract tests: Request/response schema validation for each service endpoint
- Drift detection test: Artificially shift input distribution, assert drift ratio exceeds threshold and retraining triggers

**Route Optimizer:**
- Unit tests: Known small instances with hand-computed optimal solutions
- Property tests: For any valid input, assert all demands satisfied, no vehicle exceeds capacity, all time windows respected (or penalty correctly computed)
- Benchmark test: 50-facility instance solves in < 10 seconds

**Integration:**
- End-to-end: Synthetic data → forecast → anomaly detect → plan → deliver cycle, assert database state consistency at each step
- Kubernetes: Helm install on Kind, smoke test all service endpoints, verify HPA triggers under load

---

## 10. Resume-Ready Metrics

After completion, the following quantified claims can be placed on the resume:

- "Built energy logistics platform with PyTorch demand forecaster achieving X% MAPE across 50 facilities"
- "Implemented VRPTW route optimization reducing delivery costs by Y% vs naive baseline (OR-Tools)"
- "Deployed MLOps pipeline with automated drift detection and retraining, maintaining forecast accuracy within Z% of baseline over 12-month simulation"
- "Orchestrated N Kubernetes services with Helm, HPA scaling, and CI/CD via GitHub Actions"
- "Designed Django admin dashboard with custom views, inline analytics, and model promotion workflow"

These are the numbers that go on the resume and get discussed in interviews. Every architectural decision in this PRD exists to produce a defensible, quantified claim.

---

## 11. Dependencies

- **Django 5.1+**: Web framework, admin, ORM
- **Django REST Framework 3.15+**: API layer
- **Celery 5.4+**: Async task queue
- **django-celery-beat**: Periodic task scheduling
- **PyTorch 2.2+**: Demand forecaster (TCN)
- **scikit-learn 1.4+**: Anomaly detector (Isolation Forest)
- **Google OR-Tools 9.9+**: Route optimizer (VRPTW solver)
- **MLflow 2.11+**: Experiment tracking, model registry
- **FastAPI 0.110+**: ML service serving layer
- **PostgreSQL 16**: Primary database
- **Redis 7**: Cache + Celery broker
- **Docker + Docker Compose**: Local development
- **Kubernetes + Helm**: Production deployment
- **Kind**: Local K8s cluster for development/CI
- **GitHub Actions**: CI/CD pipeline
- **Prometheus + Grafana**: Monitoring and dashboards
- **factory_boy + pytest-django**: Testing
