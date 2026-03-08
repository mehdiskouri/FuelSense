

Let me first understand the current workspace state and then plan Phase 0 rigorously. 

Read [](file:///workspace/FuelSense)

Read [](file:///workspace/FuelSense/AGENT)

Now let me re-read the exact project structure specification and environment configuration from the architecture doc, plus the full file tree:  

Read [](file:///workspace/FuelSense/AGENT/ARCHITECTURE_FuelSense.md#2570-2570), lines 2570 to 2800

Read [](file:///workspace/FuelSense/AGENT/ARCHITECTURE_FuelSense.md#195-195), lines 195 to 230

Now let me read the CI/CD, Docker, and security sections for completeness:  

Read [](file:///workspace/FuelSense/AGENT/ARCHITECTURE_FuelSense.md#2240-2240), lines 2240 to 2270

Read [](file:///workspace/FuelSense/AGENT/ARCHITECTURE_FuelSense.md#2380-2380), lines 2380 to 2470

I now have full context. Let me also check what the Dockerfile says about python version — I noticed `python:3.13-slim` in the architecture (not 3.12). Let me verify:    

Searched for text `python:3.1` (`**/ARCHITECTURE_FuelSense.md`), 2 results

Searched for text `python-version` (`**/ARCHITECTURE_FuelSense.md`), 2 results

Python 3.13 confirmed across the architecture. Now let me also review the PRD dependencies section for exact version pins: 

Read [](file:///workspace/FuelSense/AGENT/PRD_EnergyLogistics.md#700-700), lines 700 to 707

Good. Now I have everything needed. Let me produce the comprehensive Phase 0 plan.  

Updated memory file [](file:///root/.vscode-server/data/User/workspaceStorage/bd397b02d8fcd0b9bc1791e50a16da50-1/GitHub.copilot-chat/memory-tool/memories/NTgyYzg0YjgtODZkYS00Y2U2LWFhZjgtOGUxMjUwMGYyNTA2/plan.md)

Here is the comprehensive Phase 0 plan:

---

## Plan: Phase 0 — Engineering Foundation & CI Skeleton

Establish the entire project scaffolding, toolchain, CI pipeline skeleton, environment configuration, container definitions, and development Makefile. Every directory, package, requirements file, Dockerfile, compose file, CI workflow, and config artifact specified in the Architecture and PRD is created in this phase. Local development runs natively on the machine — Docker Compose is authored here but exercised only in the CI/CD pipeline.

---

### Phase 0A — Repository Directory Structure

Create the full project tree per Architecture §11. Every package gets an `__init__.py`. Every test directory gets a `conftest.py`.

**Steps**

1. Create the Django project root structure:
   - `fuelsense/` (Django project package) with `__init__.py`, `urls.py`, `wsgi.py`, `asgi.py`, `celery.py`
   - `fuelsense/settings/` with `__init__.py`, `base.py`, `development.py`, `production.py`
   - `fuelsense/core/` with `__init__.py`, `models.py`, `admin.py`, `serializers.py`, `views.py`, `urls.py`, `tasks.py`, `features.py`, `routing.py`
   - `fuelsense/core/tests/` with `__init__.py`, `conftest.py`, `test_models.py`, `test_api.py`, `test_admin.py`, `test_tasks.py`
   - `fuelsense/synthetic/` with `__init__.py`, `generator.py`, `consumption.py`, `weather.py`, `anomalies.py`
   - `fuelsense/synthetic/management/` with `__init__.py`
   - `fuelsense/synthetic/management/commands/` with `__init__.py`, `generate_synthetic_data.py`
   - `manage.py` at project root (inside the repo root, not inside the fuelsense package)

2. Create the shared library:
   - `fuelsense_common/` with `__init__.py`, `compute.py`, `registry.py`, `schemas.py`

3. Create the demand forecaster service:
   - `forecaster/` with `__init__.py`, `model.py`, `service.py`, `benchmark.py`
   - `forecaster/backends/` with `__init__.py`, `cpu_backend.py`, `gpu_backend.py`
   - `forecaster/tests/` with `__init__.py`, `conftest.py`, `test_model.py`, `test_cpu_backend.py`, `test_gpu_backend.py`, `test_service.py`

4. Create the anomaly detector service:
   - `anomaly/` with `__init__.py`, `detector.py`, `service.py`
   - `anomaly/tests/` with `__init__.py`, `conftest.py`, `test_detector.py`, `test_service.py`

5. Create the route optimizer service:
   - `optimizer/` with `__init__.py`, `service.py`, `benchmark.py`
   - `optimizer/backends/` with `__init__.py`, `cpu_backend.py`, `gpu_backend.py`
   - `optimizer/tests/` with `__init__.py`, `conftest.py`, `test_cpu_backend.py`, `test_gpu_backend.py`, `test_service.py`

6. Create the MLOps pipeline:
   - `ml_pipeline/` with `__init__.py`, `training.py`, `drift.py`
   - `ml_pipeline/tests/` with `__init__.py`, `conftest.py`, `test_training.py`, `test_drift.py`

7. Create infrastructure directories:
   - `infra/` with `prometheus.yml` (empty valid YAML), `kind-config.yaml` (empty valid YAML)
   - `infra/grafana/dashboards/` with `operations.json`, `ml-health.json`, `logistics.json`, `system.json` (empty valid JSON `{}`)

8. Create Helm chart skeleton:
   - `charts/fuelsense/` with `Chart.yaml`, `values.yaml`, `values-gpu.yaml`
   - `charts/fuelsense/templates/` with `_helpers.tpl`, `namespace.yaml`, `configmap.yaml`, `secrets.yaml`
   - `charts/fuelsense/templates/django/` with `deployment.yaml`, `service.yaml`, `ingress.yaml`, `hpa.yaml`
   - `charts/fuelsense/templates/celery/` with `worker-default.yaml`, `worker-training.yaml`, `worker-planning.yaml`, `beat.yaml`
   - `charts/fuelsense/templates/ml-services/` with `forecaster.yaml`, `forecaster-hpa.yaml`, `anomaly.yaml`, `optimizer.yaml`
   - `charts/fuelsense/templates/data/` with `postgresql.yaml`, `redis.yaml`, `mlflow.yaml`
   - `charts/fuelsense/templates/monitoring/` with `prometheus.yaml`, `grafana.yaml`, `servicemonitor.yaml`
   - `charts/fuelsense/templates/tests/` with `smoke-test.yaml`

9. Create CI/CD directory:
   - `.github/workflows/` with `ci.yml`, `gpu-tests.yml`

10. Create requirements directory:
    - `requirements/` with `base.txt`, `forecaster-cpu.txt`, `forecaster-gpu.txt`, `anomaly.txt`, `optimizer-cpu.txt`, `optimizer-gpu.txt`, `dev.txt`

**Files created:** ~120 files across the full tree

---

### Phase 0B — Dependency Pinning (7 Requirements Files)

Populate all 7 requirements files with exact version pins per PRD §11 and Architecture dependencies.

**Steps**

1. `requirements/base.txt` — Django stack:
   - `Django>=5.1,<5.2`, `djangorestframework>=3.15,<3.16`, `celery[redis]>=5.4,<5.5`, `django-celery-beat>=2.6,<2.7`, `psycopg[binary]>=3.1,<3.2`, `redis>=5.0,<5.1`, `gunicorn>=22.0,<23.0`, `httpx>=0.27,<0.28`, `django-prometheus>=2.3,<2.4`, `whitenoise>=6.6,<6.7`

2. `requirements/forecaster-cpu.txt` — PyTorch CPU + FastAPI:
   - `torch>=2.2,<2.3` (CPU index), `fastapi>=0.110,<0.112`, `uvicorn[standard]>=0.29,<0.30`, `pydantic>=2.6,<2.7`, `numpy>=1.26,<1.27`, `prometheus-client>=0.20,<0.21`

3. `requirements/forecaster-gpu.txt` — PyTorch CUDA + FastAPI:
   - `torch>=2.2,<2.3` (CUDA index), `pynvml>=11.5,<11.6`, plus all of forecaster-cpu.txt except torch

4. `requirements/anomaly.txt` — scikit-learn + FastAPI:
   - `scikit-learn>=1.4,<1.5`, `joblib>=1.3,<1.4`, `fastapi>=0.110,<0.112`, `uvicorn[standard]>=0.29,<0.30`, `pydantic>=2.6,<2.7`, `numpy>=1.26,<1.27`, `prometheus-client>=0.20,<0.21`

5. `requirements/optimizer-cpu.txt` — OR-Tools + FastAPI:
   - `ortools>=9.9,<9.10`, `fastapi>=0.110,<0.112`, `uvicorn[standard]>=0.29,<0.30`, `pydantic>=2.6,<2.7`, `numpy>=1.26,<1.27`, `prometheus-client>=0.20,<0.21`

6. `requirements/optimizer-gpu.txt` — OR-Tools + PyTorch + FastAPI:
   - Everything from optimizer-cpu.txt plus `torch>=2.2,<2.3` (CUDA index), `pynvml>=11.5,<11.6`

7. `requirements/dev.txt` — Testing and linting:
   - `pytest>=8.0,<8.1`, `pytest-django>=4.8,<4.9`, `pytest-cov>=4.1,<4.2`, `pytest-asyncio>=0.23,<0.24`, `factory-boy>=3.3,<3.4`, `ruff>=0.3,<0.4`, `mypy>=1.9,<1.10`, `django-stubs>=4.2,<4.3`, `httpx>=0.27,<0.28` (for FastAPI TestClient), `mlflow>=2.11,<2.12`

**Relevant files:** All 7 files under `requirements/`

---

### Phase 0C — pyproject.toml (Toolchain Configuration)

Single `pyproject.toml` at the repo root configuring ruff, mypy, pytest, and project metadata.

**Steps**

1. `[project]` section — name `fuelsense`, Python `>=3.13`, description, author
2. `[tool.ruff]` — target Python 3.13, line length 120, select rules (E, F, W, I, B, UP, SIM, N), ignore specific rules as needed
3. `[tool.ruff.lint.isort]` — known-first-party = `["fuelsense", "fuelsense_common", "forecaster", "anomaly", "optimizer", "ml_pipeline"]`
4. `[tool.mypy]` — strict mode for `fuelsense/`, gradual for ML services, plugins = `["mypy_django_plugin.main"]`, `django_settings_module = "fuelsense.settings.development"`, disallow_untyped_defs = true, warn_return_any = true
5. `[tool.mypy.overrides]` — per-module overrides for `forecaster.*`, `anomaly.*`, `optimizer.*`, `ml_pipeline.*` with `disallow_untyped_defs = false` (gradual adoption)
6. `[tool.pytest.ini_options]` — `DJANGO_SETTINGS_MODULE = "fuelsense.settings.development"`, `python_files = "test_*.py"`, `python_classes = "Test*"`, `python_functions = "test_*"`, `addopts = "--strict-markers --tb=short -q"`, `markers` for `slow`, `gpu`, `integration`
7. `[tool.coverage.run]` — source = `["fuelsense", "fuelsense_common", "forecaster", "anomaly", "optimizer", "ml_pipeline"]`, omit = `["*/tests/*", "*/migrations/*"]`
8. `[tool.coverage.report]` — `fail_under = 95`, `show_missing = true`, `exclude_lines` for standard pragmas

**Relevant files:** `pyproject.toml`

---

### Phase 0D — Environment Configuration

**Steps**

1. `.env.example` — All variables from Architecture §3.3 plus Django/database/Redis/Celery/MLflow:
   - `DJANGO_SETTINGS_MODULE=fuelsense.settings.development`
   - `SECRET_KEY=change-me-in-production`
   - `DATABASE_URL=postgresql://fuelsense:dev_password@localhost:5432/fuelsense`
   - `REDIS_URL=redis://localhost:6379/0`
   - `CELERY_BROKER_URL=redis://localhost:6379/1`
   - `MLFLOW_TRACKING_URI=http://localhost:5000`
   - `FUELSENSE_DEVICE=auto`
   - `FUELSENSE_CUDA_MEMORY_FRACTION=0.8`
   - `FUELSENSE_CUDA_ALLOW_TF32=true`
   - `FUELSENSE_CUDA_BENCHMARK=true`
   - `FUELSENSE_FORECAST_BATCH_SIZE_CPU=16`
   - `FUELSENSE_FORECAST_BATCH_SIZE_GPU=256`
   - `FUELSENSE_FORECAST_TRAINING_BATCH_SIZE_CPU=32`
   - `FUELSENSE_FORECAST_TRAINING_BATCH_SIZE_GPU=512`
   - `FUELSENSE_OPTIMIZER_BACKEND=ortools`
   - `FUELSENSE_OPTIMIZER_TIME_LIMIT_MS=10000`
   - `MODEL_PATH=/models/demand_tcn.pt`
   - `POSTGRES_DB=fuelsense`
   - `POSTGRES_USER=fuelsense`
   - `POSTGRES_PASSWORD=dev_password`

2. `.gitignore` — Python bytecode, `.env`, `__pycache__/`, `*.pyc`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`, `htmlcov/`, `*.egg-info/`, `dist/`, `build/`, `.coverage`, `db.sqlite3`, `staticfiles/`, `models/*.pt`, `models/*.joblib`, `mlartifacts/`, `pgdata/`, `node_modules/`, `.venv/`, `*.log`

**Relevant files:** `.env.example`, `.gitignore`

---

### Phase 0E — Makefile (17 Targets, Native Local Execution)

Adapted from Architecture §11.1. Local targets run natively (no docker compose). Compose targets retained for CI/CD reference but clearly documented as CI-only. 

**Steps**

1. `help` — grep + awk for target descriptions
2. `install` — `pip install -r requirements/base.txt -r requirements/dev.txt` (local dev dependency install)
3. `install-forecaster` — `pip install -r requirements/forecaster-cpu.txt -r requirements/dev.txt`
4. `install-anomaly` — `pip install -r requirements/anomaly.txt -r requirements/dev.txt`
5. `install-optimizer` — `pip install -r requirements/optimizer-cpu.txt -r requirements/dev.txt`
6. `install-all` — install all CPU requirements + dev
7. `lint` — `ruff check . && ruff format --check .`
8. `typecheck` — `mypy fuelsense/ fuelsense_common/`
9. `test` — `pytest --cov --cov-report=term-missing --cov-fail-under=95`
10. `test-django` — `pytest fuelsense/ -v`
11. `test-ml` — `pytest forecaster/tests/ anomaly/tests/ -v`
12. `test-optimizer` — `pytest optimizer/tests/ -v`
13. `test-all` — `pytest --cov --cov-report=term-missing --cov-report=xml --cov-fail-under=95`
14. `migrate` — `python manage.py migrate`
15. `seed` — `python manage.py generate_synthetic_data --facilities 50 --days 365`
16. `train-all` — `python manage.py train_models`
17. `bench-forecaster` — `python -m forecaster.benchmark --facilities 50 --device cpu && python -m forecaster.benchmark --facilities 50 --device cuda`
18. `bench-optimizer` — `python -m optimizer.benchmark --stops 50 --device cpu && python -m optimizer.benchmark --stops 50 --device cuda`
19. `k8s-local` — `kind create cluster --config infra/kind-config.yaml || true && helm install fuelsense charts/fuelsense/`
20. `k8s-local-gpu` — same with `-f charts/fuelsense/values-gpu.yaml`
21. `k8s-down` — `helm uninstall fuelsense || true && kind delete cluster`

**Relevant files:** `Makefile`

---

### Phase 0F — 6 Dockerfiles + 2 Compose Files

Create all 6 Dockerfiles exactly per Architecture §9.1 and both compose files per §9.2–9.3. These are authored now, validated syntactically, but exercised only in the CI pipeline.

**Steps**

1. `Dockerfile.django` — Multi-stage build: `python:3.13-slim` base, `libpq-dev gcc`, `requirements/base.txt`, collectstatic, gunicorn on :8000
2. `Dockerfile.forecaster` — CPU variant: `python:3.13-slim`, `requirements/forecaster-cpu.txt`, copy `forecaster/` + `fuelsense_common/`, uvicorn on :8001
3. `Dockerfile.forecaster-gpu` — GPU variant: `nvidia/cuda:12.4-runtime-ubuntu22.04`, python3 + pip, `requirements/forecaster-gpu.txt`, copy `forecaster/` + `fuelsense_common/`, uvicorn on :8001
4. `Dockerfile.anomaly` — `python:3.13-slim`, `requirements/anomaly.txt`, copy `anomaly/` + `fuelsense_common/`, uvicorn on :8002
5. `Dockerfile.optimizer` — `python:3.13-slim`, `requirements/optimizer-cpu.txt`, copy `optimizer/` + `fuelsense_common/`, uvicorn on :8003
6. `Dockerfile.optimizer-gpu` — `nvidia/cuda:12.4-runtime-ubuntu22.04`, `requirements/optimizer-gpu.txt`, copy `optimizer/` + `fuelsense_common/`, uvicorn on :8003
7. `docker-compose.yml` — All 13 services: postgres, redis, django, celery-default, celery-training, celery-planning, celery-beat, forecaster, anomaly, optimizer, mlflow, prometheus, grafana. Exact config from Architecture §9.2.
8. `docker-compose.gpu.yml` — Override for forecaster and optimizer with GPU Dockerfiles and NVIDIA runtime reservations per Architecture §9.3.

**Relevant files:** 6 Dockerfiles, `docker-compose.yml`, `docker-compose.gpu.yml`

---

### Phase 0G — CI/CD Pipeline Skeletons

**Steps**

1. `.github/workflows/ci.yml` — Full 5-job structure per Architecture §12:
   - **Job 1: lint-and-typecheck** — `ruff check .`, `ruff format --check .`, `mypy fuelsense/ fuelsense_common/`. Runs on all pushes and PRs. Active from Phase 0.
   - **Job 2: test-django** — PostgreSQL 16 + Redis 7 services. `pip install -r requirements/base.txt -r requirements/dev.txt`. `pytest fuelsense/ --cov --cov-report=xml --cov-fail-under=95`. Stub for now (no code to test yet), activates in Phase 1.
   - **Job 3: test-ml-services** — Install per-service requirements. `pytest forecaster/tests/`, `pytest anomaly/tests/`, `pytest optimizer/tests/`. Stub, activates in Phases 3–5.
   - **Job 4: build-images** — Needs test jobs. Matrix build: 4 Docker images (django, forecaster, anomaly, optimizer). Push to `ghcr.io` with `${{ github.sha }}` tag. Only on main. Stub, activates in Phase 6.
   - **Job 5: deploy-staging** — Needs build-images. `helm upgrade --install`. Stub.
   - **Job 6: smoke-test** — Needs deploy. Health check curls. Stub.

2. `.github/workflows/gpu-tests.yml` — Manual trigger (`workflow_dispatch`):
   - GPU backend test jobs for forecaster and optimizer
   - Runs `pytest forecaster/tests/test_gpu_backend.py`, `pytest optimizer/tests/test_gpu_backend.py`
   - Uses self-hosted runner with GPU access

**Relevant files:** `.github/workflows/ci.yml`, `.github/workflows/gpu-tests.yml`

---

### Phase 0H — Minimal Skeleton Code (Importable Packages)

Each Python file created in 0A gets minimal valid content so the project is immediately importable and lint/typecheck passes on the empty codebase.

**Steps**

1. All `__init__.py` files — empty or with `"""Package docstring."""`
2. `manage.py` — Standard Django manage.py pointing to `fuelsense.settings.development`
3. `fuelsense/settings/base.py` — Minimal Django settings: `INSTALLED_APPS` with `django.contrib.admin`, `django.contrib.auth`, `django.contrib.contenttypes`, `django.contrib.sessions`, `django.contrib.messages`, `django.contrib.staticfiles`, `rest_framework`, `django_celery_beat`, `django_prometheus`, `fuelsense.core`, `fuelsense.synthetic`. `DATABASES` from `DATABASE_URL` env var. `ROOT_URLCONF = "fuelsense.urls"`. `DEFAULT_AUTO_FIELD`. `REST_FRAMEWORK` with pagination and auth defaults. `CELERY_BROKER_URL` from env. Static files config.
4. `fuelsense/settings/development.py` — imports base, `DEBUG = True`, `ALLOWED_HOSTS = ["*"]`
5. `fuelsense/settings/production.py` — imports base, `DEBUG = False`, `ALLOWED_HOSTS` from env, `SECURE_*` settings, `CSRF_TRUSTED_ORIGINS`
6. `fuelsense/urls.py` — Admin URL + API include + health check stubs
7. `fuelsense/wsgi.py` + `fuelsense/asgi.py` — Standard Django WSGI/ASGI config
8. `fuelsense/celery.py` — Celery app with `autodiscover_tasks`, 3 queues (default, training, planning)
9. `fuelsense/core/models.py` — Empty, just a docstring comment placeholder
10. `fuelsense/core/admin.py` — Empty with `from django.contrib import admin`
11. All other `.py` files — Docstring placeholder only, no implementation

**Relevant files:** Every `.py` file in the tree (~60 Python files)

---

### Verification (Phase 0 Quality Gate)

1. **Structure** — All ~120 files exist at correct paths. `find . -name "*.py" | wc -l` matches expected count.
2. **Lint** — `make lint` passes with zero findings on the skeleton codebase.
3. **Typecheck** — `make typecheck` passes with zero errors.
4. **Import** — `python -c "import fuelsense; import fuelsense_common; import forecaster; import anomaly; import optimizer; import ml_pipeline"` succeeds.
5. **Django check** — `python manage.py check` reports no issues.
6. **Pytest** — `pytest --collect-only` discovers test files without errors (0 tests collected is fine at this stage).
7. **Dockerfile syntax** — `docker build --check` (or `hadolint`) on all 6 Dockerfiles passes.
8. **Compose syntax** — `docker compose -f docker-compose.yml config` and `docker compose -f docker-compose.yml -f docker-compose.gpu.yml config` both validate.
9. **CI pipeline** — Push triggers CI, lint-and-typecheck job runs green. Other jobs are stubs and skip gracefully.
10. **Helm lint** — `helm lint charts/fuelsense/` passes on the skeleton chart.
11. **Coverage config** — `pytest --cov --cov-report=term-missing` runs without config errors (reports 0% coverage at this stage, which is fine — the fail_under=95 gate is not enforced until code exists).

---

### Decisions

- Local dev runs natively; Docker Compose is CI/CD-only infrastructure tested via pipeline.
- Python 3.13 per Architecture Dockerfiles and CI config.
- Coverage gate at 95% enforced via `pyproject.toml` `fail_under` — active immediately but only blocks once real code lands.
- All 6 Dockerfiles and both compose files authored in Phase 0 so the CI image-build pipeline has artifacts from the start.
- Makefile targets for local development use native commands; no `docker compose exec` wrappers.
- GPU Dockerfiles use `nvidia/cuda:12.4-runtime-ubuntu22.04` as base per Architecture §9.1.
- Helm chart skeleton uses placeholder values; real template content comes in Phase 7.

---

### Relevant Files (Complete Phase 0 Deliverable Inventory)

- `pyproject.toml` — ruff, mypy, pytest, coverage, project metadata
- `.env.example` — all environment variables with defaults and descriptions
- `.gitignore` — comprehensive Python/Django/ML/infra ignore rules
- `Makefile` — 21 targets for native local development and K8s operations
- `requirements/*.txt` — 7 pinned dependency files
- `Dockerfile.django`, `Dockerfile.forecaster`, `Dockerfile.forecaster-gpu`, `Dockerfile.anomaly`, `Dockerfile.optimizer`, `Dockerfile.optimizer-gpu` — 6 container definitions
- `docker-compose.yml` — 13-service CPU development stack
- `docker-compose.gpu.yml` — GPU override
- `.github/workflows/ci.yml` — 6-job CI/CD pipeline (lint active, rest stubbed)
- `.github/workflows/gpu-tests.yml` — manual GPU test workflow
- `fuelsense/settings/base.py`, `development.py`, `production.py` — Django settings split
- `fuelsense/celery.py` — Celery app with 3 queues
- `manage.py` — Django management entry point
- `charts/fuelsense/Chart.yaml`, `values.yaml`, `values-gpu.yaml` + 23 template stubs
- `infra/prometheus.yml`, `infra/kind-config.yaml`, `infra/grafana/dashboards/*.json` — infra config placeholders
- ~60 Python files across 8 packages with valid imports

