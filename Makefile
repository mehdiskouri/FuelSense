.PHONY: help install install-forecaster install-anomaly install-optimizer install-all \
lint typecheck test test-django test-ml test-optimizer test-all \
migrate seed train-all bench-forecaster bench-optimizer \
k8s-local k8s-local-gpu k8s-down

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "%-22s %s\n", $$1, $$2}'

install: ## Install Django/base + dev dependencies
	pip install -r requirements/base.txt -r requirements/dev.txt

install-forecaster: ## Install forecaster CPU + dev dependencies
	pip install -r requirements/forecaster-cpu.txt -r requirements/dev.txt

install-anomaly: ## Install anomaly + dev dependencies
	pip install -r requirements/anomaly.txt -r requirements/dev.txt

install-optimizer: ## Install optimizer CPU + dev dependencies
	pip install -r requirements/optimizer-cpu.txt -r requirements/dev.txt

install-all: ## Install all CPU service dependencies + dev dependencies
	pip install -r requirements/base.txt \
	-r requirements/forecaster-cpu.txt \
	-r requirements/anomaly.txt \
	-r requirements/optimizer-cpu.txt \
	-r requirements/dev.txt

lint: ## Run Ruff lint and format checks
	ruff check .
	ruff format --check .

typecheck: ## Run mypy static type checks
	mypy fuelsense/ fuelsense_common/

test: ## Run full test suite with coverage gate
	pytest --cov --cov-report=term-missing --cov-fail-under=95

test-django: ## Run Django tests
	pytest fuelsense/ -v

test-ml: ## Run ML service tests
	pytest forecaster/tests/ anomaly/tests/ -v

test-optimizer: ## Run optimizer service tests
	pytest optimizer/tests/ -v

test-all: ## Run all tests with XML coverage report
	pytest --cov --cov-report=term-missing --cov-report=xml --cov-fail-under=95

migrate: ## Apply Django migrations
	python manage.py migrate

seed: ## Generate synthetic data
	python manage.py generate_synthetic_data --facilities 50 --days 365 --seed 42

train-all: ## Trigger async model training pipeline (use --wait via manage.py for blocking mode)
	python manage.py train_models

bench-forecaster: ## Run CPU and GPU forecaster benchmarks
	python -m forecaster.benchmark --facilities 50 --device cpu
	python -m forecaster.benchmark --facilities 50 --device cuda

bench-optimizer: ## Run CPU and GPU optimizer benchmarks
	python -m optimizer.benchmark --stops 50 --device cpu
	python -m optimizer.benchmark --stops 50 --device cuda

k8s-local: ## Start local kind cluster and deploy helm chart
	kind create cluster --config infra/kind-config.yaml || true
	helm upgrade --install fuelsense charts/fuelsense/ --namespace fuelsense --create-namespace --wait --timeout 300s
	helm test fuelsense -n fuelsense

k8s-local-gpu: ## Deploy helm chart with GPU values
	kind create cluster --config infra/kind-config.yaml || true
	helm upgrade --install fuelsense charts/fuelsense/ -f charts/fuelsense/values-gpu.yaml --namespace fuelsense --create-namespace --wait --timeout 300s
	helm test fuelsense -n fuelsense

k8s-down: ## Tear down local kind cluster and namespace
	helm uninstall fuelsense -n fuelsense || true
	kubectl delete namespace fuelsense --ignore-not-found=true
	kind delete cluster
