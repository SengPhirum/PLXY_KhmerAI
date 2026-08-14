#  Khmer Customer-Support LLM - developer entry points
#  Every target is safe to run repeatedly.  `make help` lists them.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

PYTHON      ?= python3
VENV        ?= .venv
VENV_PY     := $(VENV)/bin/python
VENV_PIP    := $(VENV)/bin/python -m pip
PORT        ?= 8000
HOST        ?= 127.0.0.1

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_.-]+:.*?## ' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
.PHONY: setup
setup: ## Create the virtualenv and install base+server+rag+dev requirements
	bash scripts/bootstrap.sh

.PHONY: setup-training
setup-training: ## Install the heavy training stack (GPU host / Colab only)
	$(VENV_PIP) install -r requirements/training.txt

.PHONY: doctor
doctor: ## Validate the local environment (python, RAM, disk, Ollama, dirs, env)
	$(VENV_PY) scripts/doctor.py

.PHONY: clean
clean: ## Remove caches and build artefacts (keeps data/ and models/)
	find . -path ./$(VENV) -prune -o -type d -name __pycache__ -print0 | xargs -0 rm -rf
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml build dist *.egg-info

# -----------------------------------------------------------------------------
# Quality gates
# -----------------------------------------------------------------------------
.PHONY: format
format: ## Auto-format with ruff
	$(VENV_PY) -m ruff format .
	$(VENV_PY) -m ruff check --fix .

.PHONY: lint
lint: ## Format check + lint (no mutation)
	bash scripts/lint.sh

.PHONY: typecheck
typecheck: ## Static type check
	$(VENV_PY) -m mypy common preprocessing company_data rag server security evaluation

.PHONY: test
test: ## Run the full hermetic test suite
	bash scripts/test.sh

.PHONY: test-unit
test-unit: ## Unit tests only
	$(VENV_PY) -m pytest -q -m "not integration and not e2e and not slow"

.PHONY: test-security
test-security: ## Security + prompt-injection suites
	$(VENV_PY) -m pytest -q security/tests

.PHONY: coverage
coverage: ## Test suite with coverage report
	$(VENV_PY) -m pytest --cov=. --cov-report=term-missing --cov-report=html

.PHONY: ci
ci: lint typecheck test ## Everything CI runs

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
.PHONY: data-public
data-public: ## Download the reviewed public Khmer datasets (network required)
	$(VENV_PY) datasets/download_public.py --config configs/base.yaml --all

.PHONY: data-manifest
data-manifest: ## Rebuild manifests + licence report from data/raw
	$(VENV_PY) datasets/build_manifest.py --root data/raw
	$(VENV_PY) datasets/license_report.py --out data/manifests/license_report.csv

.PHONY: data-clean
data-clean: ## Run the Khmer preprocessing pipeline raw -> cleaned -> deduped
	$(VENV_PY) -m preprocessing.pipeline --input data/raw/public --output data/cleaned --report data/manifests/preprocessing_report.json

.PHONY: data-sft
data-sft: ## Build / validate the SFT dataset splits
	bash scripts/prepare_all_data.sh

.PHONY: company-ingest
company-ingest: ## Normalise + validate company documents into canonical records
	$(VENV_PY) -m company_data.validate --input $${KHMERAI_COMPANY_DOCS_DIR:-data/raw/company} --output data/interim/company_records.jsonl --report data/manifests/company_validation.json

# -----------------------------------------------------------------------------
# RAG
# -----------------------------------------------------------------------------
.PHONY: rag-ingest
rag-ingest: ## Build a NEW index version from company records (atomic, not activated)
	$(VENV_PY) -m rag.ingestion --config configs/rag/ingestion.yaml --input data/interim/company_records.jsonl

.PHONY: rag-reindex
rag-reindex: ## Build -> regression-test -> activate a new index version
	$(VENV_PY) -m rag.reindex --config configs/rag/ingestion.yaml --input data/interim/company_records.jsonl --activate

.PHONY: rag-eval
rag-eval: ## Evaluate retrieval quality (Recall@K / MRR / nDCG)
	$(VENV_PY) -m evaluation.evaluate_retrieval --config configs/rag/retrieval.yaml --golden evaluation/golden/customer_support.jsonl

# -----------------------------------------------------------------------------
# Serving
# -----------------------------------------------------------------------------
.PHONY: serve
serve: ## Run the API with autoreload (development)
	$(VENV_PY) -m uvicorn server.main:app --host $(HOST) --port $(PORT) --reload

.PHONY: serve-prod
serve-prod: ## Run the API the way launchd runs it
	$(VENV_PY) -m uvicorn server.main:app --host $(HOST) --port $(PORT) --workers $${KHMERAI_WORKERS:-1} --no-access-log

.PHONY: ollama-build
ollama-build: ## Create the Ollama models from the Modelfiles
	bash ollama/create_model.sh

.PHONY: ollama-bench
ollama-bench: ## Run the Ollama parallelism / context benchmark matrix
	bash ollama/benchmark.sh

.PHONY: smoke
smoke: ## End-to-end smoke test against a running API
	bash scripts/smoke_test.sh

.PHONY: loadtest
loadtest: ## Async load test (see load_test/README in docs/deployment_guide.md)
	$(VENV_PY) load_test/async_benchmark.py --clients 10 --duration 60

# -----------------------------------------------------------------------------
# Evaluation / release
# -----------------------------------------------------------------------------
.PHONY: evaluate
evaluate: ## Run the whole evaluation battery and write reports
	bash scripts/evaluate_all.sh

.PHONY: release
release: ## Build a release bundle after the gates pass
	bash scripts/build_release.sh

.PHONY: backup
backup: ## Back up index, manifests, configs, prompts and reports
	bash deployment/backup/backup.sh
