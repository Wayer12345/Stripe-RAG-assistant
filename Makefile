# Makefile — stripe-rag-assistant
# Local-first RAG system over Stripe Guides.
# All commands assume the venv is activated or PYTHON points to the venv interpreter.

PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON     := venv/bin/python
UVICORN    := venv/bin/uvicorn
PYTEST     := venv/bin/pytest
RUFF       := venv/bin/ruff
MYPY       := venv/bin/mypy
RUN_PYTHON := PYTHONPATH="$(PROJECT_ROOT)" $(PYTHON)

APP_HOST   := 0.0.0.0
APP_PORT   := 8000

QDRANT_IMAGE   := qdrant/qdrant
QDRANT_STORAGE := $(PWD)/qdrant_storage
QDRANT_HOST    := http://localhost:6333
OLLAMA_HOST    := http://localhost:11434

DOCKER_IMAGE            := stripe-rag-assistant
DOCKER_CONTAINER        := stripe-rag-assistant
DOCKER_APP_PORT         := 8000
DOCKER_OLLAMA_BASE_URL  := http://host.docker.internal:11434
DOCKER_QDRANT_VOLUME    := $(PROJECT_ROOT)/data/indexes/qdrant

Q ?=
INPUT ?=
DATASET ?= tests/fixtures/sample_eval.jsonl
CHUNKS ?= data/processed/chunks.jsonl
RUN_ID ?= eval_run
DATASET_ID ?= eval_dataset
LIMIT ?=

.DEFAULT_GOAL := help

# ─────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────

.PHONY: help
help: ## Show this help message
	@echo ""
	@echo "stripe-rag-assistant — available make targets"
	@echo "────────────────────────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*##"}; {printf "  %-28s %s\n", $$1, $$2}'
	@echo ""

# ─────────────────────────────────────────────
# Environment setup
# ─────────────────────────────────────────────

.PHONY: venv
venv: ## Create Python 3.12 virtual environment
	python3.12 -m venv venv
	$(PYTHON) -m ensurepip --upgrade
	$(PYTHON) -m pip install --upgrade pip setuptools wheel

.PHONY: deps
deps: ## Install project dependencies (runtime + dev)
	$(PYTHON) -m pip install -e ".[dev]"

# ─────────────────────────────────────────────
# Code quality
# ─────────────────────────────────────────────

.PHONY: lint
lint: ## Run ruff linter (check only)
	$(RUFF) check app tests scripts

.PHONY: lint-fix
lint-fix: ## Run ruff linter and auto-fix safe issues
	$(RUFF) check --fix app tests scripts

.PHONY: format
format: ## Run ruff formatter
	$(RUFF) format app tests scripts

.PHONY: format-check
format-check: ## Check formatting without applying changes
	$(RUFF) format --check app tests scripts

.PHONY: typecheck
typecheck: ## Run mypy type checker on app/
	$(MYPY) app

.PHONY: check
check: lint format-check typecheck ## Run all quality checks (lint + format + types)

# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

.PHONY: test
test: ## Run all tests
	$(PYTEST)

.PHONY: test-unit
test-unit: ## Run unit tests only
	$(PYTEST) -m unit

.PHONY: test-integration
test-integration: ## Run integration tests only
	$(PYTEST) -m integration

.PHONY: test-e2e
test-e2e: ## Run end-to-end smoke tests
	$(PYTEST) -m e2e

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	$(PYTEST) --cov=app --cov-report=term-missing --cov-report=html:htmlcov

.PHONY: test-fast
test-fast: ## Run tests, stop on first failure
	$(PYTEST) -x

# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────

.PHONY: api
api: ## Start FastAPI dev server (hot-reload)
	$(UVICORN) main:app --host $(APP_HOST) --port $(APP_PORT) --reload

.PHONY: api-prod
api-prod: ## Start FastAPI server without hot-reload
	$(UVICORN) main:app --host $(APP_HOST) --port $(APP_PORT)

.PHONY: health
health: ## Check FastAPI health endpoint
	curl -s http://localhost:$(APP_PORT)/health | python3 -m json.tool

# ─────────────────────────────────────────────
# Qdrant
# ─────────────────────────────────────────────

.PHONY: qdrant-start
qdrant-start: ## Start Qdrant in Docker (persist data locally)
	@mkdir -p $(QDRANT_STORAGE)
	@docker rm -f qdrant >/dev/null 2>&1 || true
	docker run -d --name qdrant \
		-p 6333:6333 -p 6334:6334 \
		-v "$(QDRANT_STORAGE):/qdrant/storage:z" \
		$(QDRANT_IMAGE)
	@echo "Qdrant started. Dashboard: http://localhost:6333/dashboard"

.PHONY: qdrant-stop
qdrant-stop: ## Stop the Qdrant container
	docker stop qdrant && docker rm qdrant

.PHONY: qdrant-health
qdrant-health: ## Check Qdrant health
	curl -s $(QDRANT_HOST)/healthz

# ─────────────────────────────────────────────
# Ollama
# ─────────────────────────────────────────────

.PHONY: ollama
ollama: ## Start local Ollama server
	ollama serve

.PHONY: ollama-check
ollama-check: ## List models available in local Ollama
	curl -s $(OLLAMA_HOST)/api/tags | $(PYTHON) -m json.tool

.PHONY: ollama-pull
ollama-pull: ## Pull the default LLM model defined in configs/config.yaml
	@MODEL=$$($(PYTHON) -c "from pathlib import Path; from app.utils.config import load_settings; print(load_settings(Path('configs')).generation.model_name)"); \
	echo "Pulling $$MODEL …"; \
	ollama pull $$MODEL

# ─────────────────────────────────────────────
# Application services
# ─────────────────────────────────────────────

.PHONY: service-index
service-index: ## Run app.application.indexing_service
	$(RUN_PYTHON) -m app.application.indexing_service --config configs/config.yaml

.PHONY: service-query
service-query: ## Run app.application.query_service (Q required)
	@if [ -z "$(Q)" ]; then echo "Usage: make service-query Q='your question'"; exit 1; fi
	$(RUN_PYTHON) -m app.application.query_service --question "$(Q)" --config configs/config.yaml

.PHONY: service-eval
service-eval: ## Run app.application.eval_service (DATASET, RUN_ID, LIMIT optional)
	$(RUN_PYTHON) -m app.application.eval_service --dataset "$(DATASET)" --run-id-prefix "$(RUN_ID)" --config configs/config.yaml \
		$(if $(LIMIT),--limit $(LIMIT),)

# ─────────────────────────────────────────────
# Offline application layers
# ─────────────────────────────────────────────

.PHONY: offline-parse
offline-parse: ## Run app.application_layers.offline.build_parsed_docs
	$(RUN_PYTHON) -m app.application_layers.offline.build_parsed_docs --config configs/config.yaml

.PHONY: offline-clean
offline-clean: ## Run app.application_layers.offline.build_cleaned_docs
	$(RUN_PYTHON) -m app.application_layers.offline.build_cleaned_docs --config configs/config.yaml

.PHONY: offline-chunk
offline-chunk: ## Run app.application_layers.offline.build_chunks
	$(RUN_PYTHON) -m app.application_layers.offline.build_chunks --config configs/config.yaml

.PHONY: offline-embed
offline-embed: ## Run app.application_layers.offline.build_embeddings
	$(RUN_PYTHON) -m app.application_layers.offline.build_embeddings --config configs/config.yaml

.PHONY: offline-index
offline-index: ## Run app.application_layers.offline.build_vector_index
	$(RUN_PYTHON) -m app.application_layers.offline.build_vector_index --config configs/config.yaml

# ─────────────────────────────────────────────
# Online application layers
# ─────────────────────────────────────────────

.PHONY: online-retrieve
online-retrieve: ## Run app.application_layers.online.retrieve (Q required)
	@if [ -z "$(Q)" ]; then echo "Usage: make online-retrieve Q='your question'"; exit 1; fi
	$(RUN_PYTHON) -m app.application_layers.online.retrieve --question "$(Q)" --config configs/config.yaml

.PHONY: online-rerank
online-rerank: ## Run app.application_layers.online.rerank (Q + INPUT required)
	@if [ -z "$(Q)" ]; then echo "Usage: make online-rerank Q='your question' INPUT='data/path.json'"; exit 1; fi
	@if [ -z "$(INPUT)" ]; then echo "Usage: make online-rerank Q='your question' INPUT='data/path.json'"; exit 1; fi
	$(RUN_PYTHON) -m app.application_layers.online.rerank --question "$(Q)" --input-path "$(INPUT)" --config configs/config.yaml

.PHONY: online-context
online-context: ## Run app.application_layers.online.build_context (Q + INPUT required)
	@if [ -z "$(Q)" ]; then echo "Usage: make online-context Q='your question' INPUT='data/path.json'"; exit 1; fi
	@if [ -z "$(INPUT)" ]; then echo "Usage: make online-context Q='your question' INPUT='data/path.json'"; exit 1; fi
	$(RUN_PYTHON) -m app.application_layers.online.build_context --question "$(Q)" --input-path "$(INPUT)" --config configs/config.yaml

.PHONY: online-answer
online-answer: ## Run app.application_layers.online.generate_answer (Q + INPUT required)
	@if [ -z "$(Q)" ]; then echo "Usage: make online-answer Q='your question' INPUT='data/path.json'"; exit 1; fi
	@if [ -z "$(INPUT)" ]; then echo "Usage: make online-answer Q='your question' INPUT='data/path.json'"; exit 1; fi
	$(RUN_PYTHON) -m app.application_layers.online.generate_answer --question "$(Q)" --input-path "$(INPUT)" --config configs/config.yaml

# ─────────────────────────────────────────────
# Eval application layers
# ─────────────────────────────────────────────

.PHONY: eval-build-dataset
eval-build-dataset: ## Run app.application_layers.eval.build_eval_dataset
	$(RUN_PYTHON) -m app.application_layers.eval.build_eval_dataset --chunks "$(CHUNKS)" --output-dir "data/eval/datasets" --dataset-id "$(DATASET_ID)" --config configs/config.yaml

.PHONY: eval-retrieval
eval-retrieval: ## Run app.application_layers.eval.run_retrieval_eval
	$(RUN_PYTHON) -m app.application_layers.eval.run_retrieval_eval --dataset "$(DATASET)" --run-id "$(RUN_ID)" --config configs/config.yaml

.PHONY: eval-rerank
eval-rerank: ## Run app.application_layers.eval.run_rerank_eval
	$(RUN_PYTHON) -m app.application_layers.eval.run_rerank_eval --dataset "$(DATASET)" --run-id "$(RUN_ID)" --config configs/config.yaml

.PHONY: eval-context
eval-context: ## Run app.application_layers.eval.run_context_eval
	$(RUN_PYTHON) -m app.application_layers.eval.run_context_eval --dataset "$(DATASET)" --run-id "$(RUN_ID)" --config configs/config.yaml

.PHONY: eval-generation
eval-generation: ## Run app.application_layers.eval.run_generation_eval
	$(RUN_PYTHON) -m app.application_layers.eval.run_generation_eval --dataset "$(DATASET)" --run-id "$(RUN_ID)" --config configs/config.yaml

.PHONY: eval-citation
eval-citation: ## Run app.application_layers.eval.run_citation_eval
	$(RUN_PYTHON) -m app.application_layers.eval.run_citation_eval --dataset "$(DATASET)" --run-id "$(RUN_ID)" --config configs/config.yaml

.PHONY: eval-robustness
eval-robustness: ## Run app.application_layers.eval.run_robustness_eval
	$(RUN_PYTHON) -m app.application_layers.eval.run_robustness_eval --dataset "$(DATASET)" --run-id "$(RUN_ID)" --config configs/config.yaml

# ─────────────────────────────────────────────
# Docker
# ─────────────────────────────────────────────

.PHONY: docker-build
docker-build: ## Build API Docker image
	docker build \
		--build-arg OLLAMA_BASE_URL=$(DOCKER_OLLAMA_BASE_URL) \
		-t $(DOCKER_IMAGE) .

.PHONY: docker-up
docker-up: ## Start API container in background
	@mkdir -p "$(DOCKER_QDRANT_VOLUME)"
	@docker rm -f $(DOCKER_CONTAINER) >/dev/null 2>&1 || true
	@docker image inspect $(DOCKER_IMAGE) >/dev/null 2>&1 || (echo "Image $(DOCKER_IMAGE) not found. Run: make docker-build"; exit 1)
	docker run -d --name $(DOCKER_CONTAINER) \
		-p $(APP_PORT):$(DOCKER_APP_PORT) \
		-v "$(DOCKER_QDRANT_VOLUME):/app/data/indexes/qdrant" \
		$(DOCKER_IMAGE)

.PHONY: docker-down
docker-down: ## Stop and remove API container
	@docker rm -f $(DOCKER_CONTAINER) >/dev/null 2>&1 || true

.PHONY: docker-logs
docker-logs: ## Follow API container logs
	docker logs -f $(DOCKER_CONTAINER)

.PHONY: docker-health
docker-health: ## Check /health on local API container
	curl -s http://localhost:$(APP_PORT)/health | python3 -m json.tool

# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────

.PHONY: clean-cache
clean-cache: ## Remove Python bytecode and cache dirs
	find . -type d -name "__pycache__" -not -path "./venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -not -path "./venv/*" -delete 2>/dev/null || true
	@echo "Cache cleaned."

.PHONY: clean-cov
clean-cov: ## Remove coverage reports
	rm -rf htmlcov .coverage

.PHONY: clean
clean: clean-cache clean-cov ## Clean all generated files (cache + coverage)
