.DEFAULT_GOAL := help
API_URL ?= http://localhost:8000
UV := uv run

.PHONY: help install up down logs migrate sample-docs ingest-samples test test-unit lint format typecheck eval eval-quick check

help: ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies (uv)
	uv sync

up: ## Build and start api + postgres
	docker compose up -d --build
	@echo "Waiting for $(API_URL)/ready ..."; \
	for i in $$(seq 1 60); do curl -sf $(API_URL)/ready >/dev/null && echo "ready" && exit 0; sleep 2; done; \
	echo "API did not become ready"; docker compose logs api | tail -50; exit 1

down: ## Stop the stack (keeps data)
	docker compose down

logs: ## Tail API logs
	docker compose logs -f api

migrate: ## Apply migrations to DATABASE_URL (from .env)
	$(UV) alembic upgrade head

sample-docs: ## Regenerate the PDF/DOCX sample documents
	$(UV) python scripts/make_sample_docs.py

ingest-samples: ## Upload every file in sample_docs/ through the API and wait until ready
	$(UV) python scripts/ingest_samples.py --api-url $(API_URL)

test: ## Unit + integration tests with coverage (integration needs Docker)
	$(UV) pytest --cov=app --cov-report=term-missing --cov-fail-under=80

test-unit: ## Fast offline unit tests only
	$(UV) pytest tests/unit -q

lint: ## Ruff lint + format check
	$(UV) ruff check .
	$(UV) ruff format --check .

format: ## Auto-format
	$(UV) ruff check --fix .
	$(UV) ruff format .

typecheck: ## mypy --strict on app/
	$(UV) mypy

eval: ## Full eval: 3 retrieval configs (ablation) against the live DB + LLM
	$(UV) python -m eval.run_eval --ablation

eval-quick: ## Eval the production config only
	$(UV) python -m eval.run_eval

check: lint typecheck test ## Everything CI would run
