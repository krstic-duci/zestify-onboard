PY_DIR=knowledge_base
JS_DIR=interface

.PHONY: setup dev lint format test clean kb if api api-prod

setup:
	cd $(PY_DIR) && uv sync --frozen
	cd $(JS_DIR) && pnpm install

dev:
	cd $(PY_DIR) && uv run main.py

# API server options (choose one)
api:
	cd $(PY_DIR) && uv run fastapi dev src/api_server.py

api-prod:
	cd $(PY_DIR) && uv run fastapi run src/api_server.py

dev-frontend:
	cd $(JS_DIR) && pnpm dev

dev-full: 
	@echo "Starting RAG API server..."
	cd $(PY_DIR) && uv run fastapi dev src/api_server.py &
	@echo "Starting Next.js frontend..."
	cd $(JS_DIR) && pnpm dev

ingest:
	cd $(PY_DIR) && uv run main.py

lint:
	cd $(PY_DIR) && uv run ruff check .
	cd $(JS_DIR) && pnpm lint

format:
	cd $(PY_DIR) && uv run ruff format .
	cd $(JS_DIR) && pnpm format

test:
	cd $(PY_DIR) && pytest
	cd $(JS_DIR) && pnpm test

clean:
	cd $(PY_DIR) && rm -rf .mypy_cache .pytest_cache
	cd $(JS_DIR) && rm -rf .next

kb:
	cd $(PY_DIR)/src

if:
	cd $(JS_DIR)
