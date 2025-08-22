PY_DIR=knowledge_base
JS_DIR=interface

.PHONY: setup lint format test clean api-dev dev-full

setup:
	cd $(PY_DIR) && uv sync --frozen
	cd $(JS_DIR) && pnpm install

api-dev:
	cd $(PY_DIR) && uv run fastapi dev api_server.py

dev-frontend:
	cd $(JS_DIR) && pnpm dev

dev-full: 
	@echo "Starting RAG API server..."
	cd $(PY_DIR) && uv run fastapi dev api_server.py &
	@echo "Starting Next.js frontend..."
	cd $(JS_DIR) && pnpm dev

ingest:
	cd $(PY_DIR) && uv run ingest.py

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
