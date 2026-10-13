PYTHON ?= python
.DEFAULT_GOAL := help

.PHONY: help install lint lint-fix format format-check typecheck test check dashboard docker-build docker-up docker-down docker-logs

help:
	@echo "Usage: make <command>"
	@echo ""
	@echo "Setup"
	@echo "  install       Install project dependencies"
	@echo ""
	@echo "Quality checks"
	@echo "  lint          Check Python lint rules"
	@echo "  lint-fix      Automatically fix supported lint issues"
	@echo "  format        Format Python files"
	@echo "  format-check  Check formatting without changing files"
	@echo "  typecheck     Check types in src/"
	@echo "  test          Run tests in tests/"
	@echo "  check         Run all quality checks and tests"
	@echo ""
	@echo "Local development"
	@echo "  dashboard     Start Streamlit dashboard on port 8501"
	@echo ""
	@echo "Docker (requires Docker to be running)"
	@echo "  docker-build  Build service images"
	@echo "  docker-up     Build and start services in the background"
	@echo "  docker-down   Stop and remove containers and networks"
	@echo "  docker-logs   Follow service logs (Ctrl+C to exit)"
	@echo ""
	@echo "  help          Show this help"

install:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m ruff check .

lint-fix:
	$(PYTHON) -m ruff check . --fix

format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest tests

check:
	$(MAKE) lint
	$(MAKE) format-check
	$(MAKE) typecheck
	$(MAKE) test

dashboard:
	python -m streamlit run src/dashboard/dashboard.py --server.address=0.0.0.0 --server.port=8501

docker-build:
	docker compose build

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f
