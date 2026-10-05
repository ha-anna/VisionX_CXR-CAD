PYTHON ?= python
.DEFAULT_GOAL := help

.PHONY: help install lint format format-check typecheck test check

help:
	@echo "install       Install project dependencies"
	@echo "lint          Check lint rules"
	@echo "format        Format Python files"
	@echo "format-check  Check formatting without changing files"
	@echo "typecheck     Check types"
	@echo "test          Run tests"
	@echo "check         Run all quality checks and tests"

install:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m ruff check .

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