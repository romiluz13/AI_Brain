.PHONY: help install install-dev test test-unit test-integration test-performance lint format type-check security-check clean docs serve-docs build publish

# Default target
help:
	@echo "AI Brain Python - Development Commands"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  setup-pre-commit Setup pre-commit hooks"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-performance Run performance tests only"
	@echo "  test-framework   Run framework-specific tests (FRAMEWORK=crewai|pydantic_ai|agno|langchain|langgraph)"
	@echo "  test-coverage    Run tests with coverage report"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint             Run all linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run mypy type checking"
	@echo "  security-check   Run security analysis with bandit"
	@echo ""
	@echo "Documentation Commands:"
	@echo "  docs             Build documentation"
	@echo "  serve-docs       Serve documentation locally"
	@echo ""
	@echo "Build Commands:"
	@echo "  build            Build package"
	@echo "  publish          Publish to PyPI"
	@echo "  clean            Clean build artifacts"

# Setup Commands
install:
	poetry install --only=main

install-dev:
	poetry install --with=dev

setup-pre-commit:
	poetry run pre-commit install

# Testing Commands
test:
	poetry run pytest

test-unit:
	poetry run pytest tests/unit/ -v

test-integration:
	poetry run pytest tests/integration/ -v

test-performance:
	poetry run pytest tests/performance/ -v --benchmark-only

test-framework:
	@if [ -z "$(FRAMEWORK)" ]; then \
		echo "Error: FRAMEWORK variable is required. Use: make test-framework FRAMEWORK=crewai"; \
		exit 1; \
	fi
	poetry run pytest tests/ -m $(FRAMEWORK) -v

test-coverage:
	poetry run pytest --cov=ai_brain_python --cov-report=html --cov-report=term-missing

# Code Quality Commands
lint: type-check security-check
	poetry run flake8 ai_brain_python/
	poetry run black --check ai_brain_python/
	poetry run isort --check-only ai_brain_python/

format:
	poetry run black ai_brain_python/ tests/
	poetry run isort ai_brain_python/ tests/

type-check:
	poetry run mypy ai_brain_python/

security-check:
	poetry run bandit -r ai_brain_python/ -f json -o bandit-report.json

# Documentation Commands
docs:
	cd docs && poetry run make html

serve-docs:
	cd docs && poetry run make livehtml

# Build Commands
build:
	poetry build

publish:
	poetry publish

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Development Environment
dev-setup: install-dev setup-pre-commit
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify everything is working."

# Quick development cycle
dev-check: format lint test-unit
	@echo "Development checks passed!"

# CI simulation
ci-check: lint test test-coverage security-check
	@echo "CI checks passed!"

# Framework availability check
check-frameworks:
	@echo "Checking framework availability..."
	@poetry run python -c "from ai_brain_python import check_framework_availability; import json; print(json.dumps(check_framework_availability(), indent=2))"

# Database setup for local development
setup-db:
	@echo "Setting up local MongoDB and Redis..."
	docker-compose up -d mongodb redis

# Stop local databases
stop-db:
	docker-compose stop mongodb redis

# Performance benchmarking
benchmark:
	poetry run pytest tests/performance/ --benchmark-only --benchmark-save=benchmark_results

# Generate performance report
benchmark-report:
	poetry run pytest-benchmark compare benchmark_results/*.json --csv=benchmark_comparison.csv
