.PHONY: test lint format format-check install dev clean check

test:
	python -m pytest tests/ -v

lint:
	python -m ruff check src/ tests/

format:
	python -m black src/ tests/

format-check:
	python -m black --check src/ tests/

check: lint format-check test

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
