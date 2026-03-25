.PHONY: test lint format install dev clean

test:
	python -m pytest tests/ -v

lint:
	python -m ruff check src/ tests/

format:
	python -m black src/ tests/

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
