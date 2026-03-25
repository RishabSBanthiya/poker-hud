.PHONY: test lint format format-check install dev clean check build package release

test:
	python -m pytest tests/ -v

lint:
	python -m ruff check src/ tests/ scripts/

format:
	python -m black src/ tests/ scripts/

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

build:
	python -m scripts.build

package:
	python -m scripts.package

release: build package
	@echo "------------------------------------------------------------"
	@echo "Release build complete."
	@echo "TODO: notarize with 'xcrun notarytool submit dist/PokerHUD-*.dmg'"
	@echo "------------------------------------------------------------"
