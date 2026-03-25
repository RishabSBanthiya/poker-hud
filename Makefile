PYTHON ?= python3
PIP ?= pip3

.PHONY: test lint format format-check install dev clean check build package release

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/

format:
	$(PYTHON) -m black src/ tests/ scripts/

format-check:
	$(PYTHON) -m black --check src/ tests/

check: lint format-check test

install:
	$(PIP) install -e .

dev:
	$(PIP) install -e ".[dev]"

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +

build:
	$(PYTHON) -m scripts.build

package:
	$(PYTHON) -m scripts.package

release: build package
	@echo "------------------------------------------------------------"
	@echo "Release build complete."
	@echo "TODO: notarize with 'xcrun notarytool submit dist/PokerHUD-*.dmg'"
	@echo "------------------------------------------------------------"
