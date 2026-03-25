"""Tests verifying the project structure and package setup (S1-01)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestPackageStructure:
    """Verify all expected packages are importable."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "src",
            "src.common",
            "src.common.logging",
            "src.common.config",
            "src.common.performance",
            "src.capture",
            "src.detection",
            "src.engine",
            "src.overlay",
            "src.solver",
            "src.stats",
        ],
    )
    def test_module_is_importable(self, module_path: str) -> None:
        mod = importlib.import_module(module_path)
        assert mod is not None

    def test_src_has_version(self) -> None:
        import src

        assert hasattr(src, "__version__")
        assert src.__version__ == "0.1.0"


class TestInitFiles:
    """Verify __init__.py files exist in all packages."""

    @pytest.mark.parametrize(
        "package_dir",
        [
            "src",
            "src/common",
            "src/capture",
            "src/detection",
            "src/engine",
            "src/overlay",
            "src/solver",
            "src/stats",
        ],
    )
    def test_init_file_exists(self, package_dir: str) -> None:
        init_file = PROJECT_ROOT / package_dir / "__init__.py"
        assert init_file.exists(), f"Missing __init__.py in {package_dir}"


class TestDependencies:
    """Verify key dependencies are installed and importable."""

    @pytest.mark.parametrize(
        "package",
        [
            "structlog",
            "yaml",
            "numpy",
        ],
    )
    def test_dependency_importable(self, package: str) -> None:
        importlib.import_module(package)
