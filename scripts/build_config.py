"""Build configuration for the PokerHUD macOS application bundle.

Centralises all build metadata (name, version, paths, inclusions/exclusions,
code-signing identity) so that the build and packaging scripts share a single
source of truth.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Path constants (relative to the repository root)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
DIST_DIR = REPO_ROOT / "dist"
BUILD_DIR = REPO_ROOT / "build"
INFO_PLIST_TEMPLATE = REPO_ROOT / "Info.plist"
ICON_PATH = REPO_ROOT / "assets" / "icon.icns"
ENTRY_POINT = REPO_ROOT / "main.py"


def _read_version_from_pyproject() -> str:
    """Read the project version from pyproject.toml.

    Returns:
        The version string (e.g. ``"0.1.0"``).

    Raises:
        FileNotFoundError: If pyproject.toml is missing.
        KeyError: If the version key is not present.
    """
    with open(PYPROJECT_TOML, "rb") as fh:
        data = tomllib.load(fh)
    return data["project"]["version"]


# ---------------------------------------------------------------------------
# Data-file collection helpers
# ---------------------------------------------------------------------------

def _collect_data_files() -> list[tuple[str, list[str]]]:
    """Collect data files to include in the application bundle.

    Returns a list of ``(destination_dir, [source_file, ...])`` tuples that
    is compatible with both py2app ``data_files`` and PyInstaller
    ``--add-data`` conventions.

    Returns:
        List of (destination, sources) pairs.
    """
    pairs: list[tuple[str, list[str]]] = []

    # data/templates/
    templates_dir = DATA_DIR / "templates"
    if templates_dir.is_dir():
        template_files = [
            str(p) for p in templates_dir.rglob("*") if p.is_file()
        ]
        if template_files:
            pairs.append(("data/templates", template_files))

    # models/
    if MODELS_DIR.is_dir():
        model_files = [
            str(p)
            for p in MODELS_DIR.rglob("*")
            if p.is_file() and p.name != ".gitkeep"
        ]
        if model_files:
            pairs.append(("models", model_files))

    return pairs


# ---------------------------------------------------------------------------
# Exclusion lists
# ---------------------------------------------------------------------------

EXCLUDE_PACKAGES: list[str] = [
    "tests",
    "tests.unit",
    "tests.integration",
    "tests.fixtures",
]

EXCLUDE_PATTERNS: list[str] = [
    "*.pyc",
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "*.egg-info",
]


# ---------------------------------------------------------------------------
# BuildConfig dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildConfig:
    """Immutable build configuration for the PokerHUD macOS .app bundle.

    Attributes:
        app_name: Display name of the application.
        bundle_identifier: macOS CFBundleIdentifier.
        version: Semantic version read from pyproject.toml.
        icon_path: Path to the application icon (.icns).
        entry_point: Path to the main Python entry point.
        src_dir: Path to the ``src/`` package tree.
        data_files: Data files to embed in the bundle.
        exclude_packages: Packages to exclude from the bundle.
        exclude_patterns: Glob patterns for files/dirs to exclude.
        plist_template: Path to the Info.plist template.
        dist_dir: Output directory for the built .app and DMG.
        build_dir: Intermediate build artefacts directory.
        codesign_identity: Apple Developer ID for code signing.
            Use ``"-"`` for ad-hoc signing during development.
        bundle_python_runtime: Whether to embed the Python interpreter.
        ls_ui_element: If True the app runs as a menu-bar agent
            (no Dock icon).
    """

    app_name: str = "PokerHUD"
    bundle_identifier: str = "com.pokerhud.app"
    version: str = ""
    icon_path: Path = ICON_PATH
    entry_point: Path = ENTRY_POINT
    src_dir: Path = SRC_DIR
    data_files: list[tuple[str, list[str]]] = field(default_factory=list)
    exclude_packages: list[str] = field(
        default_factory=lambda: list(EXCLUDE_PACKAGES)
    )
    exclude_patterns: list[str] = field(
        default_factory=lambda: list(EXCLUDE_PATTERNS)
    )
    plist_template: Path = INFO_PLIST_TEMPLATE
    dist_dir: Path = DIST_DIR
    build_dir: Path = BUILD_DIR
    codesign_identity: str = "-"  # ad-hoc; replace with Developer ID
    bundle_python_runtime: bool = True
    ls_ui_element: bool = True

    def __post_init__(self) -> None:
        """Resolve version from pyproject.toml when not set explicitly."""
        if not self.version:
            # frozen dataclass — use object.__setattr__
            object.__setattr__(self, "version", _read_version_from_pyproject())
        if not self.data_files:
            object.__setattr__(self, "data_files", _collect_data_files())

    # -- Convenience helpers ------------------------------------------------

    @property
    def plist_dict(self) -> dict[str, object]:
        """Return the Info.plist key/value pairs for the .app bundle.

        Returns:
            Dictionary suitable for plistlib serialisation.
        """
        return {
            "CFBundleName": self.app_name,
            "CFBundleDisplayName": self.app_name,
            "CFBundleIdentifier": self.bundle_identifier,
            "CFBundleVersion": self.version,
            "CFBundleShortVersionString": self.version,
            "CFBundlePackageType": "APPL",
            "CFBundleSignature": "????",
            "CFBundleExecutable": self.app_name,
            "LSUIElement": self.ls_ui_element,
            "NSScreenCaptureUsageDescription": (
                "PokerHUD needs Screen Recording access to capture the "
                "poker table for real-time card detection and HUD overlay."
            ),
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "13.0",
        }

    @property
    def app_path(self) -> Path:
        """Path to the built .app bundle inside dist/."""
        return self.dist_dir / f"{self.app_name}.app"

    @property
    def dmg_path(self) -> Path:
        """Path to the output DMG installer."""
        return self.dist_dir / f"{self.app_name}-{self.version}.dmg"

    def validate(self) -> list[str]:
        """Check the configuration for common problems.

        Returns:
            A list of human-readable error strings. An empty list means
            the configuration is valid.
        """
        errors: list[str] = []
        if not self.app_name:
            errors.append("app_name must not be empty")
        if not self.bundle_identifier:
            errors.append("bundle_identifier must not be empty")
        if not self.version:
            errors.append("version must not be empty")
        if not self.entry_point.name.endswith(".py"):
            errors.append(
                f"entry_point must be a .py file, got {self.entry_point.name}"
            )
        if self.icon_path.suffix != ".icns" and str(self.icon_path) != str(ICON_PATH):
            errors.append(
                f"icon_path should be an .icns file, got {self.icon_path.suffix}"
            )
        return errors
