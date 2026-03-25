"""Build script for creating the PokerHUD macOS .app bundle.

Supports two back-ends:
  * **PyInstaller** (default) -- ``python scripts/build.py``
  * **py2app**                -- ``python scripts/build.py --backend py2app``

Run from the repository root.
"""

from __future__ import annotations

import argparse
import logging
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.build_config import BuildConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PyInstaller back-end
# ---------------------------------------------------------------------------

def _build_pyinstaller(config: BuildConfig) -> Path:
    """Build the .app bundle using PyInstaller.

    Args:
        config: Build configuration.

    Returns:
        Path to the generated .app bundle.

    Raises:
        RuntimeError: If PyInstaller exits with a non-zero status.
    """
    config.build_dir.mkdir(parents=True, exist_ok=True)
    config.dist_dir.mkdir(parents=True, exist_ok=True)

    # Build --add-data arguments
    add_data_args: list[str] = []
    for dest, sources in config.data_files:
        for src in sources:
            add_data_args.extend(["--add-data", f"{src}:{dest}"])

    # Hidden imports for all src subpackages
    hidden_imports: list[str] = []
    for pkg_dir in config.src_dir.iterdir():
        if pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists():
            hidden_imports.extend(
                ["--hidden-import", f"src.{pkg_dir.name}"]
            )

    cmd: list[str] = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name",
        config.app_name,
        "--windowed",  # .app bundle on macOS
        "--onedir",
        "--distpath",
        str(config.dist_dir),
        "--workpath",
        str(config.build_dir),
        "--specpath",
        str(config.build_dir),
        "--noconfirm",
        *add_data_args,
        *hidden_imports,
    ]

    # Icon (only include if file exists)
    if config.icon_path.is_file():
        cmd.extend(["--icon", str(config.icon_path)])

    # Excludes
    for pkg in config.exclude_packages:
        cmd.extend(["--exclude-module", pkg])

    cmd.append(str(config.entry_point))

    logger.info("Running PyInstaller: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("PyInstaller stdout:\n%s", result.stdout)
        logger.error("PyInstaller stderr:\n%s", result.stderr)
        raise RuntimeError(
            f"PyInstaller failed with exit code {result.returncode}"
        )

    app_path = config.app_path
    if not app_path.exists():
        raise RuntimeError(f"Expected .app bundle not found at {app_path}")

    # Merge custom Info.plist keys into the generated plist
    _merge_plist(app_path / "Contents" / "Info.plist", config.plist_dict)

    logger.info("Built %s successfully", app_path)
    return app_path


# ---------------------------------------------------------------------------
# py2app back-end
# ---------------------------------------------------------------------------

def _build_py2app(config: BuildConfig) -> Path:
    """Build the .app bundle using py2app via setup_py2app.py.

    Args:
        config: Build configuration.

    Returns:
        Path to the generated .app bundle.

    Raises:
        RuntimeError: If the py2app build fails.
    """
    setup_script = Path(__file__).resolve().parent.parent / "setup_py2app.py"
    if not setup_script.exists():
        raise RuntimeError(f"setup_py2app.py not found at {setup_script}")

    config.dist_dir.mkdir(parents=True, exist_ok=True)
    config.build_dir.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        sys.executable,
        str(setup_script),
        "py2app",
        "--dist-dir",
        str(config.dist_dir),
        "--bdist-base",
        str(config.build_dir),
    ]

    logger.info("Running py2app: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("py2app stdout:\n%s", result.stdout)
        logger.error("py2app stderr:\n%s", result.stderr)
        raise RuntimeError(
            f"py2app failed with exit code {result.returncode}"
        )

    app_path = config.app_path
    if not app_path.exists():
        raise RuntimeError(f"Expected .app bundle not found at {app_path}")

    logger.info("Built %s successfully", app_path)
    return app_path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _merge_plist(plist_path: Path, overrides: dict[str, object]) -> None:
    """Merge *overrides* into an existing Info.plist file.

    Args:
        plist_path: Path to the Info.plist to update.
        overrides: Keys and values to set or overwrite.
    """
    if plist_path.exists():
        with open(plist_path, "rb") as fh:
            plist = plistlib.load(fh)
    else:
        plist = {}

    plist.update(overrides)

    with open(plist_path, "wb") as fh:
        plistlib.dump(plist, fh)

    logger.info("Updated Info.plist at %s", plist_path)


def codesign(app_path: Path, identity: str = "-") -> None:
    """Ad-hoc (or real) codesign the .app bundle.

    Args:
        app_path: Path to the .app bundle.
        identity: Signing identity. ``"-"`` means ad-hoc.

    Raises:
        RuntimeError: If codesigning fails.
    """
    if not shutil.which("codesign"):
        logger.warning("codesign not found on PATH; skipping code signing")
        return

    cmd = [
        "codesign",
        "--force",
        "--deep",
        "--sign",
        identity,
        str(app_path),
    ]

    logger.info("Code signing: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("codesign stderr:\n%s", result.stderr)
        raise RuntimeError(
            f"codesign failed with exit code {result.returncode}"
        )

    logger.info("Code signed %s with identity '%s'", app_path, identity)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_app(
    config: BuildConfig | None = None,
    backend: str = "pyinstaller",
) -> Path:
    """Build the PokerHUD .app bundle.

    Args:
        config: Build configuration. Uses defaults when ``None``.
        backend: ``"pyinstaller"`` or ``"py2app"``.

    Returns:
        Path to the generated .app bundle.

    Raises:
        ValueError: If *backend* is not recognised.
        RuntimeError: If the build fails.
    """
    if config is None:
        config = BuildConfig()

    errors = config.validate()
    if errors:
        raise ValueError(
            "Invalid build configuration:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    if backend == "pyinstaller":
        app_path = _build_pyinstaller(config)
    elif backend == "py2app":
        app_path = _build_py2app(config)
    else:
        raise ValueError(f"Unknown build backend: {backend!r}")

    # Ad-hoc code sign
    codesign(app_path, config.codesign_identity)

    return app_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for building the PokerHUD application."""
    parser = argparse.ArgumentParser(
        description="Build the PokerHUD macOS .app bundle"
    )
    parser.add_argument(
        "--backend",
        choices=["pyinstaller", "py2app"],
        default="pyinstaller",
        help="Build back-end to use (default: pyinstaller)",
    )
    parser.add_argument(
        "--codesign-identity",
        default="-",
        help="Code signing identity (default: '-' for ad-hoc)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = BuildConfig(codesign_identity=args.codesign_identity)
    app_path = build_app(config, backend=args.backend)
    print(f"\nBuild complete: {app_path}")


if __name__ == "__main__":
    main()
