"""Packaging script for creating a DMG installer from the PokerHUD .app bundle.

Usage::

    python scripts/package.py                 # build + package
    python scripts/package.py --app-path dist/PokerHUD.app  # package existing .app

The resulting DMG includes an Applications shortcut for drag-and-drop install.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import tempfile
from pathlib import Path

from scripts.build import build_app
from scripts.build_config import BuildConfig

logger = logging.getLogger(__name__)


def create_dmg(
    app_path: Path,
    output_path: Path,
    volume_name: str = "PokerHUD",
) -> Path:
    """Create a DMG disk image containing the .app bundle.

    The DMG contains:
      * The .app bundle
      * A symlink to ``/Applications`` for drag-and-drop installation

    Args:
        app_path: Path to the built ``.app`` bundle.
        output_path: Destination path for the ``.dmg`` file.
        volume_name: Name shown when the DMG is mounted.

    Returns:
        Path to the created DMG.

    Raises:
        FileNotFoundError: If *app_path* does not exist.
        RuntimeError: If ``hdiutil`` fails.
    """
    if not app_path.exists():
        raise FileNotFoundError(f"App bundle not found: {app_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove stale DMG if present
    if output_path.exists():
        output_path.unlink()
        logger.info("Removed existing DMG at %s", output_path)

    with tempfile.TemporaryDirectory(prefix="pokerhud-dmg-") as staging:
        staging_dir = Path(staging)

        # Copy .app into staging area
        staged_app = staging_dir / app_path.name
        _copy_app(app_path, staged_app)

        # Create Applications symlink
        applications_link = staging_dir / "Applications"
        applications_link.symlink_to("/Applications")

        # Build the DMG with hdiutil
        cmd: list[str] = [
            "hdiutil",
            "create",
            "-volname",
            volume_name,
            "-srcfolder",
            str(staging_dir),
            "-ov",
            "-format",
            "UDZO",  # compressed read-only
            str(output_path),
        ]

        logger.info("Creating DMG: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("hdiutil stderr:\n%s", result.stderr)
            raise RuntimeError(
                f"hdiutil failed with exit code {result.returncode}"
            )

    logger.info("Created DMG at %s", output_path)
    return output_path


def _copy_app(src: Path, dst: Path) -> None:
    """Copy an .app bundle preserving symlinks and permissions.

    Uses ``ditto`` on macOS for a faithful copy that preserves resource
    forks, extended attributes, and symlinks.

    Args:
        src: Source .app path.
        dst: Destination .app path.

    Raises:
        RuntimeError: If the copy fails.
    """
    cmd = ["ditto", str(src), str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("ditto stderr:\n%s", result.stderr)
        raise RuntimeError(f"ditto failed with exit code {result.returncode}")


def stamp_version(config: BuildConfig) -> str:
    """Return a version string derived from the build configuration.

    The version is read from ``pyproject.toml`` via the build config.

    Args:
        config: Build configuration.

    Returns:
        The version string (e.g. ``"0.1.0"``).
    """
    return config.version


def package(
    config: BuildConfig | None = None,
    app_path: Path | None = None,
    backend: str = "pyinstaller",
) -> Path:
    """Build (if needed) and package the PokerHUD application into a DMG.

    Args:
        config: Build configuration. Uses defaults when ``None``.
        app_path: Path to an already-built .app. When ``None`` the app
            is built first using :func:`build_app`.
        backend: Build back-end (only used when building).

    Returns:
        Path to the created DMG.
    """
    if config is None:
        config = BuildConfig()

    version = stamp_version(config)
    logger.info("Packaging PokerHUD version %s", version)

    if app_path is None:
        app_path = build_app(config, backend=backend)

    dmg_path = config.dmg_path
    return create_dmg(
        app_path=app_path,
        output_path=dmg_path,
        volume_name=f"{config.app_name} {version}",
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for packaging PokerHUD into a DMG."""
    parser = argparse.ArgumentParser(
        description="Package the PokerHUD .app into a DMG installer"
    )
    parser.add_argument(
        "--app-path",
        type=Path,
        default=None,
        help="Path to an existing .app bundle (skips building)",
    )
    parser.add_argument(
        "--backend",
        choices=["pyinstaller", "py2app"],
        default="pyinstaller",
        help="Build back-end (default: pyinstaller)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    dmg_path = package(app_path=args.app_path, backend=args.backend)
    print(f"\nPackage complete: {dmg_path}")


if __name__ == "__main__":
    main()
