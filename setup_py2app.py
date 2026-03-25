"""py2app setup script for building the PokerHUD macOS .app bundle.

Usage::

    python setup_py2app.py py2app

This is used by ``scripts/build.py --backend py2app`` and can also be
invoked directly.
"""

from __future__ import annotations

from scripts.build_config import BuildConfig
from setuptools import setup

config = BuildConfig()

APP = [str(config.entry_point)]

OPTIONS = {
    "argv_emulation": False,
    "plist": config.plist_dict,
    "includes": [
        "src",
        "src.capture",
        "src.detection",
        "src.engine",
        "src.overlay",
        "src.solver",
        "src.stats",
        "numpy",
        "cv2",
        "structlog",
    ],
    "excludes": config.exclude_packages,
    "iconfile": str(config.icon_path) if config.icon_path.is_file() else None,
    "site_packages": True,
    "strip": True,
    "semi_standalone": False,  # bundle Python runtime
}

# Remove None values from OPTIONS
OPTIONS = {k: v for k, v in OPTIONS.items() if v is not None}

setup(
    name=config.app_name,
    version=config.version,
    app=APP,
    data_files=config.data_files,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
