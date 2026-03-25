"""Unit tests for the build configuration module."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.build_config import (
    REPO_ROOT,
    BuildConfig,
    _read_version_from_pyproject,
)


class TestReadVersion:
    """Tests for reading the version from pyproject.toml."""

    def test_reads_version_string(self) -> None:
        """Version is a non-empty string from pyproject.toml."""
        version = _read_version_from_pyproject()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_version_is_semver_like(self) -> None:
        """Version looks like a semantic version (X.Y.Z)."""
        version = _read_version_from_pyproject()
        parts = version.split(".")
        assert len(parts) >= 2, f"Expected semver-like version, got {version!r}"
        for part in parts:
            assert part.isdigit(), f"Version part {part!r} is not numeric"


class TestBuildConfigDefaults:
    """Tests for default BuildConfig values."""

    def test_default_app_name(self) -> None:
        """Default app name is PokerHUD."""
        config = BuildConfig()
        assert config.app_name == "PokerHUD"

    def test_default_bundle_identifier(self) -> None:
        """Default bundle identifier is com.pokerhud.app."""
        config = BuildConfig()
        assert config.bundle_identifier == "com.pokerhud.app"

    def test_version_auto_populated(self) -> None:
        """Version is auto-populated from pyproject.toml."""
        config = BuildConfig()
        assert config.version
        assert isinstance(config.version, str)

    def test_version_matches_pyproject(self) -> None:
        """Auto-populated version matches pyproject.toml."""
        config = BuildConfig()
        expected = _read_version_from_pyproject()
        assert config.version == expected

    def test_explicit_version_overrides(self) -> None:
        """An explicitly provided version is used instead of pyproject."""
        config = BuildConfig(version="9.9.9")
        assert config.version == "9.9.9"

    def test_entry_point_is_main_py(self) -> None:
        """Entry point defaults to main.py at the repo root."""
        config = BuildConfig()
        assert config.entry_point.name == "main.py"
        assert config.entry_point.parent == REPO_ROOT

    def test_default_codesign_identity_is_adhoc(self) -> None:
        """Default code signing is ad-hoc (dash)."""
        config = BuildConfig()
        assert config.codesign_identity == "-"

    def test_bundle_python_runtime_default(self) -> None:
        """Python runtime is bundled by default."""
        config = BuildConfig()
        assert config.bundle_python_runtime is True

    def test_ls_ui_element_default(self) -> None:
        """LSUIElement defaults to True (menu bar app, no dock icon)."""
        config = BuildConfig()
        assert config.ls_ui_element is True

    def test_paths_are_absolute(self) -> None:
        """All path fields are absolute."""
        config = BuildConfig()
        assert config.entry_point.is_absolute()
        assert config.src_dir.is_absolute()
        assert config.dist_dir.is_absolute()
        assert config.build_dir.is_absolute()
        assert config.plist_template.is_absolute()
        assert config.icon_path.is_absolute()


class TestBuildConfigPlist:
    """Tests for the plist_dict property."""

    def test_plist_has_required_keys(self) -> None:
        """plist_dict contains all required macOS bundle keys."""
        config = BuildConfig()
        plist = config.plist_dict

        required_keys = [
            "CFBundleName",
            "CFBundleDisplayName",
            "CFBundleIdentifier",
            "CFBundleVersion",
            "CFBundleShortVersionString",
            "CFBundlePackageType",
            "CFBundleExecutable",
            "LSUIElement",
            "NSScreenCaptureUsageDescription",
        ]
        for key in required_keys:
            assert key in plist, f"Missing required key: {key}"

    def test_plist_screen_capture_description(self) -> None:
        """plist includes a meaningful Screen Recording permission string."""
        config = BuildConfig()
        desc = config.plist_dict["NSScreenCaptureUsageDescription"]
        assert isinstance(desc, str)
        assert "Screen Recording" in desc or "screen" in desc.lower()

    def test_plist_ls_ui_element_true(self) -> None:
        """LSUIElement is True so the app has no dock icon."""
        config = BuildConfig()
        assert config.plist_dict["LSUIElement"] is True

    def test_plist_version_matches_config(self) -> None:
        """plist version matches the config version."""
        config = BuildConfig(version="1.2.3")
        plist = config.plist_dict
        assert plist["CFBundleVersion"] == "1.2.3"
        assert plist["CFBundleShortVersionString"] == "1.2.3"

    def test_plist_bundle_name_matches(self) -> None:
        """plist bundle name matches app_name."""
        config = BuildConfig(app_name="TestApp")
        assert config.plist_dict["CFBundleName"] == "TestApp"


class TestBuildConfigPaths:
    """Tests for computed path properties."""

    def test_app_path(self) -> None:
        """app_path points to the .app inside dist/."""
        config = BuildConfig()
        assert config.app_path == config.dist_dir / "PokerHUD.app"

    def test_dmg_path_includes_version(self) -> None:
        """dmg_path includes the version number."""
        config = BuildConfig(version="2.0.0")
        assert config.dmg_path == config.dist_dir / "PokerHUD-2.0.0.dmg"

    def test_custom_app_name_in_paths(self) -> None:
        """Custom app_name is reflected in app_path and dmg_path."""
        config = BuildConfig(app_name="MyPoker", version="1.0.0")
        assert config.app_path.name == "MyPoker.app"
        assert "MyPoker-1.0.0.dmg" == config.dmg_path.name


class TestBuildConfigValidation:
    """Tests for the validate() method."""

    def test_default_config_is_valid(self) -> None:
        """Default configuration passes validation."""
        config = BuildConfig()
        errors = config.validate()
        assert errors == []

    def test_empty_app_name_fails(self) -> None:
        """Empty app_name is rejected."""
        config = BuildConfig(app_name="")
        errors = config.validate()
        assert any("app_name" in e for e in errors)

    def test_empty_bundle_id_fails(self) -> None:
        """Empty bundle_identifier is rejected."""
        config = BuildConfig(bundle_identifier="")
        errors = config.validate()
        assert any("bundle_identifier" in e for e in errors)

    def test_bad_entry_point_extension_fails(self) -> None:
        """Entry point without .py extension is rejected."""
        config = BuildConfig(entry_point=Path("/tmp/main.sh"))
        errors = config.validate()
        assert any("entry_point" in e for e in errors)

    def test_valid_custom_config(self) -> None:
        """A fully custom but valid config passes validation."""
        config = BuildConfig(
            app_name="CustomHUD",
            bundle_identifier="com.example.custom",
            version="3.1.4",
            entry_point=Path("/tmp/app.py"),
        )
        errors = config.validate()
        assert errors == []


class TestBuildConfigExclusions:
    """Tests for exclusion lists."""

    def test_tests_excluded_by_default(self) -> None:
        """Test packages are excluded from the bundle."""
        config = BuildConfig()
        assert "tests" in config.exclude_packages

    def test_git_excluded_by_default(self) -> None:
        """.git is excluded from the bundle."""
        config = BuildConfig()
        assert ".git" in config.exclude_patterns

    def test_pycache_excluded(self) -> None:
        """__pycache__ is excluded from the bundle."""
        config = BuildConfig()
        assert "__pycache__" in config.exclude_patterns


class TestBuildConfigFrozen:
    """Tests that BuildConfig is immutable."""

    def test_cannot_set_attribute(self) -> None:
        """BuildConfig fields cannot be reassigned."""
        config = BuildConfig()
        with pytest.raises(AttributeError):
            config.app_name = "Changed"  # type: ignore[misc]
