"""Tests for the configuration management system."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from src.common.config import (
    AppConfig,
    CaptureConfig,
    DetectionConfig,
    GeneralConfig,
    OverlayConfig,
    SolverConfig,
    StatsConfig,
    load_config,
)


class TestDefaultConfig:
    """Tests for default configuration values."""

    def test_default_config_creates_all_sections(self) -> None:
        config = AppConfig()
        assert isinstance(config.general, GeneralConfig)
        assert isinstance(config.capture, CaptureConfig)
        assert isinstance(config.detection, DetectionConfig)
        assert isinstance(config.overlay, OverlayConfig)
        assert isinstance(config.stats, StatsConfig)
        assert isinstance(config.solver, SolverConfig)

    def test_default_capture_fps(self) -> None:
        config = AppConfig()
        assert config.capture.fps == 10

    def test_default_detection_threshold(self) -> None:
        config = AppConfig()
        assert config.detection.confidence_threshold == 0.85

    def test_default_log_level(self) -> None:
        config = AppConfig()
        assert config.general.log_level == "INFO"

    def test_default_debug_false(self) -> None:
        config = AppConfig()
        assert config.general.debug is False

    def test_default_overlay_colors(self) -> None:
        config = AppConfig()
        assert config.overlay.background_color == "#1a1a2e"
        assert config.overlay.accent_color == "#00d4aa"

    def test_default_stats_wal_mode(self) -> None:
        config = AppConfig()
        assert config.stats.wal_mode is True

    def test_configs_are_frozen(self) -> None:
        config = AppConfig()
        with pytest.raises(AttributeError):
            config.general = GeneralConfig(log_level="DEBUG")  # type: ignore[misc]


class TestLoadConfigFromYAML:
    """Tests for loading configuration from YAML files."""

    def test_load_yaml_overrides(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            capture:
              fps: 30
            general:
              log_level: DEBUG
              debug: true
        """)
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(yaml_content)

        config = load_config(config_file)
        assert config.capture.fps == 30
        assert config.general.log_level == "DEBUG"
        assert config.general.debug is True
        # Non-overridden values remain default
        assert config.detection.confidence_threshold == 0.85

    def test_load_yml_extension(self, tmp_path: Path) -> None:
        config_file = tmp_path / "settings.yml"
        config_file.write_text("capture:\n  fps: 25\n")
        config = load_config(config_file)
        assert config.capture.fps == 25

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        config = load_config(config_file)
        # All defaults preserved
        assert config.capture.fps == 10

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")


class TestLoadConfigFromTOML:
    """Tests for loading configuration from TOML files."""

    def test_load_toml_overrides(self, tmp_path: Path) -> None:
        toml_content = textwrap.dedent("""\
            [capture]
            fps = 60

            [overlay]
            opacity = 0.7
        """)
        config_file = tmp_path / "settings.toml"
        config_file.write_text(toml_content)

        config = load_config(config_file)
        assert config.capture.fps == 60
        assert config.overlay.opacity == 0.7

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "settings.ini"
        config_file.write_text("[section]\nkey=value\n")
        with pytest.raises(ValueError, match="Unsupported config file format"):
            load_config(config_file)


class TestLoadConfigFromEnv:
    """Tests for environment variable overrides."""

    def test_env_overrides_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POKERHUD_CAPTURE__FPS", "45")
        config = load_config()
        assert config.capture.fps == 45

    def test_env_overrides_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("capture:\n  fps: 30\n")
        monkeypatch.setenv("POKERHUD_CAPTURE__FPS", "60")

        config = load_config(config_file)
        assert config.capture.fps == 60  # Env wins over file

    def test_env_bool_coercion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POKERHUD_GENERAL__DEBUG", "true")
        config = load_config()
        assert config.general.debug is True

    def test_env_float_coercion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POKERHUD_DETECTION__CONFIDENCE_THRESHOLD", "0.95")
        config = load_config()
        assert config.detection.confidence_threshold == 0.95

    def test_env_single_segment_maps_to_general(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("POKERHUD_LOG_LEVEL", "WARNING")
        config = load_config()
        assert config.general.log_level == "WARNING"

    def test_unknown_env_keys_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POKERHUD_CAPTURE__NONEXISTENT", "value")
        config = load_config()  # Should not raise
        assert config.capture.fps == 10  # Defaults still work
