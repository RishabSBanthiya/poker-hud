"""Unit tests for the SettingsPanel."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.common.config import AppConfig, OverlayConfig, StatsConfig
from src.overlay.settings_panel import (
    SUPPORTED_CLIENTS,
    SettingsPanel,
)

# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestSettingsPanelInit:
    """Tests for SettingsPanel initialization."""

    def test_default_init(self) -> None:
        panel = SettingsPanel()
        assert panel.visible is False
        assert panel.is_dirty is False
        assert panel.get_poker_client() == "PokerStars"

    def test_custom_config_init(self) -> None:
        config = AppConfig(
            overlay=OverlayConfig(opacity=0.5),
            stats=StatsConfig(db_path="/tmp/test.db"),
        )
        panel = SettingsPanel(config=config)
        assert panel.get_config().overlay.opacity == 0.5
        assert panel.get_config().stats.db_path == "/tmp/test.db"

    def test_custom_config_path(self) -> None:
        panel = SettingsPanel(config_path="/tmp/custom.json")
        assert panel.state.config_path == "/tmp/custom.json"


# ---------------------------------------------------------------------------
# Overlay settings tests
# ---------------------------------------------------------------------------


class TestOverlaySettings:
    """Tests for overlay-related settings."""

    def setup_method(self) -> None:
        self.panel = SettingsPanel()

    def test_set_opacity(self) -> None:
        self.panel.set_opacity(0.5)
        assert self.panel.get_config().overlay.opacity == 0.5
        assert self.panel.is_dirty is True

    def test_set_opacity_invalid_high(self) -> None:
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            self.panel.set_opacity(1.5)

    def test_set_opacity_invalid_low(self) -> None:
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            self.panel.set_opacity(-0.1)

    def test_set_font_size(self) -> None:
        self.panel.set_font_size(16.0)
        assert self.panel.get_config().overlay.font_size == 16
        assert self.panel.is_dirty is True

    def test_set_font_size_invalid(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            self.panel.set_font_size(0)

    def test_set_compact_mode(self) -> None:
        self.panel.set_compact_mode(False)
        assert self.panel._state.settings.compact_mode is False
        assert self.panel.is_dirty is True

    def test_set_overlay_visible(self) -> None:
        self.panel.set_overlay_visible(False)
        s = self.panel._state.settings
        assert s.show_stats_panel is False
        assert s.show_solver_panel is False
        assert self.panel.is_dirty is True

    def test_visibility_toggle(self) -> None:
        self.panel.visible = True
        assert self.panel.visible is True
        self.panel.visible = False
        assert self.panel.visible is False


# ---------------------------------------------------------------------------
# Poker client settings tests
# ---------------------------------------------------------------------------


class TestPokerClientSettings:
    """Tests for poker client selection."""

    def setup_method(self) -> None:
        self.panel = SettingsPanel()

    def test_set_valid_client(self) -> None:
        self.panel.set_poker_client("888poker")
        assert self.panel.get_poker_client() == "888poker"
        assert self.panel.is_dirty is True

    def test_set_invalid_client(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            self.panel.set_poker_client("UnknownPoker")

    def test_all_supported_clients(self) -> None:
        for client in SUPPORTED_CLIENTS:
            panel = SettingsPanel()
            panel.set_poker_client(client)
            assert panel.get_poker_client() == client


# ---------------------------------------------------------------------------
# Database settings tests
# ---------------------------------------------------------------------------


class TestDatabaseSettings:
    """Tests for database path configuration."""

    def setup_method(self) -> None:
        self.panel = SettingsPanel()

    def test_set_db_path(self) -> None:
        self.panel.set_db_path("/tmp/test.db")
        assert self.panel.get_db_path() == "/tmp/test.db"
        assert self.panel.is_dirty is True

    def test_set_empty_db_path(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            self.panel.set_db_path("")

    def test_set_whitespace_db_path(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            self.panel.set_db_path("   ")


# ---------------------------------------------------------------------------
# Table settings tests
# ---------------------------------------------------------------------------


class TestTableSettings:
    """Tests for table size configuration."""

    def setup_method(self) -> None:
        self.panel = SettingsPanel()

    def test_set_table_size(self) -> None:
        self.panel.set_table_size(6)
        assert self.panel._state.settings.table_size == 6
        assert self.panel.is_dirty is True

    def test_set_table_size_too_small(self) -> None:
        with pytest.raises(ValueError, match="2-10"):
            self.panel.set_table_size(1)

    def test_set_table_size_too_large(self) -> None:
        with pytest.raises(ValueError, match="2-10"):
            self.panel.set_table_size(11)

    def test_set_stats_config(self) -> None:
        cfg = StatsConfig(vpip_loose_threshold=45.0)
        self.panel.set_stats_config(cfg)
        assert (
            self.panel.get_config().stats.vpip_loose_threshold == 45.0
        )
        assert self.panel.is_dirty is True


# ---------------------------------------------------------------------------
# Apply / reset tests
# ---------------------------------------------------------------------------


class TestApplyAndReset:
    """Tests for applying and resetting configurations."""

    def test_apply_config(self) -> None:
        panel = SettingsPanel()
        new_config = AppConfig(
            overlay=OverlayConfig(opacity=0.7, font_size=18),
        )
        panel.apply_config(new_config)

        assert panel.get_config().overlay.opacity == 0.7
        assert panel.get_config().overlay.font_size == 18
        assert panel.is_dirty is True

    def test_reset_to_defaults(self) -> None:
        panel = SettingsPanel()
        panel.set_poker_client("888poker")
        panel.set_table_size(6)
        panel.reset_to_defaults()

        assert panel.get_poker_client() == "PokerStars"
        assert panel._state.settings.table_size == 9
        assert panel.is_dirty is True  # reset is a change


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestPersistence:
    """Tests for save/load configuration."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        config_file = tmp_path / "test_config.json"

        # Save
        panel = SettingsPanel(config_path=str(config_file))
        panel.set_poker_client("888poker")
        panel.set_opacity(0.6)
        panel.set_table_size(6)
        panel.save_config()

        assert panel.is_dirty is False
        assert config_file.exists()

        # Load
        panel2 = SettingsPanel(config_path=str(config_file))
        loaded = panel2.load_config()

        assert panel2.get_poker_client() == "888poker"
        assert loaded.overlay.opacity == 0.6
        assert panel2._state.settings.table_size == 6
        assert panel2.is_dirty is False

    def test_save_with_override_path(self, tmp_path: Path) -> None:
        config_file = tmp_path / "override.json"
        panel = SettingsPanel()
        panel.save_config(str(config_file))
        assert config_file.exists()

    def test_load_nonexistent_file(self) -> None:
        panel = SettingsPanel(config_path="/nonexistent/path.json")
        with pytest.raises(FileNotFoundError):
            panel.load_config()

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        config_file = tmp_path / "bad.json"
        config_file.write_text("not json {{{")
        panel = SettingsPanel(config_path=str(config_file))
        with pytest.raises(json.JSONDecodeError):
            panel.load_config()

    def test_saved_json_structure(self, tmp_path: Path) -> None:
        config_file = tmp_path / "struct.json"
        panel = SettingsPanel()
        panel.save_config(str(config_file))

        data = json.loads(config_file.read_text())
        assert "overlay" in data
        assert "stats" in data
        assert "poker_client" in data
        assert "db_path" in data
        assert "table_size" in data


# ---------------------------------------------------------------------------
# Settings summary tests
# ---------------------------------------------------------------------------


class TestSettingsSummary:
    """Tests for the human-readable settings summary."""

    def test_summary_contains_key_info(self) -> None:
        panel = SettingsPanel()
        summary = panel.get_settings_summary()

        assert "PokerStars" in summary
        assert "Table Size: 9" in summary
        assert "Opacity:" in summary
        assert "Font Size:" in summary
        assert "DB Path:" in summary

    def test_summary_reflects_changes(self) -> None:
        panel = SettingsPanel()
        panel.set_poker_client("GGPoker")
        panel.set_table_size(6)

        summary = panel.get_settings_summary()
        assert "GGPoker" in summary
        assert "Table Size: 6" in summary
