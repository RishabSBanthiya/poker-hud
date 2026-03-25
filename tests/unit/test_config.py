"""Unit tests for application configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.common.config import AppConfig, OverlayConfig, StatsConfig


class TestOverlayConfig:
    """Tests for OverlayConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = OverlayConfig()
        assert cfg.opacity == 0.8
        assert cfg.font_size == 14.0
        assert cfg.compact_mode is True
        assert cfg.show_stats_panel is True
        assert cfg.show_solver_panel is True
        assert cfg.show_settings_panel is False


class TestStatsConfig:
    """Tests for StatsConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = StatsConfig()
        assert cfg.vpip_loose_threshold == 30.0
        assert cfg.vpip_tight_threshold == 15.0


class TestAppConfig:
    """Tests for AppConfig save/load."""

    def test_defaults(self) -> None:
        cfg = AppConfig()
        assert cfg.poker_client == "PokerStars"
        assert cfg.table_size == 9

    def test_save_and_load(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        original = AppConfig(
            poker_client="888poker",
            table_size=6,
            overlay=OverlayConfig(opacity=0.5, font_size=16.0),
            stats=StatsConfig(vpip_loose_threshold=35.0),
        )
        original.save(path)

        loaded = AppConfig.load(path)
        assert loaded.poker_client == "888poker"
        assert loaded.table_size == 6
        assert loaded.overlay.opacity == 0.5
        assert loaded.stats.vpip_loose_threshold == 35.0

    def test_save_creates_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "config.json"
        AppConfig().save(path)
        assert path.exists()

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            AppConfig.load(tmp_path / "nope.json")

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{invalid")
        with pytest.raises(json.JSONDecodeError):
            AppConfig.load(path)
