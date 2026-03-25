"""Unit tests for the main application class."""

from __future__ import annotations

import pytest
from src.app import AppState, PokerHUDApp
from src.common.config import AppConfig


class TestPokerHUDAppInit:
    """Tests for PokerHUDApp initialization."""

    def test_default_creation(self) -> None:
        """App can be created with default config."""
        app = PokerHUDApp()
        assert app.state == AppState.CREATED
        assert app.config is not None
        assert app.capture_pipeline is None
        assert app.detection_pipeline is None

    def test_custom_config(self) -> None:
        """App accepts custom configuration."""
        config = AppConfig(debug=True)
        app = PokerHUDApp(config=config)
        assert app.config.debug is True

    def test_initialize_transitions_state(self) -> None:
        """initialize() moves state from CREATED to INITIALIZED."""
        config = AppConfig()
        config.stats.db_path = ":memory:"
        app = PokerHUDApp(config=config, enable_overlay=False)
        app.initialize()

        assert app.state == AppState.INITIALIZED
        assert app.capture_pipeline is not None
        assert app.detection_pipeline is not None
        assert app.game_state_coordinator is not None
        assert app.stats_aggregator is not None
        assert app.strategy_advisor is not None

        app.stop()

    def test_stop_transitions_to_stopped(self) -> None:
        """stop() moves state to STOPPED."""
        config = AppConfig()
        config.stats.db_path = ":memory:"
        app = PokerHUDApp(config=config, enable_overlay=False)
        app.initialize()
        app.stop()
        assert app.state == AppState.STOPPED

    def test_start_requires_initialize(self) -> None:
        """start() raises RuntimeError if not initialized."""
        app = PokerHUDApp()
        with pytest.raises(RuntimeError, match="Cannot start"):
            app.start()


class TestPokerHUDAppConfig:
    """Tests for configuration loading."""

    def test_config_from_file(self, tmp_path) -> None:
        """AppConfig can be loaded from a JSON file."""
        config_file = tmp_path / "config.json"
        config_file.write_text(
            '{"debug": true, "stats": {"db_path": ":memory:"}}'
        )
        config = AppConfig.from_file(config_file)
        assert config.debug is True
        assert config.stats.db_path == ":memory:"

    def test_config_file_not_found(self) -> None:
        """Missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            AppConfig.from_file("/nonexistent/config.json")

    def test_config_defaults(self) -> None:
        """Default config values are sensible."""
        config = AppConfig()
        assert config.debug is False
        assert config.capture.poll_interval_ms == 200
        assert config.detection.confidence_threshold == 0.8
        assert config.stats.min_hands_for_stats == 5
        assert config.solver.equity_simulations == 10000
        assert config.overlay.enabled is True
