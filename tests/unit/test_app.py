"""Unit tests for the main application class."""

from __future__ import annotations

import pytest
from src.app import AppState, PokerHUDApp
from src.common.config import AppConfig, StatsConfig


class TestPokerHUDAppInit:
    """Tests for PokerHUDApp initialization."""

    def test_default_creation(self) -> None:
        """App can be created with default config."""
        app = PokerHUDApp(enable_overlay=False)
        assert app.state == AppState.CREATED
        assert app.config is not None
        assert app.capture_pipeline is None

    def test_custom_config(self) -> None:
        """App accepts custom configuration."""
        config = AppConfig()
        app = PokerHUDApp(config=config, enable_overlay=False, debug=True)
        assert app.config is config

    def test_initialize_transitions_state(self) -> None:
        """initialize() moves state from CREATED to INITIALIZED."""
        config = AppConfig(
            stats=StatsConfig(db_path=":memory:")
        )
        app = PokerHUDApp(config=config, enable_overlay=False)
        app.initialize()

        assert app.state == AppState.INITIALIZED
        assert app.capture_pipeline is not None
        assert app.game_state_coordinator is not None
        assert app.stats_aggregator is not None
        assert app.strategy_advisor is not None

        app.stop()

    def test_stop_transitions_to_stopped(self) -> None:
        """stop() moves state to STOPPED."""
        config = AppConfig(
            stats=StatsConfig(db_path=":memory:")
        )
        app = PokerHUDApp(config=config, enable_overlay=False)
        app.initialize()
        app.stop()
        assert app.state == AppState.STOPPED

    def test_start_requires_initialize(self) -> None:
        """start() raises RuntimeError if not initialized."""
        app = PokerHUDApp(enable_overlay=False)
        with pytest.raises(RuntimeError, match="Cannot start"):
            app.start()

    def test_double_stop_is_safe(self) -> None:
        """Calling stop() twice does not raise."""
        config = AppConfig(
            stats=StatsConfig(db_path=":memory:")
        )
        app = PokerHUDApp(config=config, enable_overlay=False)
        app.initialize()
        app.stop()
        app.stop()
        assert app.state == AppState.STOPPED


class TestPokerHUDAppConfig:
    """Tests for configuration."""

    def test_config_defaults(self) -> None:
        """Default config values are sensible."""
        config = AppConfig()
        assert config.general.debug is False
        assert config.capture.polling_interval_ms == 100
        assert config.detection.confidence_threshold == 0.85
        assert config.stats.db_path == "data/poker_hud.db"
        assert config.solver.equity_iterations == 10000
