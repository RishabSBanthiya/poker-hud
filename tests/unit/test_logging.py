"""Tests for the structured logging framework."""

from __future__ import annotations

import logging

import structlog
from src.common.logging import configure_logging, get_logger, reset_logging


class TestConfigureLogging:
    """Tests for configure_logging()."""

    def setup_method(self) -> None:
        reset_logging()

    def teardown_method(self) -> None:
        reset_logging()

    def test_configure_sets_root_level(self) -> None:
        configure_logging(level="DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_configure_is_idempotent(self) -> None:
        configure_logging(level="DEBUG")
        configure_logging(level="WARNING")  # Should be a no-op
        assert logging.getLogger().level == logging.DEBUG

    def test_configure_json_output(self, capsys: object) -> None:
        configure_logging(json_output=True, level="DEBUG")
        logger = get_logger("test")
        logger.info("hello")
        # If we got here without error, JSON renderer is configured

    def test_configure_console_output(self) -> None:
        configure_logging(json_output=False, level="DEBUG")
        logger = get_logger("test")
        logger.info("hello")

    def test_configure_with_log_file(self, tmp_path: object) -> None:
        import pathlib

        log_file = pathlib.Path(str(tmp_path)) / "test.log"
        configure_logging(log_file=str(log_file), level="DEBUG")
        root = logging.getLogger()
        assert len(root.handlers) == 2  # console + file


class TestGetLogger:
    """Tests for get_logger()."""

    def setup_method(self) -> None:
        reset_logging()

    def teardown_method(self) -> None:
        reset_logging()

    def test_returns_bound_logger(self) -> None:
        logger = get_logger("capture.screen")
        assert isinstance(logger, structlog.stdlib.BoundLogger)

    def test_binds_subsystem_from_dotted_name(self) -> None:
        configure_logging(json_output=True, level="DEBUG")
        logger = get_logger("capture.screen_capture")
        # Access the bound context via structlog internals
        ctx = logger._context  # type: ignore[attr-defined]
        assert ctx["subsystem"] == "capture"
        assert ctx["component"] == "capture.screen_capture"

    def test_binds_subsystem_for_simple_name(self) -> None:
        configure_logging(json_output=True, level="DEBUG")
        logger = get_logger("engine")
        ctx = logger._context  # type: ignore[attr-defined]
        assert ctx["subsystem"] == "engine"
        assert ctx["component"] == "engine"

    def test_binds_extra_context(self) -> None:
        configure_logging(json_output=True, level="DEBUG")
        logger = get_logger("stats.repo", table="hands")
        ctx = logger._context  # type: ignore[attr-defined]
        assert ctx["table"] == "hands"

    def test_auto_configures_logging(self) -> None:
        """get_logger should work even if configure_logging hasn't been called."""
        logger = get_logger("test.auto")
        logger.info("should_not_raise")


class TestResetLogging:
    """Tests for reset_logging()."""

    def test_allows_reconfiguration(self) -> None:
        configure_logging(level="DEBUG")
        reset_logging()
        configure_logging(level="WARNING")
        assert logging.getLogger().level == logging.WARNING
