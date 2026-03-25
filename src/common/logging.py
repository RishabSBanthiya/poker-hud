"""Structured logging framework for poker-hud.

Provides a consistent, structured logging setup using structlog with JSON output
for production and pretty console output for development. All loggers include
subsystem context binding automatically.

Usage:
    from src.common.logging import get_logger

    logger = get_logger("capture.screen")
    logger.info("frame_captured", width=1920, height=1080, latency_ms=12.3)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_configured = False


def configure_logging(
    *,
    level: str = "INFO",
    json_output: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure the global structured logging pipeline.

    Should be called once at application startup. Subsequent calls are no-ops
    unless the module-level ``_configured`` flag is reset (useful in tests).

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_output: If True, emit JSON lines (production). Otherwise pretty
            console output (development).
        log_file: Optional path to a log file. If provided, a file handler is
            added alongside the console handler.
    """
    global _configured
    if _configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Standard library root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)

    # Shared processors applied to every log event
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Attach structlog formatting to all stdlib handlers
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

    _configured = True


def get_logger(name: str, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Create a structured logger bound to the given component name.

    The *name* is split on ``"."`` — the first segment is treated as the
    subsystem (e.g. ``"capture"``, ``"detection"``), and the full string
    is stored as the component.

    Args:
        name: Dotted component name, e.g. ``"capture.screen_capture"``.
        **initial_context: Extra key-value pairs permanently bound to this
            logger instance.

    Returns:
        A structlog BoundLogger with subsystem/component context already set.
    """
    # Ensure logging is configured with defaults if the caller hasn't done so
    configure_logging()

    subsystem = name.split(".")[0] if "." in name else name
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger.bind(subsystem=subsystem, component=name, **initial_context)


def reset_logging() -> None:
    """Reset logging configuration so it can be reconfigured.

    Intended for use in tests only.
    """
    global _configured
    _configured = False
    structlog.reset_defaults()
    root = logging.getLogger()
    root.handlers.clear()
