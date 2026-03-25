"""Entry point for the Poker HUD application.

Usage:
    python main.py [--config CONFIG_PATH] [--debug] [--no-overlay]
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.app import PokerHUDApp
from src.common.config import AppConfig

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Poker HUD — real-time poker overlay with GTO advice",
        prog="poker-hud",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        default=False,
        help="Run without the overlay window (headless mode)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Poker HUD application.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success).
    """
    args = parse_args(argv)

    # Load configuration
    if args.config:
        try:
            config = AppConfig.from_file(args.config)
        except FileNotFoundError:
            print(
                f"Error: Configuration file not found: {args.config}",
                file=sys.stderr,
            )
            return 1
        except Exception as exc:
            print(
                f"Error: Failed to load configuration: {exc}",
                file=sys.stderr,
            )
            return 1
    else:
        config = AppConfig()

    if args.debug:
        config.debug = True

    enable_overlay = not args.no_overlay

    # Create and run the application
    app = PokerHUDApp(config=config, enable_overlay=enable_overlay)

    try:
        app.initialize()
        app.install_signal_handlers()
        app.start()

        print("Poker HUD is running. Press Ctrl+C to stop.")
        app.wait_for_shutdown()

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        return 1
    finally:
        app.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
