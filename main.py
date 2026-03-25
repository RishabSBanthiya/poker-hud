"""Entry point for the Poker HUD application.

Usage:
    python main.py [--config CONFIG_PATH] [--debug] [--no-overlay]
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from src.app import PokerHUDApp
from src.common.config import AppConfig, load_config

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
        help="Path to YAML/TOML configuration file",
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

    When the overlay is enabled, runs the AppKit event loop on the main
    thread (required by macOS for window rendering). In headless mode,
    blocks on a threading event instead.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success).
    """
    args = parse_args(argv)

    # Load configuration
    if args.config:
        try:
            config = load_config(args.config)
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

    enable_overlay = not args.no_overlay
    debug = args.debug

    # Create and run the application
    app = PokerHUDApp(config=config, enable_overlay=enable_overlay, debug=debug)

    try:
        if enable_overlay:
            # AppKit requires NSApplication on the main thread.
            from AppKit import NSApplication
            from PyObjCTools import AppHelper

            ns_app = NSApplication.sharedApplication()
            # Regular policy so overlay windows render on screen.
            # Accessory (2) prevents window display.
            ns_app.setActivationPolicy_(0)
            ns_app.activateIgnoringOtherApps_(True)

            app.initialize()
            app.start()

            # Install signal handlers that stop AppKit run loop
            def _signal_handler(signum: int, _frame: object) -> None:
                sig_name = signal.Signals(signum).name
                logger.info("Received %s, shutting down...", sig_name)
                app.stop()
                AppHelper.stopEventLoop()

            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)

            print("Poker HUD is running. Press Ctrl+C to stop.")
            AppHelper.runEventLoop()
        else:
            # Headless mode — no overlay, no AppKit
            app.initialize()
            app.install_signal_handlers()
            app.start()

            print("Poker HUD is running (headless). Press Ctrl+C to stop.")
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
