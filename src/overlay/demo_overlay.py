"""Demo script for the transparent overlay window.

Creates an overlay window showing sample HUD statistics,
positions it in the top-right area of the screen, runs for
10 seconds, then exits.

Usage:
    python -m src.overlay.demo_overlay
"""

from __future__ import annotations

import logging

from AppKit import NSApplication, NSTimer
from PyObjCTools import AppHelper

from src.overlay.overlay_window import OverlayConfig, OverlayWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEMO_TEXT = "HUD Active \u2014 VPIP: 24% | PFR: 18% | 3-Bet: 8%"
DEMO_DURATION_SECONDS = 10.0


def _quit_app(_timer: NSTimer) -> None:
    """Timer callback to stop the application run loop."""
    logger.info("Demo duration elapsed, shutting down")
    app = NSApplication.sharedApplication()
    app.terminate_(None)


def main() -> None:
    """Run the overlay demo."""
    # Initialize the NSApplication (required for any AppKit window)
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(2)  # NSApplicationActivationPolicyAccessory

    # Compute position: top-right of main screen with some margin
    screen_w, screen_h = OverlayWindow.screen_size()
    overlay_width = 520.0
    overlay_height = 50.0
    margin = 20.0

    config = OverlayConfig(
        x=screen_w - overlay_width - margin,
        y=screen_h - overlay_height - margin,
        width=overlay_width,
        height=overlay_height,
        font_size=16.0,
        text_color=(0.0, 1.0, 0.4, 1.0),  # green
        bg_color=(0.1, 0.1, 0.1, 0.75),
    )

    overlay = OverlayWindow(config=config, text=DEMO_TEXT)
    overlay.create()
    overlay.show()

    logger.info("Overlay demo running for %.0f seconds", DEMO_DURATION_SECONDS)

    # Schedule a timer to quit the app after the demo duration
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        DEMO_DURATION_SECONDS,
        app,
        _quit_app,
        None,
        False,
    )

    # Start the run loop
    AppHelper.runEventLoop()


if __name__ == "__main__":
    main()
