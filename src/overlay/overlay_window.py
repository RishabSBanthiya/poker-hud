"""Transparent overlay window for macOS using PyObjC/AppKit.

Provides an always-on-top, click-through, borderless transparent window
suitable for rendering HUD statistics over a poker client.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from AppKit import (
    NSApplication,
    NSBackingStoreBuffered,
    NSColor,
    NSFont,
    NSMakeRect,
    NSScreen,
    NSTextField,
    NSWindow,
    NSWindowStyleMaskBorderless,
)

logger = logging.getLogger(__name__)

# Window level constant — NSFloatingWindowLevel is 3, but we use
# NSStatusWindowLevel (25) to stay above most other floating panels.
_STATUS_WINDOW_LEVEL = 25


@dataclass
class OverlayConfig:
    """Configuration for the overlay window.

    Attributes:
        x: Horizontal position from left edge of screen.
        y: Vertical position from bottom edge of screen (AppKit coordinates).
        width: Window width in points.
        height: Window height in points.
        font_size: Font size for the HUD text.
        text_color: RGBA tuple for text color, each component 0.0-1.0.
        bg_color: RGBA tuple for background color, each component 0.0-1.0.
    """

    x: float = 0.0
    y: float = 0.0
    width: float = 500.0
    height: float = 60.0
    font_size: float = 18.0
    text_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    bg_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.6)


class OverlayWindow:
    """A transparent, always-on-top, click-through macOS overlay window.

    This window floats above all other windows, ignores mouse events
    (click-through), and renders text with a semi-transparent background.
    It is intended for displaying HUD statistics over a poker client.

    Args:
        config: Configuration for window geometry, colors, and font.
        text: Initial text to display in the overlay.
    """

    def __init__(
        self,
        config: OverlayConfig | None = None,
        text: str = "",
    ) -> None:
        self._config = config or OverlayConfig()
        self._text = text
        self._window: NSWindow | None = None
        self._text_field: NSTextField | None = None

    @property
    def config(self) -> OverlayConfig:
        """Return the current overlay configuration."""
        return self._config

    @property
    def text(self) -> str:
        """Return the current display text."""
        return self._text

    def create(self) -> None:
        """Create and configure the native macOS overlay window.

        Sets up a borderless, transparent, always-on-top window with
        click-through behavior and a text label for HUD content.
        """
        cfg = self._config
        rect = NSMakeRect(cfg.x, cfg.y, cfg.width, cfg.height)

        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect,
            NSWindowStyleMaskBorderless,
            NSBackingStoreBuffered,
            False,
        )

        # Transparency and visual setup
        self._window.setOpaque_(False)
        self._window.setHasShadow_(False)
        self._window.setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                cfg.bg_color[0],
                cfg.bg_color[1],
                cfg.bg_color[2],
                cfg.bg_color[3],
            )
        )

        # Always on top
        self._window.setLevel_(_STATUS_WINDOW_LEVEL)

        # Click-through — mouse events pass to windows underneath
        self._window.setIgnoresMouseEvents_(True)

        # Appear on all spaces / desktops
        self._window.setCollectionBehavior_(1 << 0)  # NSWindowCollectionBehaviorCanJoinAllSpaces

        # Create the text label
        content_rect = NSMakeRect(10, 5, cfg.width - 20, cfg.height - 10)
        self._text_field = NSTextField.alloc().initWithFrame_(content_rect)
        self._text_field.setStringValue_(self._text)
        self._text_field.setEditable_(False)
        self._text_field.setSelectable_(False)
        self._text_field.setBezeled_(False)
        self._text_field.setDrawsBackground_(False)
        self._text_field.setFont_(
            NSFont.monospacedSystemFontOfSize_weight_(cfg.font_size, 0.0)
        )
        self._text_field.setTextColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                cfg.text_color[0],
                cfg.text_color[1],
                cfg.text_color[2],
                cfg.text_color[3],
            )
        )

        self._window.contentView().addSubview_(self._text_field)

        logger.info(
            "Overlay window created at (%.0f, %.0f) size %.0fx%.0f",
            cfg.x,
            cfg.y,
            cfg.width,
            cfg.height,
        )

    def show(self) -> None:
        """Show the overlay window.

        Raises:
            RuntimeError: If the window has not been created yet.
        """
        if self._window is None:
            raise RuntimeError("Window not created. Call create() first.")
        self._window.orderFrontRegardless()
        logger.info("Overlay window shown")

    def hide(self) -> None:
        """Hide the overlay window."""
        if self._window is not None:
            self._window.orderOut_(None)
            logger.info("Overlay window hidden")

    def close(self) -> None:
        """Close and release the overlay window."""
        if self._window is not None:
            self._window.close()
            self._window = None
            self._text_field = None
            logger.info("Overlay window closed")

    def set_text(self, text: str) -> None:
        """Update the displayed text.

        Args:
            text: New text content for the HUD overlay.
        """
        self._text = text
        if self._text_field is not None:
            self._text_field.setStringValue_(text)

    def set_position(self, x: float, y: float) -> None:
        """Move the overlay window to a new screen position.

        Args:
            x: Horizontal position from left edge of screen.
            y: Vertical position from bottom edge of screen.
        """
        self._config.x = x
        self._config.y = y
        if self._window is not None:
            origin = self._window.frame().origin
            origin.x = x
            origin.y = y
            self._window.setFrameOrigin_(origin)

    def set_size(self, width: float, height: float) -> None:
        """Resize the overlay window.

        Args:
            width: New window width in points.
            height: New window height in points.
        """
        self._config.width = width
        self._config.height = height
        if self._window is not None:
            frame = self._window.frame()
            frame.size.width = width
            frame.size.height = height
            self._window.setFrame_display_(frame, True)

    @property
    def is_visible(self) -> bool:
        """Return whether the overlay window is currently visible."""
        if self._window is not None:
            return bool(self._window.isVisible())
        return False

    @staticmethod
    def screen_size() -> tuple[float, float]:
        """Return the main screen size as (width, height).

        Returns:
            Tuple of (width, height) in points.
        """
        screen = NSScreen.mainScreen()
        frame = screen.frame()
        return (frame.size.width, frame.size.height)
