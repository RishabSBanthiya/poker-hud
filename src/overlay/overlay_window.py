"""Transparent overlay window for macOS using PyObjC/AppKit.

Provides an always-on-top, click-through, borderless transparent window
suitable for rendering HUD statistics over a poker client. Supports
multiple panels (stats, solver, settings) and dynamic positioning
relative to a poker table window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from AppKit import (
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

# Window level constant — NSStatusWindowLevel (25) to stay above most panels.
_STATUS_WINDOW_LEVEL = 25


class PanelType(Enum):
    """Types of overlay panels."""

    STATS = "stats"
    SOLVER = "solver"
    SETTINGS = "settings"


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
        opacity: Overall window opacity 0.0-1.0.
    """

    x: float = 0.0
    y: float = 0.0
    width: float = 500.0
    height: float = 60.0
    font_size: float = 18.0
    text_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    bg_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.6)
    opacity: float = 1.0


@dataclass
class WindowInfo:
    """Information about a poker client window for overlay positioning.

    Attributes:
        x: Window x position on screen.
        y: Window y position on screen.
        width: Window width in points.
        height: Window height in points.
        title: Window title for identification.
        window_id: macOS window identifier.
    """

    x: float = 0.0
    y: float = 0.0
    width: float = 800.0
    height: float = 600.0
    title: str = ""
    window_id: int = 0


@dataclass
class PanelState:
    """State of an individual overlay panel.

    Attributes:
        panel_type: The type of this panel.
        visible: Whether the panel is currently shown.
        x_offset: Horizontal offset relative to the overlay origin.
        y_offset: Vertical offset relative to the overlay origin.
        width: Panel width in points.
        height: Panel height in points.
        content: Current text content of the panel.
    """

    panel_type: PanelType = PanelType.STATS
    visible: bool = True
    x_offset: float = 0.0
    y_offset: float = 0.0
    width: float = 300.0
    height: float = 50.0
    content: str = ""


class OverlayWindow:
    """A transparent, always-on-top, click-through macOS overlay window.

    Supports multiple panels (stats, solver, settings) and dynamic
    positioning relative to a poker client window.

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
        self._attached_window: WindowInfo | None = None
        self._panels: dict[PanelType, PanelState] = {}
        self._panel_text_fields: dict[PanelType, Any] = {}

        # Initialize default panel states
        self._panels[PanelType.STATS] = PanelState(
            panel_type=PanelType.STATS,
            visible=True,
            x_offset=10.0,
            y_offset=10.0,
            width=280.0,
            height=40.0,
        )
        self._panels[PanelType.SOLVER] = PanelState(
            panel_type=PanelType.SOLVER,
            visible=True,
            x_offset=300.0,
            y_offset=10.0,
            width=200.0,
            height=40.0,
        )
        self._panels[PanelType.SETTINGS] = PanelState(
            panel_type=PanelType.SETTINGS,
            visible=False,
            x_offset=10.0,
            y_offset=60.0,
            width=480.0,
            height=200.0,
        )

    @property
    def config(self) -> OverlayConfig:
        """Return the current overlay configuration."""
        return self._config

    @property
    def text(self) -> str:
        """Return the current display text."""
        return self._text

    @property
    def attached_window(self) -> WindowInfo | None:
        """Return the currently attached poker window info."""
        return self._attached_window

    @property
    def panels(self) -> dict[PanelType, PanelState]:
        """Return the current panel states."""
        return self._panels

    def create(self) -> None:
        """Create and configure the native macOS overlay window.

        Sets up a borderless, transparent, always-on-top window with
        click-through behavior and text labels for HUD content.
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
        self._window.setCollectionBehavior_(1 << 0)

        # Apply initial opacity
        self._window.setAlphaValue_(cfg.opacity)

        # Create the main text label
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

        # Create panel text fields
        for panel_type, state in self._panels.items():
            self._create_panel_text_field(panel_type, state)

        logger.info(
            "Overlay window created at (%.0f, %.0f) size %.0fx%.0f",
            cfg.x,
            cfg.y,
            cfg.width,
            cfg.height,
        )

    def _create_panel_text_field(
        self, panel_type: PanelType, state: PanelState
    ) -> None:
        """Create a text field for a panel and add it to the window.

        Args:
            panel_type: The type of panel to create.
            state: The panel's state with position and size info.
        """
        if self._window is None:
            return

        rect = NSMakeRect(
            state.x_offset, state.y_offset, state.width, state.height
        )
        tf = NSTextField.alloc().initWithFrame_(rect)
        tf.setStringValue_(state.content)
        tf.setEditable_(False)
        tf.setSelectable_(False)
        tf.setBezeled_(False)
        tf.setDrawsBackground_(False)
        tf.setFont_(
            NSFont.monospacedSystemFontOfSize_weight_(
                self._config.font_size, 0.0
            )
        )
        tf.setTextColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                self._config.text_color[0],
                self._config.text_color[1],
                self._config.text_color[2],
                self._config.text_color[3],
            )
        )
        tf.setHidden_(not state.visible)

        self._window.contentView().addSubview_(tf)
        self._panel_text_fields[panel_type] = tf

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
            self._panel_text_fields.clear()
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

    def set_opacity(self, opacity: float) -> None:
        """Set the overlay window opacity.

        Args:
            opacity: Opacity value from 0.0 (invisible) to 1.0 (opaque).

        Raises:
            ValueError: If opacity is outside [0.0, 1.0].
        """
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(
                f"Opacity must be between 0.0 and 1.0, got {opacity}"
            )
        self._config.opacity = opacity
        if self._window is not None:
            self._window.setAlphaValue_(opacity)

    def attach_to_window(self, window_info: WindowInfo) -> None:
        """Position overlay relative to a poker table window.

        Converts the poker window's top-left screen coordinates to
        AppKit's bottom-left coordinate system and positions the overlay
        at the top of the poker window.

        Args:
            window_info: Information about the poker client window.
        """
        self._attached_window = window_info

        # Get screen height for coordinate conversion.
        # macOS screen coordinates: origin at bottom-left.
        # Window managers typically report top-left origin.
        screen_height = self._get_screen_height()

        # Position overlay at top of poker window
        x = window_info.x
        y = screen_height - window_info.y - self._config.height

        self.set_position(x, y)
        self.set_size(window_info.width, self._config.height)

        logger.info(
            "Overlay attached to window '%s' at (%.0f, %.0f)",
            window_info.title,
            x,
            y,
        )

    def reposition_to_attached(self) -> None:
        """Reposition overlay to match the currently attached window.

        Call this when the poker window has moved or resized.
        Does nothing if no window is attached.
        """
        if self._attached_window is not None:
            self.attach_to_window(self._attached_window)

    def update_attached_window(self, window_info: WindowInfo) -> None:
        """Update the attached window info and reposition.

        Args:
            window_info: Updated poker window information.
        """
        self.attach_to_window(window_info)

    def show_panel(self, panel_type: PanelType) -> None:
        """Show a specific panel.

        Args:
            panel_type: The panel to show.
        """
        if panel_type in self._panels:
            self._panels[panel_type].visible = True
            if panel_type in self._panel_text_fields:
                self._panel_text_fields[panel_type].setHidden_(False)
            logger.debug("Panel %s shown", panel_type.value)

    def hide_panel(self, panel_type: PanelType) -> None:
        """Hide a specific panel.

        Args:
            panel_type: The panel to hide.
        """
        if panel_type in self._panels:
            self._panels[panel_type].visible = False
            if panel_type in self._panel_text_fields:
                self._panel_text_fields[panel_type].setHidden_(True)
            logger.debug("Panel %s hidden", panel_type.value)

    def is_panel_visible(self, panel_type: PanelType) -> bool:
        """Check if a panel is currently visible.

        Args:
            panel_type: The panel to check.

        Returns:
            True if the panel is visible.
        """
        if panel_type in self._panels:
            return self._panels[panel_type].visible
        return False

    def set_panel_content(self, panel_type: PanelType, content: str) -> None:
        """Update the text content of a specific panel.

        Args:
            panel_type: The panel to update.
            content: New text content.
        """
        if panel_type in self._panels:
            self._panels[panel_type].content = content
            if panel_type in self._panel_text_fields:
                self._panel_text_fields[panel_type].setStringValue_(content)

    @property
    def is_visible(self) -> bool:
        """Return whether the overlay window is currently visible."""
        if self._window is not None:
            return bool(self._window.isVisible())
        return False

    def _get_screen_height(self) -> float:
        """Return main screen height for coordinate conversion.

        Returns:
            Screen height in points.
        """
        screen = NSScreen.mainScreen()
        frame = screen.frame()
        return frame.size.height

    @staticmethod
    def screen_size() -> tuple[float, float]:
        """Return the main screen size as (width, height).

        Returns:
            Tuple of (width, height) in points.
        """
        screen = NSScreen.mainScreen()
        frame = screen.frame()
        return (frame.size.width, frame.size.height)
