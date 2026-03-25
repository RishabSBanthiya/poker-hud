"""Poker client window auto-detection on macOS.

Finds poker client windows by title pattern matching using
Quartz Window Services.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class WindowInfo:
    """Information about a detected window.

    Attributes:
        window_id: macOS window ID.
        title: Window title string.
        x: Window X position on screen.
        y: Window Y position on screen.
        width: Window width in pixels.
        height: Window height in pixels.
        owner_name: Name of the application owning the window.
    """

    window_id: int
    title: str
    x: int
    y: int
    width: int
    height: int
    owner_name: str


class WindowDetector:
    """Detects poker client windows by title pattern.

    Args:
        title_pattern: Regex pattern to match window titles against.
    """

    def __init__(self, title_pattern: str = ".*[Pp]oker.*") -> None:
        self._pattern = re.compile(title_pattern)
        self._last_window: Optional[WindowInfo] = None

    @property
    def last_window(self) -> Optional[WindowInfo]:
        """Return the most recently detected window, if any."""
        return self._last_window

    def detect(self) -> Optional[WindowInfo]:
        """Scan for a poker client window matching the title pattern.

        Returns:
            WindowInfo for the first matching window, or None if not found.
        """
        try:
            from Quartz import (  # type: ignore[import-untyped]
                CGWindowListCopyWindowInfo,
                kCGNullWindowID,
                kCGWindowListOptionOnScreenOnly,
            )
        except ImportError:
            logger.warning("Quartz not available; window detection disabled")
            return None

        window_list = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly, kCGNullWindowID
        )
        if window_list is None:
            return None

        for window in window_list:
            title = window.get("kCGWindowName", "")
            owner = window.get("kCGWindowOwnerName", "")
            if title and self._pattern.search(title):
                bounds = window.get("kCGWindowBounds", {})
                info = WindowInfo(
                    window_id=int(window.get("kCGWindowNumber", 0)),
                    title=str(title),
                    x=int(bounds.get("X", 0)),
                    y=int(bounds.get("Y", 0)),
                    width=int(bounds.get("Width", 0)),
                    height=int(bounds.get("Height", 0)),
                    owner_name=str(owner),
                )
                self._last_window = info
                logger.info(
                    "Detected poker window: '%s' (%dx%d)",
                    info.title,
                    info.width,
                    info.height,
                )
                return info

        return None
