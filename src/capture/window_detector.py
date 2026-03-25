"""Poker table window auto-detection using macOS Quartz APIs.

Enumerates visible windows via CGWindowListCopyWindowInfo and matches
known poker client title patterns. Supports detecting multiple tables
and auto-refreshing the window list. Works with both native poker
clients and browser-based poker sites (Chrome, Safari, Firefox, etc.).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# Default poker client window title patterns (case-insensitive).
# These match against both window titles and owner names, so they work for
# native poker clients AND browser-based poker sites (where the owner is
# e.g. "Google Chrome" but the tab title contains the poker site name).
DEFAULT_TITLE_PATTERNS: tuple[str, ...] = (
    # Native poker clients
    r"PokerStars",
    r"888poker",
    r"PartyPoker",
    r"GGPoker",
    r"WPT\s+Global",
    # Browser-based poker sites
    r"ClubGG",
    r"Natural8",
    r"Ignition\s+(Casino|Poker)",
    r"Bovada",
    r"ACR\s+Poker",
    r"Americas\s+Cardroom",
    r"BetOnline",
    r"Global\s+Poker",
    r"WSOP\.com",
    r"PokerKing",
    r"Replay\s+Poker",
    r"CoinPoker",
    r"PPPoker",
    r"Winamax",
    # Generic poker keyword — catches most poker sites in browser tabs.
    # Negative lookbehind/lookahead avoids matching file paths (poker-hud)
    # and project names (poker_stats).
    r"(?<!/)\bPoker\b(?![-_])",
    # Game type indicators (work across native and browser)
    r"Table\b",
    r"Hold'?em",
    r"No\s+Limit",
    r"Pot\s+Limit",
    r"Omaha",
    r"Tournament",
    r"Cash\s+Game",
    r"Sit\s*&?\s*Go",
)


@dataclass(frozen=True)
class WindowInfo:
    """Information about a detected window.

    Attributes:
        window_id: The macOS CGWindowID.
        title: The window title string.
        owner_name: The name of the application owning the window.
        x: Horizontal position of the window's top-left corner.
        y: Vertical position of the window's top-left corner.
        width: Width of the window in pixels.
        height: Height of the window in pixels.
    """

    window_id: int
    title: str
    owner_name: str
    x: int
    y: int
    width: int
    height: int

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Return the window bounds as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)


class WindowDetectionError(Exception):
    """Raised when window enumeration fails."""


class WindowDetector:
    """Detects poker client windows on macOS.

    Uses Quartz CGWindowListCopyWindowInfo to enumerate on-screen windows
    and filters them against configurable title patterns.

    Args:
        title_patterns: Regex patterns to match against window titles.
            Defaults to common poker client patterns.
        min_window_size: Minimum (width, height) for a window to be
            considered. Filters out tiny helper windows.
        cache_ttl: How long (in seconds) to cache the window list before
            re-querying. Set to 0 to disable caching.
    """

    def __init__(
        self,
        title_patterns: Optional[Sequence[str]] = None,
        min_window_size: tuple[int, int] = (400, 300),
        cache_ttl: float = 1.0,
    ) -> None:
        self._patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (title_patterns or DEFAULT_TITLE_PATTERNS)
        ]
        self._min_width, self._min_height = min_window_size
        self._cache_ttl = cache_ttl
        self._cached_windows: list[WindowInfo] = []
        self._cache_timestamp: float = 0.0

    @property
    def patterns(self) -> list[re.Pattern[str]]:
        """The compiled regex patterns used for matching."""
        return list(self._patterns)

    def detect_windows(self, force_refresh: bool = False) -> list[WindowInfo]:
        """Detect poker table windows currently visible on screen.

        Args:
            force_refresh: If True, bypass the cache and re-query the
                window list from the OS.

        Returns:
            A list of WindowInfo objects for matched poker windows,
            sorted by window ID for stable ordering.

        Raises:
            WindowDetectionError: If the Quartz API call fails.
        """
        now = time.monotonic()
        if (
            not force_refresh
            and self._cached_windows
            and (now - self._cache_timestamp) < self._cache_ttl
        ):
            return list(self._cached_windows)

        raw_windows = self._enumerate_windows()
        matched = self._filter_poker_windows(raw_windows)
        matched.sort(key=lambda w: w.window_id)

        self._cached_windows = matched
        self._cache_timestamp = time.monotonic()

        logger.info(
            "Detected %d poker window(s) out of %d total",
            len(matched),
            len(raw_windows),
        )
        return list(matched)

    def find_window_by_id(self, window_id: int) -> Optional[WindowInfo]:
        """Find a specific window by its CGWindowID.

        Args:
            window_id: The CGWindowID to search for.

        Returns:
            The matching WindowInfo, or None if not found.
        """
        for window in self.detect_windows():
            if window.window_id == window_id:
                return window
        return None

    def _enumerate_windows(self) -> list[dict]:
        """Enumerate all on-screen windows via Quartz.

        Returns:
            A list of raw window info dicts from CGWindowListCopyWindowInfo.

        Raises:
            WindowDetectionError: If the API call fails or returns None.
        """
        try:
            from Quartz import (  # type: ignore[import-untyped]
                CGWindowListCopyWindowInfo,
                kCGNullWindowID,
                kCGWindowListOptionOnScreenOnly,
            )
        except ImportError as exc:
            raise WindowDetectionError(
                "Quartz framework not available. "
                "Install with: pip install pyobjc-framework-Quartz"
            ) from exc

        window_list = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly, kCGNullWindowID
        )
        if window_list is None:
            raise WindowDetectionError(
                "CGWindowListCopyWindowInfo returned None"
            )

        return list(window_list)

    def _filter_poker_windows(
        self, raw_windows: list[dict]
    ) -> list[WindowInfo]:
        """Filter raw window dicts to those matching poker title patterns.

        Args:
            raw_windows: Raw window info dicts from Quartz.

        Returns:
            A list of WindowInfo for windows matching the poker patterns.
        """
        matched: list[WindowInfo] = []

        for win_dict in raw_windows:
            title = win_dict.get("kCGWindowName", "") or ""
            owner = win_dict.get("kCGWindowOwnerName", "") or ""
            window_id = win_dict.get("kCGWindowNumber", 0)

            bounds = win_dict.get("kCGWindowBounds", {})
            if not bounds:
                continue

            x = int(bounds.get("X", 0))
            y = int(bounds.get("Y", 0))
            width = int(bounds.get("Width", 0))
            height = int(bounds.get("Height", 0))

            if width < self._min_width or height < self._min_height:
                continue

            search_text = f"{title} {owner}"
            if any(pattern.search(search_text) for pattern in self._patterns):
                matched.append(
                    WindowInfo(
                        window_id=window_id,
                        title=title,
                        owner_name=owner,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                    )
                )

        return matched
