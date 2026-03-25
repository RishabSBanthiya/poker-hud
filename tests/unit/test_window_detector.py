"""Unit tests for the window detector module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from src.capture.window_detector import (
    WindowDetectionError,
    WindowDetector,
    WindowInfo,
)


def _make_window_dict(
    name: str = "Test Window",
    owner: str = "TestApp",
    window_id: int = 1,
    x: int = 0,
    y: int = 0,
    width: int = 800,
    height: int = 600,
) -> dict:
    """Create a mock window dict matching CGWindowListCopyWindowInfo format."""
    return {
        "kCGWindowName": name,
        "kCGWindowOwnerName": owner,
        "kCGWindowNumber": window_id,
        "kCGWindowBounds": {
            "X": x,
            "Y": y,
            "Width": width,
            "Height": height,
        },
    }


class TestWindowInfo:
    """Tests for the WindowInfo dataclass."""

    def test_bounds_property(self) -> None:
        """bounds returns (x, y, width, height) tuple."""
        info = WindowInfo(
            window_id=1,
            title="Table 1",
            owner_name="PokerStars",
            x=100,
            y=200,
            width=800,
            height=600,
        )
        assert info.bounds == (100, 200, 800, 600)

    def test_frozen(self) -> None:
        """WindowInfo is immutable."""
        info = WindowInfo(
            window_id=1,
            title="Table",
            owner_name="App",
            x=0,
            y=0,
            width=800,
            height=600,
        )
        with pytest.raises(AttributeError):
            info.title = "New Title"  # type: ignore[misc]


class TestWindowDetectorInit:
    """Tests for WindowDetector initialization."""

    def test_default_patterns(self) -> None:
        """Default patterns include common poker clients."""
        detector = WindowDetector()
        pattern_strings = [p.pattern for p in detector.patterns]
        assert any("PokerStars" in p for p in pattern_strings)
        assert any("888poker" in p for p in pattern_strings)
        assert any("PartyPoker" in p for p in pattern_strings)

    def test_custom_patterns(self) -> None:
        """Custom title patterns override defaults."""
        detector = WindowDetector(title_patterns=["MyPokerApp"])
        assert len(detector.patterns) == 1
        assert detector.patterns[0].pattern == "MyPokerApp"

    def test_min_window_size_default(self) -> None:
        """Default minimum window size is reasonable."""
        detector = WindowDetector()
        assert detector._min_width == 400
        assert detector._min_height == 300


class TestWindowDetectorDetect:
    """Tests for detect_windows using mocked Quartz APIs."""

    def _mock_quartz(self, window_list: list[dict]):
        """Create a mock Quartz module returning the given window list."""
        mock_quartz = MagicMock()
        mock_quartz.CGWindowListCopyWindowInfo.return_value = window_list
        mock_quartz.kCGWindowListOptionOnScreenOnly = 1
        mock_quartz.kCGNullWindowID = 0
        return mock_quartz

    def test_detects_pokerstars_window(self) -> None:
        """A window titled 'PokerStars - Table 1' is detected."""
        windows = [
            _make_window_dict(
                name="PokerStars - Table 1 - No Limit Hold'em",
                owner="PokerStars",
                window_id=42,
            )
        ]
        mock_q = self._mock_quartz(windows)
        detector = WindowDetector()

        with patch.dict("sys.modules", {"Quartz": mock_q}):
            result = detector.detect_windows(force_refresh=True)

        assert len(result) == 1
        assert result[0].window_id == 42
        assert result[0].title == "PokerStars - Table 1 - No Limit Hold'em"
        assert result[0].owner_name == "PokerStars"

    def test_detects_multiple_tables(self) -> None:
        """Multiple poker windows are all detected."""
        windows = [
            _make_window_dict(
                name="Table 1", owner="PokerStars", window_id=1
            ),
            _make_window_dict(
                name="Table 2", owner="PokerStars", window_id=2
            ),
        ]
        mock_q = self._mock_quartz(windows)
        detector = WindowDetector()

        with patch.dict("sys.modules", {"Quartz": mock_q}):
            result = detector.detect_windows(force_refresh=True)

        assert len(result) == 2

    def test_ignores_non_poker_windows(self) -> None:
        """Non-poker windows (e.g., Finder) are not returned."""
        windows = [
            _make_window_dict(name="Finder", owner="Finder", window_id=1),
            _make_window_dict(
                name="Safari - Google", owner="Safari", window_id=2
            ),
        ]
        mock_q = self._mock_quartz(windows)
        detector = WindowDetector()

        with patch.dict("sys.modules", {"Quartz": mock_q}):
            result = detector.detect_windows(force_refresh=True)

        assert len(result) == 0

    def test_ignores_small_windows(self) -> None:
        """Windows below the minimum size threshold are filtered out."""
        windows = [
            _make_window_dict(
                name="PokerStars Tooltip",
                owner="PokerStars",
                window_id=1,
                width=100,
                height=50,
            ),
        ]
        mock_q = self._mock_quartz(windows)
        detector = WindowDetector()

        with patch.dict("sys.modules", {"Quartz": mock_q}):
            result = detector.detect_windows(force_refresh=True)

        assert len(result) == 0

    def test_matches_owner_name_too(self) -> None:
        """A window with a generic title but poker owner is detected."""
        windows = [
            _make_window_dict(
                name="Game Window",
                owner="888poker",
                window_id=10,
            ),
        ]
        mock_q = self._mock_quartz(windows)
        detector = WindowDetector()

        with patch.dict("sys.modules", {"Quartz": mock_q}):
            result = detector.detect_windows(force_refresh=True)

        assert len(result) == 1
        assert result[0].owner_name == "888poker"

    def test_sorted_by_window_id(self) -> None:
        """Results are sorted by window_id for stable ordering."""
        windows = [
            _make_window_dict(name="Table 2", owner="PokerStars", window_id=5),
            _make_window_dict(name="Table 1", owner="PokerStars", window_id=2),
        ]
        mock_q = self._mock_quartz(windows)
        detector = WindowDetector()

        with patch.dict("sys.modules", {"Quartz": mock_q}):
            result = detector.detect_windows(force_refresh=True)

        assert result[0].window_id == 2
        assert result[1].window_id == 5

    def test_cache_returns_same_results(self) -> None:
        """Subsequent calls within TTL return cached results."""
        windows = [
            _make_window_dict(name="Table 1", owner="PokerStars", window_id=1),
        ]
        mock_q = self._mock_quartz(windows)
        detector = WindowDetector(cache_ttl=10.0)

        with patch.dict("sys.modules", {"Quartz": mock_q}):
            result1 = detector.detect_windows(force_refresh=True)
            mock_q.CGWindowListCopyWindowInfo.return_value = []
            result2 = detector.detect_windows()

        assert len(result1) == 1
        assert len(result2) == 1
        assert mock_q.CGWindowListCopyWindowInfo.call_count == 1

    def test_force_refresh_bypasses_cache(self) -> None:
        """force_refresh=True queries Quartz again even within TTL."""
        windows = [
            _make_window_dict(name="Table 1", owner="PokerStars", window_id=1),
        ]
        mock_q = self._mock_quartz(windows)
        detector = WindowDetector(cache_ttl=10.0)

        with patch.dict("sys.modules", {"Quartz": mock_q}):
            detector.detect_windows(force_refresh=True)
            mock_q.CGWindowListCopyWindowInfo.return_value = []
            result = detector.detect_windows(force_refresh=True)

        assert len(result) == 0
        assert mock_q.CGWindowListCopyWindowInfo.call_count == 2

    def test_quartz_returns_none_raises(self) -> None:
        """WindowDetectionError raised when Quartz returns None."""
        mock_q = self._mock_quartz([])
        mock_q.CGWindowListCopyWindowInfo.return_value = None
        detector = WindowDetector()

        with patch.dict("sys.modules", {"Quartz": mock_q}):
            with pytest.raises(
                WindowDetectionError, match="returned None"
            ):
                detector.detect_windows(force_refresh=True)

    def test_quartz_import_error_raises(self) -> None:
        """WindowDetectionError wraps ImportError when Quartz is unavailable."""
        detector = WindowDetector()
        with patch.dict("sys.modules", {"Quartz": None}):
            with pytest.raises(WindowDetectionError, match="not available"):
                detector.detect_windows(force_refresh=True)

    def test_holdem_pattern_matches(self) -> None:
        """The Hold'em pattern matches window titles."""
        windows = [
            _make_window_dict(
                name="No Limit Hold'em $0.50/$1",
                owner="SomeApp",
                window_id=7,
            ),
        ]
        mock_q = self._mock_quartz(windows)
        detector = WindowDetector()

        with patch.dict("sys.modules", {"Quartz": mock_q}):
            result = detector.detect_windows(force_refresh=True)

        assert len(result) == 1

    def test_detects_poker_in_chrome(self) -> None:
        """A poker site running in Chrome is detected via tab title."""
        windows = [
            _make_window_dict(
                name="PokerStars - No Limit Hold'em - Table 1",
                owner="Google Chrome",
                window_id=20,
            ),
        ]
        mock_q = self._mock_quartz(windows)
        detector = WindowDetector()

        with patch.dict("sys.modules", {"Quartz": mock_q}):
            result = detector.detect_windows(force_refresh=True)

        assert len(result) == 1
        assert result[0].owner_name == "Google Chrome"

    def test_detects_poker_in_safari(self) -> None:
        """A poker site running in Safari is detected via tab title."""
        windows = [
            _make_window_dict(
                name="Global Poker - Cash Game",
                owner="Safari",
                window_id=30,
            ),
        ]
        mock_q = self._mock_quartz(windows)
        detector = WindowDetector()

        with patch.dict("sys.modules", {"Quartz": mock_q}):
            result = detector.detect_windows(force_refresh=True)

        assert len(result) == 1

    def test_detects_poker_in_firefox(self) -> None:
        """A poker site running in Firefox is detected via tab title."""
        windows = [
            _make_window_dict(
                name="ClubGG - Tournament Lobby",
                owner="Firefox",
                window_id=40,
            ),
        ]
        mock_q = self._mock_quartz(windows)
        detector = WindowDetector()

        with patch.dict("sys.modules", {"Quartz": mock_q}):
            result = detector.detect_windows(force_refresh=True)

        assert len(result) == 1

    def test_detects_browser_poker_sites(self) -> None:
        """Browser-only poker sites are detected by name."""
        browser_poker_sites = [
            ("Ignition Casino - No Limit Hold'em", "Google Chrome", 1),
            ("Bovada - Tournament", "Safari", 2),
            ("ACR Poker - Table 5", "Firefox", 3),
            ("BetOnline - Cash Game", "Google Chrome", 4),
            ("WSOP.com - Ring Game", "Safari", 5),
            ("CoinPoker - Sit & Go", "Firefox", 6),
        ]
        windows = [
            _make_window_dict(name=name, owner=owner, window_id=wid)
            for name, owner, wid in browser_poker_sites
        ]
        mock_q = self._mock_quartz(windows)
        detector = WindowDetector()

        with patch.dict("sys.modules", {"Quartz": mock_q}):
            result = detector.detect_windows(force_refresh=True)

        assert len(result) == len(browser_poker_sites)

    def test_generic_poker_keyword_in_browser(self) -> None:
        """A browser tab with 'Poker' in the title is detected."""
        windows = [
            _make_window_dict(
                name="SomeNew Poker Site - Lobby",
                owner="Google Chrome",
                window_id=50,
            ),
        ]
        mock_q = self._mock_quartz(windows)
        detector = WindowDetector()

        with patch.dict("sys.modules", {"Quartz": mock_q}):
            result = detector.detect_windows(force_refresh=True)

        assert len(result) == 1

    def test_ignores_non_poker_browser_tabs(self) -> None:
        """Browser tabs without poker keywords are not detected."""
        windows = [
            _make_window_dict(
                name="Gmail - Inbox",
                owner="Google Chrome",
                window_id=60,
            ),
            _make_window_dict(
                name="YouTube - Home",
                owner="Safari",
                window_id=61,
            ),
        ]
        mock_q = self._mock_quartz(windows)
        detector = WindowDetector()

        with patch.dict("sys.modules", {"Quartz": mock_q}):
            result = detector.detect_windows(force_refresh=True)

        assert len(result) == 0


class TestFindWindowById:
    """Tests for find_window_by_id."""

    def test_finds_existing_window(self) -> None:
        """Returns the WindowInfo for a known window ID."""
        windows = [
            _make_window_dict(name="Table 1", owner="PokerStars", window_id=42),
        ]
        mock_q = MagicMock()
        mock_q.CGWindowListCopyWindowInfo.return_value = windows
        mock_q.kCGWindowListOptionOnScreenOnly = 1
        mock_q.kCGNullWindowID = 0
        detector = WindowDetector()

        with patch.dict("sys.modules", {"Quartz": mock_q}):
            result = detector.find_window_by_id(42)

        assert result is not None
        assert result.window_id == 42

    def test_returns_none_for_unknown_id(self) -> None:
        """Returns None when the window ID does not exist."""
        mock_q = MagicMock()
        mock_q.CGWindowListCopyWindowInfo.return_value = []
        mock_q.kCGWindowListOptionOnScreenOnly = 1
        mock_q.kCGNullWindowID = 0
        detector = WindowDetector()

        with patch.dict("sys.modules", {"Quartz": mock_q}):
            result = detector.find_window_by_id(999)

        assert result is None
