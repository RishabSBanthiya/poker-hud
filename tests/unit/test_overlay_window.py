"""Unit tests for OverlayWindow.

All AppKit calls are mocked so these tests run in CI without a GUI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.overlay.overlay_window import OverlayConfig, OverlayWindow


# ---------------------------------------------------------------------------
# OverlayConfig tests
# ---------------------------------------------------------------------------


class TestOverlayConfig:
    """Tests for the OverlayConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = OverlayConfig()
        assert cfg.x == 0.0
        assert cfg.y == 0.0
        assert cfg.width == 500.0
        assert cfg.height == 60.0
        assert cfg.font_size == 18.0
        assert cfg.text_color == (1.0, 1.0, 1.0, 1.0)
        assert cfg.bg_color == (0.0, 0.0, 0.0, 0.6)

    def test_custom_values(self) -> None:
        cfg = OverlayConfig(x=100, y=200, width=300, height=40, font_size=14.0)
        assert cfg.x == 100
        assert cfg.y == 200
        assert cfg.width == 300
        assert cfg.height == 40
        assert cfg.font_size == 14.0


# ---------------------------------------------------------------------------
# OverlayWindow tests (mocked AppKit)
# ---------------------------------------------------------------------------


def _build_mock_window() -> MagicMock:
    """Create a mock NSWindow with the methods OverlayWindow calls."""
    mock_win = MagicMock()
    mock_content_view = MagicMock()
    mock_win.contentView.return_value = mock_content_view

    mock_frame = MagicMock()
    mock_frame.origin.x = 0.0
    mock_frame.origin.y = 0.0
    mock_frame.size.width = 500.0
    mock_frame.size.height = 60.0
    mock_win.frame.return_value = mock_frame
    mock_win.isVisible.return_value = False

    return mock_win


def _build_mock_text_field() -> MagicMock:
    """Create a mock NSTextField."""
    return MagicMock()


@pytest.fixture()
def mock_appkit():
    """Patch all AppKit objects used by OverlayWindow."""
    mock_win = _build_mock_window()
    mock_tf = _build_mock_text_field()

    with (
        patch(
            "src.overlay.overlay_window.NSWindow"
        ) as mock_nswindow_cls,
        patch(
            "src.overlay.overlay_window.NSTextField"
        ) as mock_nstf_cls,
        patch("src.overlay.overlay_window.NSColor") as mock_color,
        patch("src.overlay.overlay_window.NSFont") as mock_font,
        patch("src.overlay.overlay_window.NSMakeRect") as mock_make_rect,
    ):
        mock_nswindow_cls.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = (
            mock_win
        )
        mock_nstf_cls.alloc.return_value.initWithFrame_.return_value = mock_tf
        mock_make_rect.side_effect = lambda x, y, w, h: (x, y, w, h)

        yield {
            "window": mock_win,
            "text_field": mock_tf,
            "NSWindow": mock_nswindow_cls,
            "NSTextField": mock_nstf_cls,
            "NSColor": mock_color,
            "NSFont": mock_font,
            "NSMakeRect": mock_make_rect,
        }


class TestOverlayWindowInit:
    """Tests for OverlayWindow initialization (no AppKit calls)."""

    def test_default_config(self) -> None:
        ow = OverlayWindow()
        assert ow.config.width == 500.0
        assert ow.text == ""

    def test_custom_config_and_text(self) -> None:
        cfg = OverlayConfig(x=10, y=20, width=400, height=50)
        ow = OverlayWindow(config=cfg, text="Hello")
        assert ow.config.x == 10
        assert ow.text == "Hello"

    def test_is_visible_before_create(self) -> None:
        ow = OverlayWindow()
        assert ow.is_visible is False


class TestOverlayWindowCreate:
    """Tests for window creation and configuration."""

    def test_create_sets_borderless_transparent_ontop(
        self, mock_appkit: dict
    ) -> None:
        ow = OverlayWindow(text="Test")
        ow.create()

        win = mock_appkit["window"]
        win.setOpaque_.assert_called_once_with(False)
        win.setHasShadow_.assert_called_once_with(False)
        win.setIgnoresMouseEvents_.assert_called_once_with(True)
        win.setLevel_.assert_called_once_with(25)

    def test_create_sets_background_color(self, mock_appkit: dict) -> None:
        cfg = OverlayConfig(bg_color=(0.1, 0.2, 0.3, 0.8))
        ow = OverlayWindow(config=cfg)
        ow.create()

        mock_appkit["NSColor"].colorWithCalibratedRed_green_blue_alpha_.assert_any_call(
            0.1, 0.2, 0.3, 0.8
        )

    def test_create_configures_text_field(self, mock_appkit: dict) -> None:
        ow = OverlayWindow(text="VPIP: 24%")
        ow.create()

        tf = mock_appkit["text_field"]
        tf.setStringValue_.assert_called_once_with("VPIP: 24%")
        tf.setEditable_.assert_called_once_with(False)
        tf.setSelectable_.assert_called_once_with(False)
        tf.setBezeled_.assert_called_once_with(False)
        tf.setDrawsBackground_.assert_called_once_with(False)

    def test_create_adds_text_field_to_content_view(
        self, mock_appkit: dict
    ) -> None:
        ow = OverlayWindow()
        ow.create()

        content_view = mock_appkit["window"].contentView()
        content_view.addSubview_.assert_called_once_with(
            mock_appkit["text_field"]
        )


class TestOverlayWindowActions:
    """Tests for show, hide, close, and mutation methods."""

    def test_show_raises_if_not_created(self) -> None:
        ow = OverlayWindow()
        with pytest.raises(RuntimeError, match="not created"):
            ow.show()

    def test_show_calls_order_front(self, mock_appkit: dict) -> None:
        ow = OverlayWindow()
        ow.create()
        ow.show()
        mock_appkit["window"].orderFrontRegardless.assert_called_once()

    def test_hide(self, mock_appkit: dict) -> None:
        ow = OverlayWindow()
        ow.create()
        ow.hide()
        mock_appkit["window"].orderOut_.assert_called_once_with(None)

    def test_close_releases_references(self, mock_appkit: dict) -> None:
        ow = OverlayWindow()
        ow.create()
        ow.close()
        mock_appkit["window"].close.assert_called_once()
        assert ow._window is None
        assert ow._text_field is None

    def test_set_text_updates_text_field(self, mock_appkit: dict) -> None:
        ow = OverlayWindow()
        ow.create()
        ow.set_text("New text")
        assert ow.text == "New text"
        mock_appkit["text_field"].setStringValue_.assert_called_with("New text")

    def test_set_text_before_create(self) -> None:
        ow = OverlayWindow()
        ow.set_text("Early text")
        assert ow.text == "Early text"

    def test_set_position_updates_config(self, mock_appkit: dict) -> None:
        ow = OverlayWindow()
        ow.create()
        ow.set_position(100.0, 200.0)
        assert ow.config.x == 100.0
        assert ow.config.y == 200.0
        mock_appkit["window"].setFrameOrigin_.assert_called_once()

    def test_set_size_updates_config(self, mock_appkit: dict) -> None:
        ow = OverlayWindow()
        ow.create()
        ow.set_size(600.0, 80.0)
        assert ow.config.width == 600.0
        assert ow.config.height == 80.0
        mock_appkit["window"].setFrame_display_.assert_called_once()

    def test_is_visible_delegates_to_window(self, mock_appkit: dict) -> None:
        ow = OverlayWindow()
        ow.create()
        mock_appkit["window"].isVisible.return_value = True
        assert ow.is_visible is True


@pytest.mark.requires_gui
class TestOverlayWindowGUI:
    """Tests that require a real GUI environment.

    These are skipped in CI. Run locally with:
        pytest -m requires_gui
    """

    def test_create_and_show_real_window(self) -> None:
        """Smoke test: create and immediately close a real window."""
        from AppKit import NSApplication

        NSApplication.sharedApplication()
        ow = OverlayWindow(text="GUI test")
        ow.create()
        ow.show()
        assert ow.is_visible
        ow.close()
