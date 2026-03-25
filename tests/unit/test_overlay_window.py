"""Unit tests for OverlayWindow.

All AppKit calls are mocked so these tests run in CI without a GUI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from src.overlay.overlay_window import (
    OverlayConfig,
    OverlayWindow,
    PanelType,
    WindowInfo,
)

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
        assert cfg.opacity == 1.0

    def test_custom_values(self) -> None:
        cfg = OverlayConfig(x=100, y=200, width=300, height=40, font_size=14.0)
        assert cfg.x == 100
        assert cfg.y == 200
        assert cfg.width == 300
        assert cfg.height == 40
        assert cfg.font_size == 14.0


# ---------------------------------------------------------------------------
# WindowInfo tests
# ---------------------------------------------------------------------------


class TestWindowInfo:
    """Tests for the WindowInfo dataclass."""

    def test_defaults(self) -> None:
        info = WindowInfo()
        assert info.x == 0.0
        assert info.y == 0.0
        assert info.width == 800.0
        assert info.height == 600.0
        assert info.title == ""

    def test_custom_values(self) -> None:
        info = WindowInfo(x=100, y=200, width=1024, height=768, title="PokerStars")
        assert info.x == 100
        assert info.title == "PokerStars"


# ---------------------------------------------------------------------------
# Mock helpers
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
        patch("src.overlay.overlay_window.NSScreen") as mock_screen,
    ):
        init_method = (
            mock_nswindow_cls.alloc.return_value
            .initWithContentRect_styleMask_backing_defer_
        )
        init_method.return_value = mock_win
        # Each alloc().initWithFrame_() call returns a new mock text field
        panel_fields = [MagicMock() for _ in range(4)]
        mock_nstf_cls.alloc.return_value.initWithFrame_.side_effect = [
            mock_tf
        ] + panel_fields
        mock_make_rect.side_effect = lambda x, y, w, h: (x, y, w, h)

        # Mock screen for coordinate conversion
        mock_screen_frame = MagicMock()
        mock_screen_frame.size.height = 1080.0
        mock_screen_frame.size.width = 1920.0
        mock_screen.mainScreen.return_value.frame.return_value = mock_screen_frame

        yield {
            "window": mock_win,
            "text_field": mock_tf,
            "panel_fields": panel_fields,
            "NSWindow": mock_nswindow_cls,
            "NSTextField": mock_nstf_cls,
            "NSColor": mock_color,
            "NSFont": mock_font,
            "NSMakeRect": mock_make_rect,
            "NSScreen": mock_screen,
        }


# ---------------------------------------------------------------------------
# OverlayWindow init tests
# ---------------------------------------------------------------------------


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

    def test_default_panels_initialized(self) -> None:
        ow = OverlayWindow()
        assert PanelType.STATS in ow.panels
        assert PanelType.SOLVER in ow.panels
        assert PanelType.SETTINGS in ow.panels
        assert ow.panels[PanelType.STATS].visible is True
        assert ow.panels[PanelType.SETTINGS].visible is False

    def test_no_attached_window_initially(self) -> None:
        ow = OverlayWindow()
        assert ow.attached_window is None


# ---------------------------------------------------------------------------
# OverlayWindow create tests
# ---------------------------------------------------------------------------


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

    def test_create_sets_opacity(self, mock_appkit: dict) -> None:
        cfg = OverlayConfig(opacity=0.7)
        ow = OverlayWindow(config=cfg)
        ow.create()

        mock_appkit["window"].setAlphaValue_.assert_called_once_with(0.7)

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
        # Main text field + 3 panel text fields
        assert content_view.addSubview_.call_count == 4


# ---------------------------------------------------------------------------
# OverlayWindow actions tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Opacity tests
# ---------------------------------------------------------------------------


class TestOverlayOpacity:
    """Tests for opacity control."""

    def test_set_opacity_updates_config(self, mock_appkit: dict) -> None:
        ow = OverlayWindow()
        ow.create()
        ow.set_opacity(0.5)
        assert ow.config.opacity == 0.5
        mock_appkit["window"].setAlphaValue_.assert_called_with(0.5)

    def test_set_opacity_rejects_invalid(self) -> None:
        ow = OverlayWindow()
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            ow.set_opacity(1.5)
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            ow.set_opacity(-0.1)


# ---------------------------------------------------------------------------
# Panel management tests
# ---------------------------------------------------------------------------


class TestOverlayPanels:
    """Tests for panel show/hide and content management."""

    def test_show_panel(self, mock_appkit: dict) -> None:
        ow = OverlayWindow()
        ow.create()
        ow.hide_panel(PanelType.STATS)
        assert ow.is_panel_visible(PanelType.STATS) is False
        ow.show_panel(PanelType.STATS)
        assert ow.is_panel_visible(PanelType.STATS) is True

    def test_hide_panel(self, mock_appkit: dict) -> None:
        ow = OverlayWindow()
        ow.create()
        ow.hide_panel(PanelType.SOLVER)
        assert ow.is_panel_visible(PanelType.SOLVER) is False

    def test_set_panel_content(self, mock_appkit: dict) -> None:
        ow = OverlayWindow()
        ow.create()
        ow.set_panel_content(PanelType.STATS, "22/18/8/2.1")
        assert ow.panels[PanelType.STATS].content == "22/18/8/2.1"

    def test_is_panel_visible_unknown_panel(self) -> None:
        ow = OverlayWindow()
        # PanelType enum only has 3 values, so we test existing ones
        assert ow.is_panel_visible(PanelType.SETTINGS) is False


# ---------------------------------------------------------------------------
# Window attachment tests
# ---------------------------------------------------------------------------


class TestOverlayAttachment:
    """Tests for attaching to a poker window."""

    def test_attach_to_window(self, mock_appkit: dict) -> None:
        ow = OverlayWindow()
        ow.create()
        info = WindowInfo(x=100, y=50, width=1024, height=768, title="PS")
        ow.attach_to_window(info)

        assert ow.attached_window is info
        assert ow.config.x == 100.0
        # y = screen_height(1080) - window_y(50) - overlay_height(60)
        assert ow.config.y == 1080.0 - 50.0 - 60.0

    def test_attach_updates_width(self, mock_appkit: dict) -> None:
        ow = OverlayWindow()
        ow.create()
        info = WindowInfo(x=0, y=0, width=1200, height=800)
        ow.attach_to_window(info)
        assert ow.config.width == 1200.0

    def test_reposition_to_attached(self, mock_appkit: dict) -> None:
        ow = OverlayWindow()
        ow.create()
        info = WindowInfo(x=100, y=50, width=1024, height=768)
        ow.attach_to_window(info)

        # Simulate window move
        info.x = 200
        ow.update_attached_window(info)
        assert ow.config.x == 200.0

    def test_reposition_noop_without_attachment(self, mock_appkit: dict) -> None:
        ow = OverlayWindow()
        ow.create()
        ow.reposition_to_attached()
        # Should not raise, just do nothing
        assert ow.attached_window is None


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
