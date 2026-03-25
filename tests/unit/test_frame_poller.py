"""Unit tests for the frame poller module."""

from __future__ import annotations

import numpy as np
import pytest
from src.capture.frame_poller import FramePoller, FrameStats


def _make_frame(
    height: int = 100,
    width: int = 100,
    value: int = 0,
) -> np.ndarray:
    """Create a solid-color BGR frame for testing."""
    return np.full((height, width, 3), value, dtype=np.uint8)


class TestFrameStatsInit:
    """Tests for FrameStats."""

    def test_defaults(self) -> None:
        """Stats start at zero."""
        stats = FrameStats()
        assert stats.total_captured == 0
        assert stats.changes_detected == 0
        assert stats.frames_skipped == 0
        assert stats.last_change_time is None

    def test_change_rate_zero_when_empty(self) -> None:
        """Change rate is 0.0 with no captures."""
        stats = FrameStats()
        assert stats.change_rate == 0.0

    def test_change_rate_calculation(self) -> None:
        """Change rate is changes_detected / total_captured."""
        stats = FrameStats(total_captured=10, changes_detected=3)
        assert stats.change_rate == pytest.approx(0.3)

    def test_reset(self) -> None:
        """Reset clears all counters."""
        stats = FrameStats(
            total_captured=5, changes_detected=2, frames_skipped=3
        )
        stats.reset()
        assert stats.total_captured == 0
        assert stats.changes_detected == 0
        assert stats.frames_skipped == 0
        assert stats.last_change_time is None


class TestFramePollerInit:
    """Tests for FramePoller initialization."""

    def test_default_init(self) -> None:
        """FramePoller can be created with default parameters."""
        poller = FramePoller()
        assert poller.polling_interval == pytest.approx(0.2)

    def test_custom_params(self) -> None:
        """FramePoller accepts custom parameters."""
        poller = FramePoller(
            polling_interval=0.5,
            change_threshold=10.0,
            change_pixel_ratio=0.05,
            downscale_factor=2,
        )
        assert poller.polling_interval == pytest.approx(0.5)

    def test_negative_interval_raises(self) -> None:
        """Negative polling interval raises ValueError."""
        with pytest.raises(ValueError, match="polling_interval"):
            FramePoller(polling_interval=-1.0)

    def test_negative_threshold_raises(self) -> None:
        """Negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="change_threshold"):
            FramePoller(change_threshold=-1.0)

    def test_invalid_pixel_ratio_raises(self) -> None:
        """Pixel ratio outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="change_pixel_ratio"):
            FramePoller(change_pixel_ratio=1.5)

    def test_invalid_downscale_raises(self) -> None:
        """Downscale factor < 1 raises ValueError."""
        with pytest.raises(ValueError, match="downscale_factor"):
            FramePoller(downscale_factor=0)

    def test_interval_setter(self) -> None:
        """Polling interval can be updated via setter."""
        poller = FramePoller()
        poller.polling_interval = 0.5
        assert poller.polling_interval == pytest.approx(0.5)

    def test_interval_setter_negative_raises(self) -> None:
        """Setting negative interval raises ValueError."""
        poller = FramePoller()
        with pytest.raises(ValueError, match="polling_interval"):
            poller.polling_interval = -1.0


class TestFramePollerProcessFrame:
    """Tests for process_frame change detection logic."""

    def test_first_frame_always_changed(self) -> None:
        """The very first frame is always considered a change."""
        poller = FramePoller()
        frame = _make_frame(value=128)
        assert poller.process_frame(frame) is True
        assert poller.stats.total_captured == 1
        assert poller.stats.changes_detected == 1

    def test_identical_frame_not_changed(self) -> None:
        """An identical second frame is not considered changed."""
        poller = FramePoller()
        frame = _make_frame(value=128)
        poller.process_frame(frame)

        assert poller.process_frame(frame.copy()) is False
        assert poller.stats.total_captured == 2
        assert poller.stats.changes_detected == 1
        assert poller.stats.frames_skipped == 1

    def test_very_different_frame_detected(self) -> None:
        """A dramatically different frame is detected as changed."""
        poller = FramePoller()
        frame_black = _make_frame(value=0)
        frame_white = _make_frame(value=255)

        poller.process_frame(frame_black)
        assert poller.process_frame(frame_white) is True
        assert poller.stats.changes_detected == 2

    def test_minor_noise_ignored(self) -> None:
        """Small per-pixel noise below threshold is ignored."""
        poller = FramePoller(change_threshold=10.0, change_pixel_ratio=0.01)
        frame1 = _make_frame(value=128)
        frame2 = frame1.copy()
        # Add noise of magnitude 2 (well below threshold of 10)
        frame2 = (frame2.astype(np.int16) + 2).clip(0, 255).astype(np.uint8)

        poller.process_frame(frame1)
        assert poller.process_frame(frame2) is False

    def test_different_shape_always_changed(self) -> None:
        """Frames of different sizes are always detected as changed."""
        poller = FramePoller()
        frame1 = _make_frame(height=100, width=100, value=0)
        frame2 = _make_frame(height=200, width=200, value=0)

        poller.process_frame(frame1)
        assert poller.process_frame(frame2) is True

    def test_localized_change_detected(self) -> None:
        """A significant change in a region of the frame is detected."""
        poller = FramePoller(
            change_threshold=5.0,
            change_pixel_ratio=0.01,
            downscale_factor=1,
        )
        frame1 = _make_frame(height=100, width=100, value=0)
        frame2 = frame1.copy()
        # Change 5% of pixels (5x100 area) by a large amount
        frame2[:5, :, :] = 255

        poller.process_frame(frame1)
        assert poller.process_frame(frame2) is True

    def test_stats_last_change_time_updated(self) -> None:
        """last_change_time is set after a detected change."""
        poller = FramePoller()
        frame = _make_frame(value=100)
        poller.process_frame(frame)
        assert poller.stats.last_change_time is not None

    def test_change_rate_reflects_activity(self) -> None:
        """Change rate correctly reflects the ratio of changed frames."""
        poller = FramePoller()
        frame = _make_frame(value=0)

        poller.process_frame(frame)  # changed (first)
        poller.process_frame(frame)  # skipped
        poller.process_frame(frame)  # skipped
        poller.process_frame(frame)  # skipped

        assert poller.stats.change_rate == pytest.approx(0.25)


class TestFramePollerReset:
    """Tests for the reset method."""

    def test_reset_clears_state(self) -> None:
        """Reset clears the reference frame and stats."""
        poller = FramePoller()
        frame = _make_frame(value=128)
        poller.process_frame(frame)
        poller.process_frame(frame)

        poller.reset()
        assert poller.stats.total_captured == 0
        assert poller.stats.changes_detected == 0

        # After reset, next frame should be treated as the first
        assert poller.process_frame(frame) is True


class TestFramePollerShouldPoll:
    """Tests for the should_poll timing method."""

    def test_should_poll_initially(self) -> None:
        """should_poll returns True before any polling has occurred."""
        poller = FramePoller(polling_interval=1.0)
        assert poller.should_poll() is True

    def test_should_not_poll_immediately_after(self) -> None:
        """should_poll returns False right after a process_frame call."""
        poller = FramePoller(polling_interval=10.0)
        frame = _make_frame()
        poller.process_frame(frame)
        assert poller.should_poll() is False

    def test_zero_interval_always_polls(self) -> None:
        """With zero interval, should_poll always returns True."""
        poller = FramePoller(polling_interval=0.0)
        frame = _make_frame()
        poller.process_frame(frame)
        assert poller.should_poll() is True
