"""Unit tests for the capture pipeline coordinator."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from src.capture.frame_poller import FramePoller
from src.capture.pipeline import (
    CapturePipeline,
    PipelineConfig,
    PipelineState,
)
from src.capture.screen_capture import (
    ScreenCapture,
    ScreenCaptureError,
    ScreenRecordingPermissionError,
)
from src.capture.window_detector import (
    WindowDetector,
    WindowInfo,
)


def _make_window(
    window_id: int = 1,
    title: str = "PokerStars - Table 1",
    owner: str = "PokerStars",
    x: int = 0,
    y: int = 0,
    width: int = 800,
    height: int = 600,
) -> WindowInfo:
    """Create a WindowInfo for testing."""
    return WindowInfo(
        window_id=window_id,
        title=title,
        owner_name=owner,
        x=x,
        y=y,
        width=width,
        height=height,
    )


def _make_frame(
    height: int = 600, width: int = 800, value: int = 0
) -> np.ndarray:
    """Create a solid-color BGR frame."""
    return np.full((height, width, 3), value, dtype=np.uint8)


class TestPipelineConfig:
    """Tests for PipelineConfig defaults."""

    def test_defaults(self) -> None:
        """Default config values are sensible."""
        config = PipelineConfig()
        assert config.polling_interval == 0.2
        assert config.max_consecutive_errors == 10
        assert config.window_refresh_interval == 5.0


class TestPipelineState:
    """Tests for pipeline state transitions."""

    def test_initial_state_is_stopped(self) -> None:
        """A new pipeline starts in STOPPED state."""
        pipeline = CapturePipeline()
        assert pipeline.state == PipelineState.STOPPED

    def test_start_changes_state_to_running(self) -> None:
        """Starting the pipeline transitions to RUNNING."""
        detector = MagicMock(spec=WindowDetector)
        detector.detect_windows.return_value = []
        capture = MagicMock(spec=ScreenCapture)
        poller = MagicMock(spec=FramePoller)

        pipeline = CapturePipeline(
            window_detector=detector,
            screen_capture=capture,
            frame_poller=poller,
        )
        pipeline.start()
        try:
            assert pipeline.state == PipelineState.RUNNING
        finally:
            pipeline.stop()

    def test_stop_changes_state_to_stopped(self) -> None:
        """Stopping the pipeline transitions to STOPPED."""
        detector = MagicMock(spec=WindowDetector)
        detector.detect_windows.return_value = []
        pipeline = CapturePipeline(window_detector=detector)
        pipeline.start()
        pipeline.stop()
        assert pipeline.state == PipelineState.STOPPED

    def test_double_start_raises(self) -> None:
        """Starting an already-running pipeline raises RuntimeError."""
        detector = MagicMock(spec=WindowDetector)
        detector.detect_windows.return_value = []
        pipeline = CapturePipeline(window_detector=detector)
        pipeline.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                pipeline.start()
        finally:
            pipeline.stop()

    def test_stop_when_stopped_is_noop(self) -> None:
        """Stopping an already-stopped pipeline does nothing."""
        pipeline = CapturePipeline()
        pipeline.stop()  # Should not raise
        assert pipeline.state == PipelineState.STOPPED

    def test_pause_and_resume(self) -> None:
        """Pipeline can be paused and resumed."""
        detector = MagicMock(spec=WindowDetector)
        detector.detect_windows.return_value = []
        pipeline = CapturePipeline(window_detector=detector)
        pipeline.start()
        try:
            pipeline.pause()
            assert pipeline.state == PipelineState.PAUSED

            pipeline.resume()
            assert pipeline.state == PipelineState.RUNNING
        finally:
            pipeline.stop()

    def test_pause_when_not_running_is_noop(self) -> None:
        """Pausing a stopped pipeline does nothing."""
        pipeline = CapturePipeline()
        pipeline.pause()
        assert pipeline.state == PipelineState.STOPPED

    def test_resume_when_not_paused_is_noop(self) -> None:
        """Resuming a non-paused pipeline does nothing."""
        pipeline = CapturePipeline()
        pipeline.resume()
        assert pipeline.state == PipelineState.STOPPED


class TestPipelineHandlers:
    """Tests for frame handler registration."""

    def test_register_handler(self) -> None:
        """A handler can be registered."""
        pipeline = CapturePipeline()
        handler = MagicMock()
        pipeline.register_handler(handler)
        assert handler in pipeline._handlers

    def test_register_same_handler_twice_is_idempotent(self) -> None:
        """Registering the same handler twice only adds it once."""
        pipeline = CapturePipeline()
        handler = MagicMock()
        pipeline.register_handler(handler)
        pipeline.register_handler(handler)
        assert pipeline._handlers.count(handler) == 1

    def test_unregister_handler(self) -> None:
        """A registered handler can be removed."""
        pipeline = CapturePipeline()
        handler = MagicMock()
        pipeline.register_handler(handler)
        pipeline.unregister_handler(handler)
        assert handler not in pipeline._handlers

    def test_unregister_unknown_handler_raises(self) -> None:
        """Removing an unregistered handler raises ValueError."""
        pipeline = CapturePipeline()
        with pytest.raises(ValueError):
            pipeline.unregister_handler(MagicMock())


class TestPipelineTick:
    """Tests for the _tick method (single capture cycle)."""

    def test_tick_dispatches_changed_frame(self) -> None:
        """A changed frame is dispatched to registered handlers."""
        window = _make_window()
        frame = _make_frame()

        detector = MagicMock(spec=WindowDetector)
        detector.detect_windows.return_value = [window]

        capture = MagicMock(spec=ScreenCapture)
        capture.capture_frame.return_value = frame

        poller = MagicMock(spec=FramePoller)
        poller.process_frame.return_value = True

        config = PipelineConfig(window_refresh_interval=0.0)
        pipeline = CapturePipeline(
            config=config,
            window_detector=detector,
            screen_capture=capture,
            frame_poller=poller,
        )

        handler = MagicMock()
        pipeline.register_handler(handler)

        pipeline._tick()

        handler.assert_called_once()
        dispatched_frame, dispatched_window = handler.call_args[0]
        np.testing.assert_array_equal(dispatched_frame, frame)
        assert dispatched_window == window

    def test_tick_skips_unchanged_frame(self) -> None:
        """An unchanged frame is not dispatched."""
        window = _make_window()
        frame = _make_frame()

        detector = MagicMock(spec=WindowDetector)
        detector.detect_windows.return_value = [window]

        capture = MagicMock(spec=ScreenCapture)
        capture.capture_frame.return_value = frame

        poller = MagicMock(spec=FramePoller)
        poller.process_frame.return_value = False

        config = PipelineConfig(window_refresh_interval=0.0)
        pipeline = CapturePipeline(
            config=config,
            window_detector=detector,
            screen_capture=capture,
            frame_poller=poller,
        )

        handler = MagicMock()
        pipeline.register_handler(handler)

        pipeline._tick()

        handler.assert_not_called()

    def test_tick_without_window_does_nothing(self) -> None:
        """When no poker window is found, tick returns without capturing."""
        detector = MagicMock(spec=WindowDetector)
        detector.detect_windows.return_value = []

        capture = MagicMock(spec=ScreenCapture)

        config = PipelineConfig(window_refresh_interval=0.0)
        pipeline = CapturePipeline(
            config=config,
            window_detector=detector,
            screen_capture=capture,
        )

        pipeline._tick()

        capture.capture_frame.assert_not_called()

    def test_handler_exception_does_not_crash_pipeline(self) -> None:
        """An exception in a handler is caught and logged, not propagated."""
        window = _make_window()
        frame = _make_frame()

        detector = MagicMock(spec=WindowDetector)
        detector.detect_windows.return_value = [window]

        capture = MagicMock(spec=ScreenCapture)
        capture.capture_frame.return_value = frame

        poller = MagicMock(spec=FramePoller)
        poller.process_frame.return_value = True

        config = PipelineConfig(window_refresh_interval=0.0)
        pipeline = CapturePipeline(
            config=config,
            window_detector=detector,
            screen_capture=capture,
            frame_poller=poller,
        )

        bad_handler = MagicMock(side_effect=RuntimeError("boom"))
        good_handler = MagicMock()
        pipeline.register_handler(bad_handler)
        pipeline.register_handler(good_handler)

        pipeline._tick()  # Should not raise

        bad_handler.assert_called_once()
        good_handler.assert_called_once()


class TestPipelineErrorRecovery:
    """Tests for error recovery behavior."""

    def test_permission_error_enters_error_state(self) -> None:
        """A ScreenRecordingPermissionError stops the pipeline."""
        window = _make_window()

        detector = MagicMock(spec=WindowDetector)
        detector.detect_windows.return_value = [window]

        capture = MagicMock(spec=ScreenCapture)
        capture.capture_frame.side_effect = ScreenRecordingPermissionError(
            "denied"
        )

        config = PipelineConfig(
            polling_interval=0.01,
            window_refresh_interval=0.0,
        )
        pipeline = CapturePipeline(
            config=config,
            window_detector=detector,
            screen_capture=capture,
        )
        pipeline.start()

        # Wait for the pipeline thread to process and enter error state
        time.sleep(0.2)

        assert pipeline.state == PipelineState.ERROR
        pipeline.stop()

    def test_transient_errors_recovered(self) -> None:
        """Transient capture errors are retried until max_consecutive_errors."""
        window = _make_window()

        detector = MagicMock(spec=WindowDetector)
        detector.detect_windows.return_value = [window]

        # Fail a few times then succeed
        call_count = 0

        def capture_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise ScreenCaptureError("transient")
            return _make_frame()

        capture = MagicMock(spec=ScreenCapture)
        capture.capture_frame.side_effect = capture_side_effect

        poller = MagicMock(spec=FramePoller)
        poller.process_frame.return_value = True
        poller.stats = MagicMock()

        config = PipelineConfig(
            polling_interval=0.01,
            error_recovery_delay=0.01,
            max_consecutive_errors=10,
            window_refresh_interval=0.0,
        )
        pipeline = CapturePipeline(
            config=config,
            window_detector=detector,
            screen_capture=capture,
            frame_poller=poller,
        )
        pipeline.start()
        time.sleep(0.5)
        pipeline.stop()

        # Pipeline should have recovered after 3 failures
        assert call_count > 3

    def test_max_errors_enters_error_state(self) -> None:
        """Exceeding max_consecutive_errors enters ERROR state."""
        window = _make_window()

        detector = MagicMock(spec=WindowDetector)
        detector.detect_windows.return_value = [window]

        capture = MagicMock(spec=ScreenCapture)
        capture.capture_frame.side_effect = ScreenCaptureError("persistent")

        config = PipelineConfig(
            polling_interval=0.01,
            error_recovery_delay=0.01,
            max_consecutive_errors=3,
            window_refresh_interval=0.0,
        )
        pipeline = CapturePipeline(
            config=config,
            window_detector=detector,
            screen_capture=capture,
        )
        pipeline.start()
        time.sleep(0.3)

        assert pipeline.state == PipelineState.ERROR
        pipeline.stop()


class TestPipelineWindowTracking:
    """Tests for active window tracking logic."""

    def test_tracks_first_detected_window(self) -> None:
        """Pipeline selects the first detected window."""
        window = _make_window(window_id=42, title="Table 1")
        frame = _make_frame()

        detector = MagicMock(spec=WindowDetector)
        detector.detect_windows.return_value = [window]

        capture = MagicMock(spec=ScreenCapture)
        capture.capture_frame.return_value = frame

        poller = MagicMock(spec=FramePoller)
        poller.process_frame.return_value = False

        config = PipelineConfig(window_refresh_interval=0.0)
        pipeline = CapturePipeline(
            config=config,
            window_detector=detector,
            screen_capture=capture,
            frame_poller=poller,
        )

        pipeline._tick()

        assert pipeline.active_window is not None
        assert pipeline.active_window.window_id == 42

    def test_keeps_tracking_same_window(self) -> None:
        """Pipeline keeps the same window if it's still present."""
        window1 = _make_window(window_id=1, title="Table 1")
        window2 = _make_window(window_id=2, title="Table 2")
        frame = _make_frame()

        detector = MagicMock(spec=WindowDetector)
        detector.detect_windows.return_value = [window1, window2]

        capture = MagicMock(spec=ScreenCapture)
        capture.capture_frame.return_value = frame

        poller = MagicMock(spec=FramePoller)
        poller.process_frame.return_value = False

        config = PipelineConfig(window_refresh_interval=0.0)
        pipeline = CapturePipeline(
            config=config,
            window_detector=detector,
            screen_capture=capture,
            frame_poller=poller,
        )

        pipeline._tick()
        assert pipeline.active_window.window_id == 1

        # Second tick: both windows still present, should keep window 1
        pipeline._tick()
        assert pipeline.active_window.window_id == 1

    def test_switches_window_when_active_lost(self) -> None:
        """Pipeline switches to a new window when the active one disappears."""
        window1 = _make_window(window_id=1, title="Table 1")
        window2 = _make_window(window_id=2, title="Table 2")
        frame = _make_frame()

        detector = MagicMock(spec=WindowDetector)
        detector.detect_windows.return_value = [window1, window2]

        capture = MagicMock(spec=ScreenCapture)
        capture.capture_frame.return_value = frame

        poller = MagicMock(spec=FramePoller)
        poller.process_frame.return_value = False
        poller.reset = MagicMock()

        config = PipelineConfig(window_refresh_interval=0.0)
        pipeline = CapturePipeline(
            config=config,
            window_detector=detector,
            screen_capture=capture,
            frame_poller=poller,
        )

        pipeline._tick()
        assert pipeline.active_window.window_id == 1

        # Window 1 disappears
        detector.detect_windows.return_value = [window2]
        pipeline._tick()
        assert pipeline.active_window.window_id == 2
        poller.reset.assert_called()
