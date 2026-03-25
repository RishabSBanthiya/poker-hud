"""Capture pipeline coordinator.

Orchestrates window detection, screen capture, and frame change detection
into a continuous pipeline that emits changed frames via callbacks.
Runs in a background thread with start/stop/pause controls and automatic
error recovery.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np

from src.capture.frame_poller import FramePoller, FrameStats
from src.capture.screen_capture import (
    CaptureRegion,
    ScreenCapture,
    ScreenCaptureError,
    ScreenRecordingPermissionError,
)
from src.capture.window_detector import (
    WindowDetectionError,
    WindowDetector,
    WindowInfo,
)

logger = logging.getLogger(__name__)

# Type alias for frame handler callbacks.
FrameHandler = Callable[[np.ndarray, WindowInfo], None]


class PipelineState(Enum):
    """States of the capture pipeline."""

    STOPPED = auto()
    RUNNING = auto()
    PAUSED = auto()
    ERROR = auto()


@dataclass
class PipelineConfig:
    """Configuration for the capture pipeline.

    Attributes:
        polling_interval: Time between frame captures in seconds.
        change_threshold: Frame difference threshold for change detection.
        change_pixel_ratio: Fraction of pixels that must change.
        window_refresh_interval: How often to re-scan for windows (seconds).
        max_consecutive_errors: Errors before entering ERROR state.
        error_recovery_delay: Seconds to wait before retrying after errors.
    """

    polling_interval: float = 0.2
    change_threshold: float = 5.0
    change_pixel_ratio: float = 0.01
    window_refresh_interval: float = 5.0
    max_consecutive_errors: int = 10
    error_recovery_delay: float = 1.0


class CapturePipeline:
    """Continuous capture pipeline with change detection and callbacks.

    Orchestrates the full capture flow:
    1. Detect poker table windows
    2. Capture screen region for each window
    3. Detect frame changes
    4. Emit changed frames to registered handlers

    The pipeline runs in a background thread and supports start, stop,
    and pause controls. It automatically recovers from transient errors
    such as a poker window being closed and reopened.

    Args:
        config: Pipeline configuration. Uses defaults if not provided.
        window_detector: Optional pre-configured WindowDetector instance.
        screen_capture: Optional pre-configured ScreenCapture instance.
        frame_poller: Optional pre-configured FramePoller instance.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        window_detector: Optional[WindowDetector] = None,
        screen_capture: Optional[ScreenCapture] = None,
        frame_poller: Optional[FramePoller] = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._detector = window_detector or WindowDetector()
        self._capture = screen_capture or ScreenCapture()
        self._poller = frame_poller or FramePoller(
            polling_interval=self._config.polling_interval,
            change_threshold=self._config.change_threshold,
            change_pixel_ratio=self._config.change_pixel_ratio,
        )

        self._handlers: list[FrameHandler] = []
        self._state = PipelineState.STOPPED
        self._state_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially

        self._active_window: Optional[WindowInfo] = None
        self._consecutive_errors = 0
        self._last_window_refresh: float = 0.0

    @property
    def state(self) -> PipelineState:
        """Current pipeline state."""
        with self._state_lock:
            return self._state

    @property
    def active_window(self) -> Optional[WindowInfo]:
        """The currently tracked poker window, if any."""
        return self._active_window

    @property
    def frame_stats(self) -> FrameStats:
        """Frame polling statistics from the current session."""
        return self._poller.stats

    def register_handler(self, handler: FrameHandler) -> None:
        """Register a callback to receive changed frames.

        Args:
            handler: A callable that accepts (frame, window_info).
                frame is a BGR numpy array, window_info describes
                the source window.
        """
        if handler not in self._handlers:
            self._handlers.append(handler)
            logger.debug("Registered frame handler: %s", handler)

    def unregister_handler(self, handler: FrameHandler) -> None:
        """Remove a previously registered frame handler.

        Args:
            handler: The handler to remove.

        Raises:
            ValueError: If the handler was not registered.
        """
        self._handlers.remove(handler)
        logger.debug("Unregistered frame handler: %s", handler)

    def start(self) -> None:
        """Start the capture pipeline in a background thread.

        Raises:
            RuntimeError: If the pipeline is already running.
        """
        with self._state_lock:
            if self._state == PipelineState.RUNNING:
                raise RuntimeError("Pipeline is already running")

            self._stop_event.clear()
            self._pause_event.set()
            self._state = PipelineState.RUNNING
            self._consecutive_errors = 0

        self._thread = threading.Thread(
            target=self._run_loop,
            name="capture-pipeline",
            daemon=True,
        )
        self._thread.start()
        logger.info("Capture pipeline started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the capture pipeline and wait for the thread to finish.

        Args:
            timeout: Maximum seconds to wait for the thread to join.
        """
        with self._state_lock:
            if self._state == PipelineState.STOPPED:
                return
            self._state = PipelineState.STOPPED

        self._stop_event.set()
        self._pause_event.set()  # Unblock if paused

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    "Pipeline thread did not stop within %.1fs", timeout
                )

        self._thread = None
        self._active_window = None
        self._poller.reset()
        logger.info("Capture pipeline stopped")

    def pause(self) -> None:
        """Pause frame capture. The pipeline thread stays alive but idle."""
        with self._state_lock:
            if self._state != PipelineState.RUNNING:
                return
            self._state = PipelineState.PAUSED

        self._pause_event.clear()
        logger.info("Capture pipeline paused")

    def resume(self) -> None:
        """Resume a paused pipeline."""
        with self._state_lock:
            if self._state != PipelineState.PAUSED:
                return
            self._state = PipelineState.RUNNING

        self._pause_event.set()
        logger.info("Capture pipeline resumed")

    def _run_loop(self) -> None:
        """Main loop executed in the background thread."""
        logger.debug("Pipeline loop starting")

        while not self._stop_event.is_set():
            # Block while paused
            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            try:
                self._tick()
                self._consecutive_errors = 0
            except ScreenRecordingPermissionError:
                logger.error(
                    "Screen Recording permission denied — stopping pipeline"
                )
                with self._state_lock:
                    self._state = PipelineState.ERROR
                return
            except (ScreenCaptureError, WindowDetectionError) as exc:
                self._consecutive_errors += 1
                logger.warning(
                    "Pipeline error (%d/%d): %s",
                    self._consecutive_errors,
                    self._config.max_consecutive_errors,
                    exc,
                )
                if (
                    self._consecutive_errors
                    >= self._config.max_consecutive_errors
                ):
                    logger.error(
                        "Max consecutive errors reached — entering ERROR state"
                    )
                    with self._state_lock:
                        self._state = PipelineState.ERROR
                    return
                self._stop_event.wait(self._config.error_recovery_delay)
                continue
            except Exception:
                logger.exception("Unexpected error in pipeline loop")
                with self._state_lock:
                    self._state = PipelineState.ERROR
                return

            # Wait for the next polling interval
            self._stop_event.wait(self._config.polling_interval)

        logger.debug("Pipeline loop exiting")

    def _tick(self) -> None:
        """Execute a single capture cycle.

        Refreshes windows if needed, captures a frame from the active
        window, checks for changes, and dispatches to handlers.
        """
        # Refresh windows periodically
        now = time.monotonic()
        if (
            self._active_window is None
            or (now - self._last_window_refresh)
            >= self._config.window_refresh_interval
        ):
            self._refresh_active_window()
            self._last_window_refresh = now

        if self._active_window is None:
            logger.debug("No poker window found, skipping tick")
            return

        # Configure capture region to match the active window
        win = self._active_window
        self._capture.region = CaptureRegion(
            x=win.x, y=win.y, width=win.width, height=win.height
        )

        frame = self._capture.capture_frame()

        if self._poller.process_frame(frame):
            self._dispatch_frame(frame, win)

    def _refresh_active_window(self) -> None:
        """Re-detect poker windows and select the active one.

        If the currently active window is still present, keep tracking it.
        Otherwise, select the first available poker window.
        """
        try:
            windows = self._detector.detect_windows(force_refresh=True)
        except WindowDetectionError as exc:
            logger.warning("Window detection failed: %s", exc)
            return

        if not windows:
            if self._active_window is not None:
                logger.info(
                    "Active window '%s' is no longer available",
                    self._active_window.title,
                )
                self._active_window = None
                self._poller.reset()
            return

        # Try to keep tracking the current window
        if self._active_window is not None:
            for win in windows:
                if win.window_id == self._active_window.window_id:
                    self._active_window = win  # Update bounds
                    return

            logger.info(
                "Active window '%s' lost, switching to '%s'",
                self._active_window.title,
                windows[0].title,
            )
            self._poller.reset()

        self._active_window = windows[0]
        logger.info(
            "Tracking poker window: '%s' (id=%d, %dx%d)",
            self._active_window.title,
            self._active_window.window_id,
            self._active_window.width,
            self._active_window.height,
        )

    def _dispatch_frame(
        self, frame: np.ndarray, window: WindowInfo
    ) -> None:
        """Send a changed frame to all registered handlers.

        Args:
            frame: The changed BGR frame.
            window: Info about the source window.
        """
        for handler in self._handlers:
            try:
                handler(frame, window)
            except Exception:
                logger.exception(
                    "Error in frame handler %s", handler
                )
