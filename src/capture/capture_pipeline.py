"""Capture pipeline coordinator.

Orchestrates window detection, screen capture, and frame polling
into a single pipeline that delivers frames to downstream consumers.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from src.capture.frame_poller import FramePoller
from src.capture.screen_capture import CaptureRegion, ScreenCapture
from src.capture.window_detector import WindowDetector, WindowInfo

logger = logging.getLogger(__name__)


class CapturePipeline:
    """Coordinates window detection, screen capture, and frame polling.

    The pipeline detects the poker window, configures the capture region,
    and polls for new frames, delivering them via a callback.

    Args:
        window_detector: Detects the poker client window.
        screen_capture: Captures screen frames.
        frame_poller: Polls for frame changes.
    """

    def __init__(
        self,
        window_detector: WindowDetector,
        screen_capture: ScreenCapture,
        frame_poller: FramePoller,
    ) -> None:
        self._window_detector = window_detector
        self._screen_capture = screen_capture
        self._frame_poller = frame_poller
        self._frame_callback: Optional[Callable[[np.ndarray], None]] = None
        self._current_window: Optional[WindowInfo] = None

    @property
    def current_window(self) -> Optional[WindowInfo]:
        """The currently tracked poker window."""
        return self._current_window

    @property
    def is_running(self) -> bool:
        """Whether the capture pipeline is actively polling."""
        return self._frame_poller.is_running

    def set_frame_callback(
        self, callback: Callable[[np.ndarray], None]
    ) -> None:
        """Set the callback for newly captured frames.

        Args:
            callback: Function accepting a BGR numpy array.
        """
        self._frame_callback = callback
        self._frame_poller.set_callback(callback)

    def detect_window(self) -> Optional[WindowInfo]:
        """Attempt to detect the poker client window.

        If found, configures the screen capture region to match the window.

        Returns:
            WindowInfo if a window was found, None otherwise.
        """
        window = self._window_detector.detect()
        if window is not None:
            self._current_window = window
            self._screen_capture.region = CaptureRegion(
                x=window.x,
                y=window.y,
                width=max(window.width, 1),
                height=max(window.height, 1),
            )
            logger.info(
                "Capture region set to window '%s' at (%d, %d) %dx%d",
                window.title,
                window.x,
                window.y,
                window.width,
                window.height,
            )
        return window

    def start(self) -> None:
        """Start the capture pipeline.

        Attempts window detection, then starts frame polling.
        """
        self.detect_window()
        self._frame_poller.start()
        logger.info("Capture pipeline started")

    def stop(self) -> None:
        """Stop the capture pipeline."""
        self._frame_poller.stop()
        self._current_window = None
        logger.info("Capture pipeline stopped")

    def pause(self) -> None:
        """Pause frame capture without stopping."""
        self._frame_poller.pause()

    def resume(self) -> None:
        """Resume frame capture."""
        self._frame_poller.resume()
