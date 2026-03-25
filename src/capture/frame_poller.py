"""Smart frame polling with change detection.

Polls screen captures at a configurable interval, detecting
frame changes to avoid redundant processing.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np

from src.capture.screen_capture import ScreenCapture

logger = logging.getLogger(__name__)


class FramePoller:
    """Polls for screen captures at a fixed interval with change detection.

    Captures frames from ScreenCapture and invokes a callback only when
    the frame has changed significantly from the previous capture.

    Args:
        screen_capture: The ScreenCapture instance to poll.
        poll_interval_ms: Milliseconds between capture attempts.
        change_threshold: Fraction of pixels that must differ to trigger callback.
    """

    def __init__(
        self,
        screen_capture: ScreenCapture,
        poll_interval_ms: int = 200,
        change_threshold: float = 0.01,
    ) -> None:
        self._capture = screen_capture
        self._poll_interval_s = poll_interval_ms / 1000.0
        self._change_threshold = change_threshold
        self._callback: Optional[Callable[[np.ndarray], None]] = None
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None
        self._last_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """Whether the poller is currently running."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Whether the poller is paused."""
        return self._paused

    def set_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set the callback invoked when a new frame is captured.

        Args:
            callback: Function accepting a BGR numpy array.
        """
        self._callback = callback

    def start(self) -> None:
        """Start polling for frames in a background thread."""
        if self._running:
            logger.warning("FramePoller already running")
            return

        self._running = True
        self._paused = False
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="frame-poller"
        )
        self._thread.start()
        logger.info(
            "FramePoller started (interval=%.0fms, threshold=%.3f)",
            self._poll_interval_s * 1000,
            self._change_threshold,
        )

    def stop(self) -> None:
        """Stop the polling thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._last_frame = None
        logger.info("FramePoller stopped")

    def pause(self) -> None:
        """Pause frame polling without stopping the thread."""
        self._paused = True
        logger.info("FramePoller paused")

    def resume(self) -> None:
        """Resume frame polling after a pause."""
        self._paused = False
        logger.info("FramePoller resumed")

    def has_frame_changed(
        self, new_frame: np.ndarray, old_frame: Optional[np.ndarray]
    ) -> bool:
        """Determine whether two frames differ significantly.

        Args:
            new_frame: The newly captured frame.
            old_frame: The previous frame, or None if this is the first capture.

        Returns:
            True if the frames differ above the change threshold.
        """
        if old_frame is None:
            return True
        if new_frame.shape != old_frame.shape:
            return True

        diff = np.abs(new_frame.astype(np.int16) - old_frame.astype(np.int16))
        changed_pixels = np.count_nonzero(diff.sum(axis=2) > 10)
        total_pixels = new_frame.shape[0] * new_frame.shape[1]

        if total_pixels == 0:
            return False

        change_ratio = changed_pixels / total_pixels
        return change_ratio >= self._change_threshold

    def _poll_loop(self) -> None:
        """Internal polling loop running in a background thread."""
        while self._running:
            if not self._paused:
                try:
                    frame = self._capture.capture_frame()
                    with self._lock:
                        if self.has_frame_changed(frame, self._last_frame):
                            self._last_frame = frame.copy()
                            if self._callback is not None:
                                self._callback(frame)
                except Exception:
                    logger.exception("Error during frame capture")

            time.sleep(self._poll_interval_s)
