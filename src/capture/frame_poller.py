"""Smart polling with frame change detection.

Compares consecutive captured frames to detect meaningful changes,
avoiding redundant processing of identical frames. Uses lightweight
numpy-based frame differencing with a configurable threshold.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default change detection parameters.
_DEFAULT_POLLING_INTERVAL = 0.2  # 200ms
_DEFAULT_CHANGE_THRESHOLD = 5.0  # Mean absolute difference threshold
_DEFAULT_CHANGE_PIXEL_RATIO = 0.01  # 1% of pixels must change


@dataclass
class FrameStats:
    """Statistics about the frame polling process.

    Attributes:
        total_captured: Total number of frames captured from the source.
        changes_detected: Number of frames where meaningful change was detected.
        frames_skipped: Number of frames skipped due to no change.
        last_change_time: Timestamp of the last detected change, or None.
    """

    total_captured: int = 0
    changes_detected: int = 0
    frames_skipped: int = 0
    last_change_time: Optional[float] = None

    @property
    def change_rate(self) -> float:
        """Fraction of captured frames that had changes (0.0 to 1.0)."""
        if self.total_captured == 0:
            return 0.0
        return self.changes_detected / self.total_captured

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.total_captured = 0
        self.changes_detected = 0
        self.frames_skipped = 0
        self.last_change_time = None


class FramePoller:
    """Polls for frame changes using lightweight differencing.

    Compares each new frame to the previously accepted frame using
    mean absolute difference (MAD) and a pixel change ratio. Only
    frames with meaningful visual changes are emitted.

    Args:
        polling_interval: Minimum time between polls in seconds.
        change_threshold: Per-pixel mean absolute difference threshold.
            A higher value requires more dramatic changes to trigger.
        change_pixel_ratio: Fraction of pixels that must exceed the
            threshold for the frame to be considered changed.
        downscale_factor: Factor to downscale frames before comparison.
            Higher values trade accuracy for speed. Must be >= 1.
    """

    def __init__(
        self,
        polling_interval: float = _DEFAULT_POLLING_INTERVAL,
        change_threshold: float = _DEFAULT_CHANGE_THRESHOLD,
        change_pixel_ratio: float = _DEFAULT_CHANGE_PIXEL_RATIO,
        downscale_factor: int = 4,
    ) -> None:
        if polling_interval < 0:
            raise ValueError(
                f"polling_interval must be non-negative, got {polling_interval}"
            )
        if change_threshold < 0:
            raise ValueError(
                f"change_threshold must be non-negative, got {change_threshold}"
            )
        if not 0.0 <= change_pixel_ratio <= 1.0:
            raise ValueError(
                f"change_pixel_ratio must be between 0 and 1, "
                f"got {change_pixel_ratio}"
            )
        if downscale_factor < 1:
            raise ValueError(
                f"downscale_factor must be >= 1, got {downscale_factor}"
            )

        self._polling_interval = polling_interval
        self._change_threshold = change_threshold
        self._change_pixel_ratio = change_pixel_ratio
        self._downscale_factor = downscale_factor
        self._previous_frame: Optional[np.ndarray] = None
        self._last_poll_time: float = 0.0
        self._stats = FrameStats()

    @property
    def polling_interval(self) -> float:
        """The minimum time between polls in seconds."""
        return self._polling_interval

    @polling_interval.setter
    def polling_interval(self, value: float) -> None:
        """Update the polling interval.

        Args:
            value: New interval in seconds, must be non-negative.
        """
        if value < 0:
            raise ValueError(
                f"polling_interval must be non-negative, got {value}"
            )
        self._polling_interval = value

    @property
    def stats(self) -> FrameStats:
        """Current frame polling statistics."""
        return self._stats

    def should_poll(self) -> bool:
        """Check whether enough time has elapsed for the next poll.

        Returns:
            True if the polling interval has elapsed since the last poll.
        """
        return (time.monotonic() - self._last_poll_time) >= self._polling_interval

    def process_frame(self, frame: np.ndarray) -> bool:
        """Process a new frame and determine if it has meaningful changes.

        Updates internal statistics and stores the frame as the new
        reference if a change is detected.

        Args:
            frame: A BGR numpy array (height, width, 3), dtype uint8.

        Returns:
            True if the frame has meaningful changes compared to the
            previous frame, False otherwise. The first frame always
            returns True.
        """
        self._last_poll_time = time.monotonic()
        self._stats.total_captured += 1

        if self._previous_frame is None:
            self._previous_frame = frame.copy()
            self._stats.changes_detected += 1
            self._stats.last_change_time = time.monotonic()
            return True

        has_changed = self._detect_change(self._previous_frame, frame)

        if has_changed:
            self._previous_frame = frame.copy()
            self._stats.changes_detected += 1
            self._stats.last_change_time = time.monotonic()
        else:
            self._stats.frames_skipped += 1

        return has_changed

    def reset(self) -> None:
        """Reset the poller state, clearing the reference frame and stats."""
        self._previous_frame = None
        self._last_poll_time = 0.0
        self._stats.reset()

    def _detect_change(
        self, prev: np.ndarray, curr: np.ndarray
    ) -> bool:
        """Detect whether two frames differ meaningfully.

        Uses downscaled frame differencing for performance. Computes the
        per-pixel absolute difference and checks if a sufficient fraction
        of pixels exceed the threshold.

        Args:
            prev: The previous reference frame (BGR, uint8).
            curr: The current frame to compare (BGR, uint8).

        Returns:
            True if meaningful change is detected.
        """
        if prev.shape != curr.shape:
            return True

        # Downscale for faster comparison
        p = prev[:: self._downscale_factor, :: self._downscale_factor]
        c = curr[:: self._downscale_factor, :: self._downscale_factor]

        # Compute per-pixel mean absolute difference across channels
        diff = np.abs(p.astype(np.int16) - c.astype(np.int16))
        pixel_diff = diff.mean(axis=2)  # Mean across BGR channels

        # Count pixels that exceed the threshold
        changed_pixels = np.count_nonzero(
            pixel_diff > self._change_threshold
        )
        total_pixels = pixel_diff.size

        ratio = changed_pixels / total_pixels if total_pixels > 0 else 0.0
        return bool(ratio >= self._change_pixel_ratio)
