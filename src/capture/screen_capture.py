"""Core screen capture module using macOS ScreenCaptureKit via PyObjC.

Provides a ScreenCapture class that captures screen frames as BGR numpy arrays,
compatible with OpenCV processing pipelines. Requires macOS 12.3+ and Screen
Recording permission.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from Foundation import NSDate, NSRunLoop  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 0.1


def _wait_with_runloop(event: threading.Event, timeout: float) -> bool:
    """Wait for a threading.Event while spinning the NSRunLoop.

    ScreenCaptureKit completion handlers are dispatched via GCD.  When the
    calling thread has no active run loop (e.g. a plain ``threading.Thread``),
    the callbacks may never be delivered — causing a hang or, on Apple Silicon,
    a SIGTRAP (trace trap).  Spinning the NSRunLoop in short increments lets
    the GCD callbacks land on this thread.

    Args:
        event: The event to wait for.
        timeout: Maximum seconds to wait.

    Returns:
        True if the event was set before the timeout, False otherwise.
    """
    deadline = time.monotonic() + timeout
    while not event.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        NSRunLoop.currentRunLoop().runUntilDate_(
            NSDate.dateWithTimeIntervalSinceNow_(min(0.05, remaining))
        )
    return True


@dataclass
class CaptureRegion:
    """Defines a rectangular capture region on screen.

    Attributes:
        x: Horizontal position in global screen coordinates. May be negative
           on multi-monitor setups where a display sits left of the primary.
        y: Vertical position in global screen coordinates. May be negative
           on multi-monitor setups where a display sits above the primary.
        width: Width of the capture region in pixels.
        height: Height of the capture region in pixels.
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Capture region dimensions must be positive, "
                f"got width={self.width}, height={self.height}"
            )


class ScreenCaptureError(Exception):
    """Base exception for screen capture errors."""


class ScreenRecordingPermissionError(ScreenCaptureError):
    """Raised when macOS Screen Recording permission is not granted."""


class ScreenCapture:
    """Captures screen frames using macOS ScreenCaptureKit.

    Uses SCShareableContent, SCStreamConfiguration, and SCStream to capture
    display content and convert it to BGR numpy arrays for OpenCV processing.

    Args:
        region: Optional capture region. If None, captures the full primary display.
        display_id: Optional display ID to capture. If None, uses the primary display.
        max_retries: Number of retry attempts on transient failures.

    Example:
        >>> capture = ScreenCapture()
        >>> frame = capture.capture_frame()
        >>> print(frame.shape)  # (height, width, 3)
        >>> print(frame.dtype)  # uint8
    """

    def __init__(
        self,
        region: Optional[CaptureRegion] = None,
        display_id: Optional[int] = None,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        self._region = region
        self._display_id = display_id
        self._max_retries = max_retries

    @property
    def region(self) -> Optional[CaptureRegion]:
        """The configured capture region, or None for full display."""
        return self._region

    @region.setter
    def region(self, value: Optional[CaptureRegion]) -> None:
        """Update the capture region.

        Args:
            value: New capture region, or None for full display.
        """
        self._region = value

    def capture_frame(self) -> np.ndarray:
        """Capture a single frame from the screen.

        Includes retry logic for transient failures. Permission errors are
        raised immediately without retrying.

        Returns:
            A numpy array with shape (height, width, 3), dtype uint8, in BGR
            color order (OpenCV-compatible).

        Raises:
            ScreenRecordingPermissionError: If Screen Recording permission is
                not granted in macOS System Preferences.
            ScreenCaptureError: If capture fails after all retries.
        """
        try:
            from ScreenCaptureKit import (  # type: ignore[import-untyped]
                SCShareableContent,
                SCStreamConfiguration,
            )
        except ImportError as exc:
            raise ScreenCaptureError(
                "ScreenCaptureKit PyObjC bindings not available. "
                "Install with: pip install pyobjc-framework-ScreenCaptureKit"
            ) from exc

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                display = self._get_display(SCShareableContent)
                config = self._build_config(SCStreamConfiguration, display)
                raw_frame = self._capture_raw_frame(display, config)
                frame = self._convert_to_bgr(raw_frame)

                if self._region is not None:
                    h, w = frame.shape[:2]
                    # Only crop if the frame is larger than the
                    # requested region (i.e. ScreenCaptureKit did
                    # not apply sourceRect for us).
                    if (
                        w != self._region.width
                        or h != self._region.height
                    ):
                        frame = self._crop_to_region(frame)

                return frame
            except ScreenRecordingPermissionError:
                raise
            except ScreenCaptureError as exc:
                last_error = exc
                logger.warning(
                    "Capture attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries,
                    exc,
                )
                if attempt < self._max_retries - 1:
                    time.sleep(_RETRY_DELAY_SECONDS)

        raise ScreenCaptureError(
            f"Capture failed after {self._max_retries} attempts: {last_error}"
        )

    def _crop_to_region(self, frame: np.ndarray) -> np.ndarray:
        """Crop a full-display frame to the configured region.

        This is a fallback for when ScreenCaptureKit's sourceRect does
        not crop as expected. Ensures the returned frame matches the
        requested region dimensions.

        Args:
            frame: The full-display BGR frame.

        Returns:
            The cropped frame matching the configured region.

        Raises:
            ScreenCaptureError: If the region is out of bounds.
        """
        if self._region is None:
            return frame

        h, w = frame.shape[:2]
        r = self._region
        if r.x + r.width > w or r.y + r.height > h:
            logger.warning(
                "Region %s extends beyond frame bounds (%dx%d), "
                "clamping to available area",
                r,
                w,
                h,
            )
            x_end = min(r.x + r.width, w)
            y_end = min(r.y + r.height, h)
            return frame[r.y : y_end, r.x : x_end].copy()

        return frame[r.y : r.y + r.height, r.x : r.x + r.width].copy()

    def _get_display(self, sc_shareable_content_cls: type) -> object:
        """Retrieve the target display via SCShareableContent.

        Args:
            sc_shareable_content_cls: The SCShareableContent class from
                ScreenCaptureKit.

        Returns:
            An SCDisplay object representing the target display.

        Raises:
            ScreenRecordingPermissionError: If permission is denied.
            ScreenCaptureError: If no displays are found.
        """
        content_ready = threading.Event()
        result: dict = {"content": None, "error": None}

        def handler(content: object, error: object) -> None:
            result["content"] = content
            result["error"] = error
            content_ready.set()

        sc_shareable_content_cls.getShareableContentWithCompletionHandler_(handler)

        if not _wait_with_runloop(content_ready, timeout=10.0):
            raise ScreenCaptureError(
                "Timed out waiting for SCShareableContent response"
            )

        if result["error"] is not None:
            error_desc = str(result["error"])
            if "permission" in error_desc.lower() or "denied" in error_desc.lower():
                raise ScreenRecordingPermissionError(
                    "Screen Recording permission not granted. "
                    "Enable it in System Preferences > Privacy & Security > "
                    "Screen Recording."
                )
            raise ScreenCaptureError(
                f"Failed to get shareable content: {error_desc}"
            )

        content = result["content"]
        if content is None:
            raise ScreenCaptureError("SCShareableContent returned None")

        displays = content.displays()
        if not displays or len(displays) == 0:
            raise ScreenCaptureError("No displays found")

        if self._display_id is not None:
            for display in displays:
                if display.displayID() == self._display_id:
                    return display
            raise ScreenCaptureError(
                f"Display with ID {self._display_id} not found. "
                f"Available: {[d.displayID() for d in displays]}"
            )

        return displays[0]

    def _build_config(
        self, sc_stream_config_cls: type, display: object
    ) -> object:
        """Build an SCStreamConfiguration for the capture.

        When a region is specified, sourceRect coordinates must be relative
        to the display origin (not global screen coordinates). The output
        width/height must also account for the display's pixel scale factor
        (Retina displays report 2x).

        Args:
            sc_stream_config_cls: The SCStreamConfiguration class.
            display: The target SCDisplay object.

        Returns:
            A configured SCStreamConfiguration object.
        """
        config = sc_stream_config_cls.alloc().init()

        # Determine the display's scale factor (Retina = 2, standard = 1).
        display_width_pt = display.width()
        display_height_pt = display.height()

        if self._region is not None:
            # Convert global screen coordinates to display-relative.
            # Display frame origin comes from the display object.
            try:
                display_frame = display.frame()
                display_x = int(display_frame.origin.x)
                display_y = int(display_frame.origin.y)
            except (AttributeError, TypeError):
                # Fallback: assume primary display at (0, 0)
                display_x = 0
                display_y = 0

            rel_x = self._region.x - display_x
            rel_y = self._region.y - display_y

            # Clamp sourceRect to stay within display bounds (in points).
            src_x = max(0, rel_x)
            src_y = max(0, rel_y)
            src_w = min(self._region.width, display_width_pt - src_x)
            src_h = min(self._region.height, display_height_pt - src_y)

            if src_w <= 0 or src_h <= 0:
                raise ScreenCaptureError(
                    f"Region {self._region} is entirely outside display "
                    f"bounds ({display_width_pt}x{display_height_pt} at "
                    f"{display_x},{display_y})"
                )

            # Output dimensions match source rect (in points — SCK scales
            # automatically to native resolution).
            config.setWidth_(src_w)
            config.setHeight_(src_h)
            config.setSourceRect_(
                _make_cg_rect(src_x, src_y, src_w, src_h)
            )
        else:
            config.setWidth_(display_width_pt)
            config.setHeight_(display_height_pt)

        config.setPixelFormat_(0x42475241)  # kCVPixelFormatType_32BGRA
        config.setShowsCursor_(False)

        return config

    def _capture_raw_frame(self, display: object, config: object) -> object:
        """Capture a single raw frame as a CGImage.

        Uses SCScreenshotManager.captureImageWithFilter which returns a
        CGImage directly — more reliable than captureSampleBufferWithFilter
        which can return empty CMSampleBuffers.

        Args:
            display: The SCDisplay to capture.
            config: The SCStreamConfiguration to use.

        Returns:
            A CGImage containing the captured frame.

        Raises:
            ScreenCaptureError: If the capture operation fails.
        """
        from ScreenCaptureKit import (  # type: ignore[import-untyped]
            SCContentFilter,
            SCScreenshotManager,
        )

        content_filter = (
            SCContentFilter.alloc()
            .initWithDisplay_excludingWindows_(display, [])
        )

        frame_ready = threading.Event()
        result: dict = {"image": None, "error": None}

        def screenshot_handler(image: object, error: object) -> None:
            result["image"] = image
            result["error"] = error
            frame_ready.set()

        SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_(
            content_filter, config, screenshot_handler
        )

        if not _wait_with_runloop(frame_ready, timeout=10.0):
            raise ScreenCaptureError("Timed out waiting for frame capture")

        if result["error"] is not None:
            raise ScreenCaptureError(
                f"Frame capture failed: {result['error']}"
            )

        if result["image"] is None:
            raise ScreenCaptureError("Frame capture returned None")

        return result["image"]

    def _convert_to_bgr(self, cg_image: object) -> np.ndarray:
        """Convert a CGImage to a BGR numpy array.

        Args:
            cg_image: A CGImage from SCScreenshotManager.

        Returns:
            A numpy array with shape (height, width, 3), dtype uint8, BGR order.

        Raises:
            ScreenCaptureError: If conversion fails.
        """
        import Quartz  # type: ignore[import-untyped]

        try:
            width = Quartz.CGImageGetWidth(cg_image)
            height = Quartz.CGImageGetHeight(cg_image)

            if width == 0 or height == 0:
                raise ScreenCaptureError(
                    f"CGImage has zero dimensions: {width}x{height}"
                )

            # Create a bitmap context to render the CGImage into BGRA
            bytes_per_row = width * 4
            color_space = Quartz.CGColorSpaceCreateDeviceRGB()
            # kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little
            # = BGRA pixel format
            bitmap_info = (
                Quartz.kCGImageAlphaPremultipliedFirst
                | Quartz.kCGBitmapByteOrder32Little
            )

            context = Quartz.CGBitmapContextCreate(
                None, width, height, 8, bytes_per_row, color_space, bitmap_info
            )
            if context is None:
                raise ScreenCaptureError(
                    "Failed to create CGBitmapContext"
                )

            # Draw the CGImage into the bitmap context
            rect = Quartz.CGRectMake(0, 0, width, height)
            Quartz.CGContextDrawImage(context, rect, cg_image)

            # Extract pixel data from the context
            data = Quartz.CGBitmapContextGetData(context)
            if data is None:
                raise ScreenCaptureError(
                    "Failed to get bitmap context data"
                )

            total_bytes = bytes_per_row * height
            if hasattr(data, "as_buffer"):
                raw = np.frombuffer(
                    data.as_buffer(total_bytes), dtype=np.uint8
                )
            else:
                raw = np.frombuffer(data, dtype=np.uint8)

            buf = raw.reshape(height, width, 4)

            # Drop alpha channel: BGRA -> BGR
            bgr_frame = buf[:, :, :3].copy()

            return bgr_frame

        except ScreenCaptureError:
            raise
        except Exception as exc:
            raise ScreenCaptureError(
                f"Failed to convert CGImage to BGR array: {exc}"
            ) from exc


def _make_cg_rect(
    x: float, y: float, width: float, height: float
) -> tuple:
    """Create a CGRect-compatible tuple.

    Args:
        x: X origin.
        y: Y origin.
        width: Width.
        height: Height.

    Returns:
        A nested tuple ((x, y), (width, height)) representing a CGRect.
    """
    return ((x, y), (width, height))


def validate_frame(frame: np.ndarray) -> bool:
    """Validate that a frame meets the expected format.

    Checks that the frame is a 3-channel BGR image with uint8 dtype.

    Args:
        frame: The numpy array to validate.

    Returns:
        True if the frame is valid, False otherwise.
    """
    if not isinstance(frame, np.ndarray):
        return False
    if frame.dtype != np.uint8:
        return False
    if frame.ndim != 3:
        return False
    if frame.shape[2] != 3:
        return False
    return True
