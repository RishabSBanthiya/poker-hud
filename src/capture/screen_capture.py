"""Core screen capture module using macOS ScreenCaptureKit via PyObjC.

Provides a ScreenCapture class that captures screen frames as BGR numpy arrays,
compatible with OpenCV processing pipelines. Requires macOS 12.3+ and Screen
Recording permission.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CaptureRegion:
    """Defines a rectangular capture region on screen.

    Attributes:
        x: Horizontal offset in pixels from the left edge.
        y: Vertical offset in pixels from the top edge.
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
        if self.x < 0 or self.y < 0:
            raise ValueError(
                f"Capture region offsets must be non-negative, "
                f"got x={self.x}, y={self.y}"
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
    ) -> None:
        self._region = region
        self._display_id = display_id

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

        Returns:
            A numpy array with shape (height, width, 3), dtype uint8, in BGR
            color order (OpenCV-compatible).

        Raises:
            ScreenRecordingPermissionError: If Screen Recording permission is
                not granted in macOS System Preferences.
            ScreenCaptureError: If capture fails for any other reason.
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

        display = self._get_display(SCShareableContent)
        config = self._build_config(SCStreamConfiguration, display)
        raw_frame = self._capture_raw_frame(display, config)
        return self._convert_to_bgr(raw_frame)

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

        if not content_ready.wait(timeout=10.0):
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

        Args:
            sc_stream_config_cls: The SCStreamConfiguration class.
            display: The target SCDisplay object.

        Returns:
            A configured SCStreamConfiguration object.
        """
        config = sc_stream_config_cls.alloc().init()

        if self._region is not None:
            config.setWidth_(self._region.width)
            config.setHeight_(self._region.height)
            config.setSourceRect_(
                _make_cg_rect(
                    self._region.x,
                    self._region.y,
                    self._region.width,
                    self._region.height,
                )
            )
        else:
            config.setWidth_(display.width())
            config.setHeight_(display.height())

        config.setPixelFormat_(0x42475241)  # kCVPixelFormatType_32BGRA
        config.setShowsCursor_(False)

        return config

    def _capture_raw_frame(self, display: object, config: object) -> object:
        """Capture a single raw frame using SCStream.

        Args:
            display: The SCDisplay to capture.
            config: The SCStreamConfiguration to use.

        Returns:
            An SCStreamOutput sample buffer containing the frame data.

        Raises:
            ScreenCaptureError: If the capture operation fails.
        """
        from Quartz import CGMainDisplayID  # type: ignore[import-untyped]
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

        SCScreenshotManager.captureSampleBufferWithFilter_configuration_completionHandler_(
            content_filter, config, screenshot_handler
        )

        if not frame_ready.wait(timeout=10.0):
            raise ScreenCaptureError("Timed out waiting for frame capture")

        if result["error"] is not None:
            raise ScreenCaptureError(
                f"Frame capture failed: {result['error']}"
            )

        if result["image"] is None:
            raise ScreenCaptureError("Frame capture returned None")

        return result["image"]

    def _convert_to_bgr(self, sample_buffer: object) -> np.ndarray:
        """Convert a CMSampleBuffer to a BGR numpy array.

        Args:
            sample_buffer: A CMSampleBuffer from SCScreenshotManager.

        Returns:
            A numpy array with shape (height, width, 3), dtype uint8, BGR order.

        Raises:
            ScreenCaptureError: If conversion fails.
        """
        import CoreMedia  # type: ignore[import-untyped]
        import CoreVideo  # type: ignore[import-untyped]

        try:
            pixel_buffer = CoreMedia.CMSampleBufferGetImageBuffer(
                sample_buffer
            )
            if pixel_buffer is None:
                raise ScreenCaptureError(
                    "Failed to get pixel buffer from sample buffer"
                )

            CoreVideo.CVPixelBufferLockBaseAddress(pixel_buffer, 0)
            try:
                base_address = CoreVideo.CVPixelBufferGetBaseAddress(
                    pixel_buffer
                )
                width = CoreVideo.CVPixelBufferGetWidth(pixel_buffer)
                height = CoreVideo.CVPixelBufferGetHeight(pixel_buffer)
                bytes_per_row = CoreVideo.CVPixelBufferGetBytesPerRow(
                    pixel_buffer
                )

                if base_address is None:
                    raise ScreenCaptureError(
                        "Pixel buffer base address is None"
                    )

                # Create numpy array from the raw buffer (BGRA format)
                buf = (
                    np.frombuffer(base_address, dtype=np.uint8)
                    .reshape(height, bytes_per_row // 1)[:, : width * 4]
                    .reshape(height, width, 4)
                )

                # Drop alpha channel: BGRA -> BGR
                bgr_frame = buf[:, :, :3].copy()

            finally:
                CoreVideo.CVPixelBufferUnlockBaseAddress(pixel_buffer, 0)

            return bgr_frame

        except ScreenCaptureError:
            raise
        except Exception as exc:
            raise ScreenCaptureError(
                f"Failed to convert sample buffer to BGR array: {exc}"
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
