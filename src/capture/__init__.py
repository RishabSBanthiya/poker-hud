"""Screen capture subsystem for the Poker HUD.

Provides macOS screen capture via ScreenCaptureKit/PyObjC, returning frames
as BGR numpy arrays compatible with OpenCV.
"""

from src.capture.screen_capture import (
    CaptureRegion,
    ScreenCapture,
    ScreenCaptureError,
    ScreenRecordingPermissionError,
    validate_frame,
)

__all__ = [
    "CaptureRegion",
    "ScreenCapture",
    "ScreenCaptureError",
    "ScreenRecordingPermissionError",
    "validate_frame",
]
