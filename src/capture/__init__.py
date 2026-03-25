"""Screen capture subsystem for the Poker HUD.

Provides macOS screen capture via ScreenCaptureKit/PyObjC, poker window
auto-detection, smart frame change polling, and a coordinated capture
pipeline. Frames are returned as BGR numpy arrays compatible with OpenCV.
"""

from src.capture.frame_poller import FramePoller, FrameStats
from src.capture.pipeline import (
    CapturePipeline,
    FrameHandler,
    PipelineConfig,
    PipelineState,
)
from src.capture.screen_capture import (
    CaptureRegion,
    ScreenCapture,
    ScreenCaptureError,
    ScreenRecordingPermissionError,
    validate_frame,
)
from src.capture.window_detector import (
    WindowDetectionError,
    WindowDetector,
    WindowInfo,
)

__all__ = [
    "CapturePipeline",
    "CaptureRegion",
    "FrameHandler",
    "FramePoller",
    "FrameStats",
    "PipelineConfig",
    "PipelineState",
    "ScreenCapture",
    "ScreenCaptureError",
    "ScreenRecordingPermissionError",
    "WindowDetectionError",
    "WindowDetector",
    "WindowInfo",
    "validate_frame",
]
