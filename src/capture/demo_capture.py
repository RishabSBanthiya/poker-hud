"""Demo script for screen capture.

Captures a single frame from the primary display and saves it as
capture_demo.png. Prints frame dimensions and dtype information.

Usage:
    python -m src.capture.demo_capture
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from src.capture.screen_capture import (
    ScreenCapture,
    ScreenCaptureError,
    ScreenRecordingPermissionError,
    validate_frame,
)


def main() -> None:
    """Run the screen capture demo."""
    print("Poker HUD - Screen Capture Demo")
    print("=" * 40)

    capture = ScreenCapture()

    try:
        print("Capturing frame from primary display...")
        frame = capture.capture_frame()
    except ScreenRecordingPermissionError:
        print(
            "\nERROR: Screen Recording permission not granted.",
            file=sys.stderr,
        )
        print(
            "Go to System Preferences > Privacy & Security > Screen Recording",
            file=sys.stderr,
        )
        print("and enable access for Terminal / your IDE.", file=sys.stderr)
        sys.exit(1)
    except ScreenCaptureError as exc:
        print(f"\nERROR: Screen capture failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nFrame captured successfully!")
    print(f"  Shape:  {frame.shape} (height, width, channels)")
    print(f"  Dtype:  {frame.dtype}")
    print(f"  Size:   {frame.nbytes / (1024 * 1024):.1f} MB")
    print(f"  Valid:  {validate_frame(frame)}")

    output_path = Path("capture_demo.png")
    cv2.imwrite(str(output_path), frame)
    print(f"\nSaved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
