"""Unit tests for the screen capture module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.capture.screen_capture import (
    CaptureRegion,
    ScreenCapture,
    ScreenCaptureError,
    ScreenRecordingPermissionError,
    validate_frame,
)


class TestCaptureRegion:
    """Tests for the CaptureRegion dataclass."""

    def test_valid_region(self) -> None:
        """Region with valid positive dimensions is created successfully."""
        region = CaptureRegion(x=100, y=200, width=800, height=600)
        assert region.x == 100
        assert region.y == 200
        assert region.width == 800
        assert region.height == 600

    def test_zero_width_raises(self) -> None:
        """Region with zero width raises ValueError."""
        with pytest.raises(ValueError, match="dimensions must be positive"):
            CaptureRegion(x=0, y=0, width=0, height=600)

    def test_negative_height_raises(self) -> None:
        """Region with negative height raises ValueError."""
        with pytest.raises(ValueError, match="dimensions must be positive"):
            CaptureRegion(x=0, y=0, width=800, height=-1)

    def test_negative_offset_valid_for_multimonitor(self) -> None:
        """Region with negative offsets is valid (multi-monitor setups)."""
        region = CaptureRegion(x=-1849, y=38, width=800, height=600)
        assert region.x == -1849
        assert region.y == 38

    def test_zero_offset_valid(self) -> None:
        """Region with zero offsets is valid."""
        region = CaptureRegion(x=0, y=0, width=1920, height=1080)
        assert region.x == 0
        assert region.y == 0


class TestScreenCaptureInit:
    """Tests for ScreenCapture instantiation."""

    def test_default_init(self) -> None:
        """ScreenCapture can be instantiated with no arguments."""
        capture = ScreenCapture()
        assert capture.region is None

    def test_init_with_region(self) -> None:
        """ScreenCapture accepts a capture region."""
        region = CaptureRegion(x=0, y=0, width=800, height=600)
        capture = ScreenCapture(region=region)
        assert capture.region is region

    def test_init_with_display_id(self) -> None:
        """ScreenCapture accepts a display ID."""
        capture = ScreenCapture(display_id=1)
        assert capture.region is None

    def test_region_setter(self) -> None:
        """Capture region can be updated after initialization."""
        capture = ScreenCapture()
        assert capture.region is None

        region = CaptureRegion(x=50, y=50, width=400, height=300)
        capture.region = region
        assert capture.region is region

        capture.region = None
        assert capture.region is None

    def test_init_with_custom_retries(self) -> None:
        """ScreenCapture accepts a custom max_retries value."""
        capture = ScreenCapture(max_retries=5)
        assert capture._max_retries == 5


class TestValidateFrame:
    """Tests for the validate_frame utility function."""

    def test_valid_bgr_frame(self) -> None:
        """A proper BGR uint8 frame passes validation."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert validate_frame(frame) is True

    def test_wrong_dtype(self) -> None:
        """Frame with float dtype fails validation."""
        frame = np.zeros((480, 640, 3), dtype=np.float32)
        assert validate_frame(frame) is False

    def test_wrong_channels(self) -> None:
        """Frame with 4 channels (BGRA) fails validation."""
        frame = np.zeros((480, 640, 4), dtype=np.uint8)
        assert validate_frame(frame) is False

    def test_grayscale(self) -> None:
        """Grayscale frame (2D array) fails validation."""
        frame = np.zeros((480, 640), dtype=np.uint8)
        assert validate_frame(frame) is False

    def test_not_numpy(self) -> None:
        """Non-numpy input fails validation."""
        assert validate_frame([[0, 0, 0]]) is False  # type: ignore[arg-type]

    def test_single_pixel(self) -> None:
        """A 1x1 BGR frame is valid."""
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        assert validate_frame(frame) is True


class TestScreenCaptureFrame:
    """Tests for the capture_frame method using mocks."""

    def test_missing_screencapturekit_raises(self) -> None:
        """ImportError is wrapped in ScreenCaptureError when bindings missing."""
        capture = ScreenCapture()

        with patch.dict("sys.modules", {"ScreenCaptureKit": None}):
            with pytest.raises(ScreenCaptureError, match="not available"):
                capture.capture_frame()

    def test_capture_returns_bgr_array(self) -> None:
        """capture_frame returns a valid BGR numpy array when mocked end-to-end."""
        capture = ScreenCapture()
        expected_frame = np.random.randint(
            0, 255, (1080, 1920, 3), dtype=np.uint8
        )

        with patch.object(
            capture, "_get_display", return_value=MagicMock()
        ), patch.object(
            capture, "_build_config", return_value=MagicMock()
        ), patch.object(
            capture, "_capture_raw_frame", return_value=MagicMock()
        ), patch.object(
            capture, "_convert_to_bgr", return_value=expected_frame
        ):
            mock_sck = MagicMock()
            with patch.dict(
                "sys.modules",
                {
                    "ScreenCaptureKit": mock_sck,
                },
            ):
                frame = capture.capture_frame()

        assert validate_frame(frame)
        assert frame.shape == (1080, 1920, 3)
        assert frame.dtype == np.uint8
        np.testing.assert_array_equal(frame, expected_frame)

    def test_capture_with_region_returns_correct_size(self) -> None:
        """capture_frame respects the configured region dimensions."""
        region = CaptureRegion(x=100, y=100, width=640, height=480)
        capture = ScreenCapture(region=region)
        expected_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch.object(
            capture, "_get_display", return_value=MagicMock()
        ), patch.object(
            capture, "_build_config", return_value=MagicMock()
        ), patch.object(
            capture, "_capture_raw_frame", return_value=MagicMock()
        ), patch.object(
            capture, "_convert_to_bgr", return_value=expected_frame
        ):
            mock_sck = MagicMock()
            with patch.dict(
                "sys.modules",
                {
                    "ScreenCaptureKit": mock_sck,
                },
            ):
                frame = capture.capture_frame()

        assert frame.shape == (480, 640, 3)

    def test_retry_on_transient_failure(self) -> None:
        """capture_frame retries on transient ScreenCaptureError."""
        capture = ScreenCapture(max_retries=3)
        expected_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        call_count = 0

        def mock_get_display(*args):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ScreenCaptureError("transient error")
            return MagicMock()

        with patch.object(
            capture, "_get_display", side_effect=mock_get_display
        ), patch.object(
            capture, "_build_config", return_value=MagicMock()
        ), patch.object(
            capture, "_capture_raw_frame", return_value=MagicMock()
        ), patch.object(
            capture, "_convert_to_bgr", return_value=expected_frame
        ):
            mock_sck = MagicMock()
            with patch.dict(
                "sys.modules",
                {"ScreenCaptureKit": mock_sck},
            ):
                frame = capture.capture_frame()

        assert validate_frame(frame)
        assert call_count == 3

    def test_permission_error_not_retried(self) -> None:
        """Permission errors are raised immediately without retrying."""
        capture = ScreenCapture(max_retries=3)

        with patch.object(
            capture,
            "_get_display",
            side_effect=ScreenRecordingPermissionError("denied"),
        ):
            mock_sck = MagicMock()
            with patch.dict(
                "sys.modules",
                {"ScreenCaptureKit": mock_sck},
            ):
                with pytest.raises(ScreenRecordingPermissionError):
                    capture.capture_frame()

    def test_all_retries_exhausted_raises(self) -> None:
        """ScreenCaptureError is raised when all retries fail."""
        capture = ScreenCapture(max_retries=2)

        with patch.object(
            capture,
            "_get_display",
            side_effect=ScreenCaptureError("persistent error"),
        ):
            mock_sck = MagicMock()
            with patch.dict(
                "sys.modules",
                {"ScreenCaptureKit": mock_sck},
            ):
                with pytest.raises(ScreenCaptureError, match="2 attempts"):
                    capture.capture_frame()


class TestCropToRegion:
    """Tests for the _crop_to_region fallback method."""

    def test_crop_extracts_correct_subregion(self) -> None:
        """Crop returns the exact pixels from the requested region."""
        capture = ScreenCapture(
            region=CaptureRegion(x=10, y=20, width=50, height=30)
        )
        frame = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
        cropped = capture._crop_to_region(frame)
        assert cropped.shape == (30, 50, 3)
        np.testing.assert_array_equal(cropped, frame[20:50, 10:60])

    def test_crop_clamps_out_of_bounds(self) -> None:
        """Crop clamps when region extends beyond frame."""
        capture = ScreenCapture(
            region=CaptureRegion(x=90, y=90, width=50, height=50)
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cropped = capture._crop_to_region(frame)
        assert cropped.shape == (10, 10, 3)

    def test_crop_noop_without_region(self) -> None:
        """Crop returns the original frame when no region is set."""
        capture = ScreenCapture()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = capture._crop_to_region(frame)
        np.testing.assert_array_equal(result, frame)


@pytest.mark.requires_screen_recording
class TestScreenCaptureLive:
    """Integration tests that require actual Screen Recording permission.

    These tests are skipped by default. Run with:
        pytest -m requires_screen_recording
    """

    def test_live_capture_returns_valid_frame(self) -> None:
        """Live capture from primary display returns a valid BGR frame."""
        capture = ScreenCapture()
        frame = capture.capture_frame()

        assert validate_frame(frame)
        assert frame.shape[0] > 0
        assert frame.shape[1] > 0

    def test_live_capture_with_region(self) -> None:
        """Live capture with a region returns a frame of the expected size."""
        region = CaptureRegion(x=0, y=0, width=200, height=200)
        capture = ScreenCapture(region=region)
        frame = capture.capture_frame()

        assert validate_frame(frame)
        assert frame.shape == (200, 200, 3)
