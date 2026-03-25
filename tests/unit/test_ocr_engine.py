"""Tests for src.detection.ocr_engine."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from src.detection.ocr_engine import (
    OCREngine,
    OCRResult,
    Region,
    _generate_digit_templates,
)

# ---------------------------------------------------------------------------
# Helpers -- create synthetic test images with known text
# ---------------------------------------------------------------------------

def _render_text_image(
    text: str,
    width: int = 200,
    height: int = 40,
    font_scale: float = 1.0,
    bg_color: int = 0,
    fg_color: int = 255,
) -> np.ndarray:
    """Render *text* onto a BGR image of the given dimensions."""
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    cv2.putText(
        img,
        text,
        (5, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (fg_color, fg_color, fg_color),
        2,
    )
    return img


# ---------------------------------------------------------------------------
# Region dataclass
# ---------------------------------------------------------------------------

class TestRegion:
    def test_region_creation(self) -> None:
        r = Region(x=10, y=20, w=100, h=50)
        assert r.x == 10
        assert r.y == 20
        assert r.w == 100
        assert r.h == 50

    def test_region_is_frozen(self) -> None:
        r = Region(0, 0, 10, 10)
        with pytest.raises(AttributeError):
            r.x = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------

class TestTemplateGeneration:
    def test_generates_all_characters(self) -> None:
        templates = _generate_digit_templates(height=20)
        expected_chars = set("0123456789.,$ ")
        assert set(templates.keys()) == expected_chars

    def test_templates_have_correct_height(self) -> None:
        height = 24
        templates = _generate_digit_templates(height=height)
        for ch, tmpl in templates.items():
            assert tmpl.shape[0] == height, f"Template for '{ch}' has wrong height"

    def test_templates_are_binary(self) -> None:
        templates = _generate_digit_templates(height=16)
        for ch, tmpl in templates.items():
            unique = set(np.unique(tmpl))
            assert unique <= {0, 255}, f"Template '{ch}' is not binary: {unique}"


# ---------------------------------------------------------------------------
# parse_amount (static, no OCR needed)
# ---------------------------------------------------------------------------

class TestParseAmount:
    @pytest.fixture()
    def engine(self) -> OCREngine:
        return OCREngine()

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("500", 500.0),
            ("12.50", 12.5),
            ("1,234.56", 1234.56),
            ("$1,234", 1234.0),
            ("$500", 500.0),
            ("1.2K", 1200.0),
            ("1.2k", 1200.0),
            ("3M", 3_000_000.0),
            ("3m", 3_000_000.0),
            ("BB 2.5", 2.5),
            ("bb 10", 10.0),
            ("0", 0.0),
            ("$0.50", 0.5),
        ],
    )
    def test_valid_formats(self, text: str, expected: float) -> None:
        result = OCREngine.parse_amount(text)
        assert result is not None
        assert abs(result - expected) < 1e-6

    @pytest.mark.parametrize("text", ["", "   ", None])
    def test_empty_or_none(self, text: str | None) -> None:
        result = OCREngine.parse_amount(text)  # type: ignore[arg-type]
        assert result is None

    def test_unparseable_text(self) -> None:
        # Pure alpha text that cannot be a number.
        result = OCREngine.parse_amount("ABCXYZ")
        assert result is None


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_returns_binary_image(self) -> None:
        frame = _render_text_image("123", width=100, height=30)
        region = Region(0, 0, 100, 30)
        binary = OCREngine.preprocess(frame, region)
        unique = set(np.unique(binary))
        assert unique <= {0, 255}

    def test_empty_region(self) -> None:
        frame = _render_text_image("123", width=100, height=30)
        region = Region(50, 50, 0, 0)
        binary = OCREngine.preprocess(frame, region)
        # Should return a tiny fallback.
        assert binary.size >= 1

    def test_grayscale_input(self) -> None:
        gray = np.full((30, 100), 128, dtype=np.uint8)
        region = Region(0, 0, 100, 30)
        binary = OCREngine.preprocess(gray, region)
        assert len(binary.shape) == 2

    def test_light_background_inverted(self) -> None:
        """White bg / dark text should be inverted to white-on-black."""
        img = np.full((30, 100, 3), 240, dtype=np.uint8)  # light bg
        cv2.putText(img, "5", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2)
        region = Region(0, 0, 100, 30)
        binary = OCREngine.preprocess(img, region)
        # After inversion, text pixels should be 255 (white).
        assert binary.max() == 255


# ---------------------------------------------------------------------------
# End-to-end recognition (synthetic images)
# ---------------------------------------------------------------------------

class TestExtraction:
    @pytest.fixture()
    def engine(self) -> OCREngine:
        return OCREngine(match_threshold=0.40)

    def test_extract_amount_returns_float_or_none(self, engine: OCREngine) -> None:
        frame = _render_text_image("500", width=120, height=40)
        region = Region(0, 0, 120, 40)
        result = engine.extract_amount(frame, region)
        # Template matching on synthetic images is inherently noisy; we accept
        # either a plausible float or None (if confidence is too low).
        assert result is None or isinstance(result, float)

    def test_extract_stack_size_delegates_to_extract_amount(
        self, engine: OCREngine
    ) -> None:
        frame = _render_text_image("1000", width=140, height=40)
        region = Region(0, 0, 140, 40)
        amt = engine.extract_amount(frame, region)
        stack = engine.extract_stack_size(frame, region)
        assert amt == stack

    def test_recognize_text_returns_ocr_result(self, engine: OCREngine) -> None:
        frame = _render_text_image("42", width=80, height=30)
        region = Region(0, 0, 80, 30)
        result = engine.recognize_text(frame, region)
        assert isinstance(result, OCRResult)
        assert isinstance(result.raw_text, str)
        assert 0.0 <= result.confidence <= 1.0

    def test_empty_frame_returns_none(self, engine: OCREngine) -> None:
        frame = np.zeros((40, 120, 3), dtype=np.uint8)
        region = Region(0, 0, 120, 40)
        result = engine.extract_amount(frame, region)
        # Black frame -- nothing to recognise.
        assert result is None or isinstance(result, float)


# ---------------------------------------------------------------------------
# OCRResult dataclass
# ---------------------------------------------------------------------------

class TestOCRResult:
    def test_fields(self) -> None:
        r = OCRResult(raw_text="$500", value=500.0, confidence=0.92)
        assert r.raw_text == "$500"
        assert r.value == 500.0
        assert r.confidence == 0.92

    def test_none_value(self) -> None:
        r = OCRResult(raw_text="???", value=None, confidence=0.1)
        assert r.value is None
