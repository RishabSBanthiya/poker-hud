"""OCR engine for extracting numeric values from poker table screen regions.

Uses a lightweight digit template matching approach to extract bet amounts,
stack sizes, and other numeric values from captured frames. No external OCR
dependencies (e.g. pytesseract) required -- relies on OpenCV contour analysis
and template matching.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Characters we need to recognize in poker numeric displays.
_CHAR_SET = "0123456789.,$ "


@dataclass(frozen=True)
class Region:
    """Rectangular region within a frame.

    Attributes:
        x: Left edge in pixels.
        y: Top edge in pixels.
        w: Width in pixels.
        h: Height in pixels.
    """

    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class OCRResult:
    """Result of an OCR extraction attempt.

    Attributes:
        raw_text: The raw string of characters recognized.
        value: Parsed numeric value, or None if parsing failed.
        confidence: Mean confidence score across recognized characters (0.0-1.0).
    """

    raw_text: str
    value: Optional[float]
    confidence: float


def _generate_digit_templates(
    height: int = 20,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    thickness: int = 1,
) -> dict[str, np.ndarray]:
    """Generate binary template images for each character in _CHAR_SET.

    Args:
        height: Target height for rendered templates in pixels.
        font_face: OpenCV font face constant.
        thickness: Stroke thickness for rendered text.

    Returns:
        Mapping from character to binary (0/255) template image.
    """
    templates: dict[str, np.ndarray] = {}
    font_scale = cv2.getFontScaleFromHeight(font_face, height, thickness)

    for ch in _CHAR_SET:
        if ch == " ":
            # Space is a blank template -- used as a gap filler.
            templates[ch] = np.zeros((height, max(height // 3, 4)), dtype=np.uint8)
            continue

        (tw, th), baseline = cv2.getTextSize(ch, font_face, font_scale, thickness)
        # Pad a little to avoid clipping.
        canvas_h = th + baseline + 4
        canvas_w = tw + 4
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cv2.putText(canvas, ch, (2, th + 2), font_face, font_scale, 255, thickness)

        # Resize to uniform height.
        scale = height / canvas_h
        new_w = max(int(canvas_w * scale), 1)
        resized = cv2.resize(canvas, (new_w, height), interpolation=cv2.INTER_AREA)
        _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        templates[ch] = binary

    return templates


class OCREngine:
    """Lightweight OCR engine for poker table numeric extraction.

    Uses template matching against pre-rendered digit images to recognize
    characters in screen-captured regions.  Designed for the limited character
    set found on poker tables (digits, currency symbols, decimal separators).

    Example::

        engine = OCREngine()
        amount = engine.extract_amount(frame, Region(100, 200, 80, 24))
    """

    # Multiplier suffixes recognized in shorthand amounts (e.g. "1.2K").
    _SUFFIX_MULTIPLIERS: dict[str, float] = {
        "k": 1_000.0,
        "m": 1_000_000.0,
        "b": 1_000_000_000.0,
    }

    def __init__(
        self,
        template_height: int = 20,
        match_threshold: float = 0.55,
    ) -> None:
        """Initialise the OCR engine.

        Args:
            template_height: Pixel height for digit templates.
            match_threshold: Minimum normalised correlation to accept a match.
        """
        self._template_height = template_height
        self._match_threshold = match_threshold
        self._templates = _generate_digit_templates(height=template_height)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_amount(
        self, frame: np.ndarray, region: Region
    ) -> Optional[float]:
        """Extract a monetary / chip amount from *region* of *frame*.

        Handles formats such as ``"$1,234.56"``, ``"500"``, ``"1.2K"``,
        ``"BB 2.5"``.

        Args:
            frame: Source image as a BGR numpy array.
            region: Rectangular area to read.

        Returns:
            Parsed float value, or ``None`` if extraction failed.
        """
        result = self._recognize(frame, region)
        if result.value is not None:
            return result.value
        return None

    def extract_stack_size(
        self, frame: np.ndarray, region: Region
    ) -> Optional[float]:
        """Extract a player's stack size from *region* of *frame*.

        Functionally identical to :meth:`extract_amount` but provided as a
        separate entry-point for semantic clarity -- callers dealing with
        stack-size regions can use this without worrying about format
        differences.

        Args:
            frame: Source image as a BGR numpy array.
            region: Rectangular area to read.

        Returns:
            Parsed float value, or ``None`` if extraction failed.
        """
        return self.extract_amount(frame, region)

    def recognize_text(
        self, frame: np.ndarray, region: Region
    ) -> OCRResult:
        """Run OCR on a region and return the full result with confidence.

        Args:
            frame: Source image as a BGR numpy array.
            region: Rectangular area to read.

        Returns:
            An :class:`OCRResult` containing raw text, parsed value, and
            confidence score.
        """
        return self._recognize(frame, region)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess(frame: np.ndarray, region: Region) -> np.ndarray:
        """Crop, convert to grayscale, threshold, and denoise a region.

        Args:
            frame: Source BGR image.
            region: Area to crop.

        Returns:
            Binary (0/255) grayscale image ready for template matching.
        """
        cropped = frame[
            region.y : region.y + region.h,
            region.x : region.x + region.w,
        ]

        if cropped.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        # Convert to grayscale.
        if len(cropped.shape) == 3:
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped.copy()

        # Denoise with a small Gaussian blur.
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive threshold handles varying backgrounds better than a global
        # threshold.  We use a small block size because the text regions are
        # typically small.
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        # If the majority of pixels are white, invert so text is white-on-black
        # (matching our templates).
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        return binary

    # ------------------------------------------------------------------
    # Template matching internals
    # ------------------------------------------------------------------

    def _recognize(self, frame: np.ndarray, region: Region) -> OCRResult:
        """Core recognition pipeline: preprocess -> match -> parse.

        Args:
            frame: Source BGR image.
            region: Area to read.

        Returns:
            :class:`OCRResult` with extracted data.
        """
        binary = self.preprocess(frame, region)

        # Resize binary image so its height matches template height.
        if binary.shape[0] < 2 or binary.shape[1] < 2:
            return OCRResult(raw_text="", value=None, confidence=0.0)

        scale = self._template_height / binary.shape[0]
        new_w = max(int(binary.shape[1] * scale), 1)
        resized = cv2.resize(
            binary, (new_w, self._template_height), interpolation=cv2.INTER_AREA
        )
        _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

        chars, confidences = self._match_templates(resized)
        raw_text = "".join(chars)
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        value = self.parse_amount(raw_text)

        return OCRResult(raw_text=raw_text, value=value, confidence=mean_conf)

    def _match_templates(
        self, binary: np.ndarray
    ) -> tuple[list[str], list[float]]:
        """Slide templates across *binary* image left-to-right.

        Uses ``cv2.matchTemplate`` with normalised cross-correlation.  At each
        horizontal position the best-matching character is accepted if its
        score exceeds :attr:`_match_threshold`, and the scan position advances
        by the template width.

        Args:
            binary: Binary image (height == ``_template_height``).

        Returns:
            Tuple of (recognised characters, per-character confidence scores).
        """
        img_h, img_w = binary.shape[:2]
        chars: list[str] = []
        confidences: list[float] = []
        x_pos = 0

        while x_pos < img_w - 2:
            best_score = -1.0
            best_char = ""
            best_width = 1

            for ch, tmpl in self._templates.items():
                t_h, t_w = tmpl.shape[:2]
                if t_w > img_w - x_pos or t_h > img_h:
                    continue

                roi = binary[:t_h, x_pos : x_pos + t_w]
                if roi.shape != tmpl.shape:
                    continue

                result = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
                score = float(result[0, 0]) if result.size > 0 else -1.0

                if score > best_score:
                    best_score = score
                    best_char = ch
                    best_width = t_w

            if best_score >= self._match_threshold and best_char:
                # Skip runs of spaces.
                if best_char != " " or (chars and chars[-1] != " "):
                    chars.append(best_char)
                    confidences.append(best_score)
                x_pos += best_width
            else:
                # No match -- advance by a minimal step.
                x_pos += max(best_width // 2, 1)

        return chars, confidences

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @classmethod
    def parse_amount(cls, text: str) -> Optional[float]:
        """Parse a raw OCR string into a numeric value.

        Supported formats:
        - Plain integers: ``"500"``
        - Decimals: ``"12.50"``
        - With thousands separator: ``"1,234.56"``
        - Currency prefix: ``"$1,234"``
        - Shorthand suffix: ``"1.2K"``, ``"3M"``
        - Big-blind notation: ``"BB 2.5"``

        Args:
            text: Raw OCR text (may include whitespace, ``$``, ``BB``).

        Returns:
            Parsed float, or ``None`` if the string cannot be interpreted.
        """
        if not text or not text.strip():
            return None

        cleaned = text.strip()

        # Remove leading "BB" / "bb" prefix.
        cleaned = re.sub(r"(?i)^bb\s*", "", cleaned)

        # Remove dollar sign.
        cleaned = cleaned.replace("$", "")

        # Strip remaining whitespace.
        cleaned = cleaned.strip()

        if not cleaned:
            return None

        # Check for multiplier suffix (K, M, B).
        multiplier = 1.0
        if cleaned and cleaned[-1].lower() in cls._SUFFIX_MULTIPLIERS:
            multiplier = cls._SUFFIX_MULTIPLIERS[cleaned[-1].lower()]
            cleaned = cleaned[:-1]

        # Remove thousands separators (commas).
        cleaned = cleaned.replace(",", "")

        # Strip any remaining non-numeric characters (except dot).
        cleaned = re.sub(r"[^0-9.]", "", cleaned)

        if not cleaned:
            return None

        try:
            return float(cleaned) * multiplier
        except ValueError:
            logger.debug("Failed to parse amount from '%s'", text)
            return None
