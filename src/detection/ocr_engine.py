"""OCR engine for reading text from poker table regions.

Extracts player names, bet amounts, and pot sizes from screen regions
using image preprocessing and text recognition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result of OCR on a screen region.

    Attributes:
        text: The recognized text string.
        confidence: Recognition confidence (0.0-1.0).
        region: The (x, y, width, height) region that was processed.
    """

    text: str
    confidence: float
    region: tuple[int, int, int, int]


class OCREngine:
    """OCR engine for poker table text extraction.

    Uses image preprocessing (thresholding, morphology) to enhance
    text regions before recognition.
    """

    def __init__(self) -> None:
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Whether the OCR engine is ready."""
        return self._initialized

    def initialize(self) -> None:
        """Initialize the OCR engine."""
        self._initialized = True
        logger.info("OCR engine initialized")

    def extract_text(
        self,
        frame: np.ndarray,
        region: tuple[int, int, int, int],
    ) -> OCRResult:
        """Extract text from a specific region of a frame.

        Args:
            frame: Input BGR image.
            region: (x, y, width, height) region to process.

        Returns:
            OCRResult with extracted text and confidence.
        """
        x, y, w, h = region
        if (
            x < 0
            or y < 0
            or x + w > frame.shape[1]
            or y + h > frame.shape[0]
        ):
            return OCRResult(text="", confidence=0.0, region=region)

        # Placeholder — actual OCR implementation will use
        # Vision framework or tesseract
        return OCRResult(text="", confidence=0.0, region=region)

    def shutdown(self) -> None:
        """Release OCR resources."""
        self._initialized = False
        logger.info("OCR engine shut down")
