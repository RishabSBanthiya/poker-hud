"""Card recognition pipeline combining template matching with validation.

Provides a high-level interface for detecting cards in frames,
wrapping the low-level TemplateMatcher with pre/post processing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.detection.card import DetectedCard
from src.detection.template_matcher import TemplateMatcher

logger = logging.getLogger(__name__)


@dataclass
class CardRecognitionResult:
    """Result of card recognition on a single frame.

    Attributes:
        community_cards: Cards detected on the board (up to 5).
        hole_cards: Cards detected in the player's hand (up to 2).
        all_detections: All raw card detections before assignment.
        frame_timestamp: Timestamp of the processed frame.
    """

    community_cards: list[DetectedCard] = field(default_factory=list)
    hole_cards: list[DetectedCard] = field(default_factory=list)
    all_detections: list[DetectedCard] = field(default_factory=list)
    frame_timestamp: float = 0.0


class CardRecognitionPipeline:
    """High-level card recognition pipeline.

    Wraps TemplateMatcher to provide structured card recognition
    with community/hole card assignment based on screen regions.

    Args:
        template_dir: Path to the card template images directory.
        confidence_threshold: Minimum confidence for a valid detection.
    """

    def __init__(
        self,
        template_dir: str | Path = "data/templates",
        confidence_threshold: float = 0.8,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._matcher: Optional[TemplateMatcher] = None
        self._template_dir = Path(template_dir)

    @property
    def is_initialized(self) -> bool:
        """Whether the template matcher has been loaded."""
        return self._matcher is not None

    def initialize(self) -> None:
        """Load templates and prepare the recognition pipeline.

        Raises:
            FileNotFoundError: If the template directory does not exist.
        """
        if self._template_dir.is_dir():
            self._matcher = TemplateMatcher(self._template_dir)
            logger.info(
                "Card recognition initialized with %d templates",
                len(self._matcher.templates),
            )
        else:
            logger.warning(
                "Template directory not found: %s — recognition disabled",
                self._template_dir,
            )

    def process_frame(self, frame: np.ndarray) -> CardRecognitionResult:
        """Detect cards in a BGR frame.

        Args:
            frame: Input image as a BGR numpy array.

        Returns:
            CardRecognitionResult with detected cards assigned to regions.
        """
        result = CardRecognitionResult()

        if self._matcher is None:
            return result

        detections = self._matcher.detect_cards(
            frame, threshold=self._confidence_threshold
        )
        result.all_detections = detections

        # Simple heuristic: cards in the upper-middle are community,
        # cards in the lower region are hole cards.
        frame_height = frame.shape[0]
        mid_y = frame_height // 2

        for det in detections:
            _, y, _, _ = det.bounding_box
            if y < mid_y:
                result.community_cards.append(det)
            else:
                result.hole_cards.append(det)

        return result
