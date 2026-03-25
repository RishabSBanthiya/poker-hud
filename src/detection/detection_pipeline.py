"""End-to-end detection pipeline combining card recognition, OCR, and player ID.

Orchestrates all detection components to produce a unified detection
result from a single frame.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from src.detection.card_recognition import (
    CardRecognitionPipeline,
    CardRecognitionResult,
)
from src.detection.ocr_engine import OCREngine
from src.detection.player_identifier import PlayerIdentifier, PlayerInfo

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Combined result from all detection subsystems.

    Attributes:
        card_result: Card recognition result.
        players: List of identified players.
        timestamp: Time when detection was performed.
    """

    card_result: CardRecognitionResult = field(
        default_factory=CardRecognitionResult
    )
    players: list[PlayerInfo] = field(default_factory=list)
    timestamp: float = 0.0


class DetectionPipeline:
    """Orchestrates card recognition, OCR, and player identification.

    Processes frames through all detection components and delivers
    unified results via a callback.

    Args:
        card_recognition: Card recognition pipeline.
        ocr_engine: OCR engine for text extraction.
        player_identifier: Player identification component.
    """

    def __init__(
        self,
        card_recognition: CardRecognitionPipeline,
        ocr_engine: OCREngine,
        player_identifier: PlayerIdentifier,
    ) -> None:
        self._card_recognition = card_recognition
        self._ocr = ocr_engine
        self._player_identifier = player_identifier
        self._result_callback: Optional[
            Callable[[DetectionResult], None]
        ] = None

    def set_result_callback(
        self, callback: Callable[[DetectionResult], None]
    ) -> None:
        """Set the callback for detection results.

        Args:
            callback: Function accepting a DetectionResult.
        """
        self._result_callback = callback

    def initialize(self) -> None:
        """Initialize all detection components."""
        self._card_recognition.initialize()
        self._ocr.initialize()
        logger.info("Detection pipeline initialized")

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """Run all detection components on a frame.

        Args:
            frame: Input BGR image.

        Returns:
            Combined detection result.
        """
        timestamp = time.time()

        card_result = self._card_recognition.process_frame(frame)
        card_result.frame_timestamp = timestamp

        players = self._player_identifier.identify_players(frame)

        result = DetectionResult(
            card_result=card_result,
            players=players,
            timestamp=timestamp,
        )

        if self._result_callback is not None:
            self._result_callback(result)

        logger.debug(
            "Detection: %d cards, %d players",
            len(card_result.all_detections),
            len(players),
        )
        return result

    def shutdown(self) -> None:
        """Release all detection resources."""
        self._ocr.shutdown()
        logger.info("Detection pipeline shut down")
