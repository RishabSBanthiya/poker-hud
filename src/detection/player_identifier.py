"""Player identification from screen regions via OCR.

Identifies players at the poker table by reading name labels
from predefined seat regions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.detection.ocr_engine import OCREngine

logger = logging.getLogger(__name__)


@dataclass
class PlayerInfo:
    """Information about a detected player at the table.

    Attributes:
        seat_index: Seat position (0-based).
        name: Player name as read by OCR.
        confidence: Name recognition confidence.
        is_hero: Whether this is the local player.
    """

    seat_index: int
    name: str
    confidence: float = 0.0
    is_hero: bool = False


class PlayerIdentifier:
    """Identifies players at the table using OCR on seat name regions.

    Args:
        ocr_engine: The OCR engine to use for text extraction.
        max_seats: Maximum number of seats at the table.
    """

    def __init__(
        self,
        ocr_engine: OCREngine,
        max_seats: int = 9,
    ) -> None:
        self._ocr = ocr_engine
        self._max_seats = max_seats
        self._known_players: dict[int, PlayerInfo] = {}

    @property
    def known_players(self) -> dict[int, PlayerInfo]:
        """Return currently known players by seat index."""
        return dict(self._known_players)

    def identify_players(
        self, frame: np.ndarray
    ) -> list[PlayerInfo]:
        """Scan the frame for player names at each seat.

        Args:
            frame: Input BGR image of the full table.

        Returns:
            List of identified players.
        """
        # Placeholder — actual implementation would define seat regions
        # and run OCR on each one
        return list(self._known_players.values())

    def set_player(self, seat_index: int, name: str, is_hero: bool = False) -> None:
        """Manually set a player at a seat position.

        Args:
            seat_index: Seat position (0-based).
            name: Player name.
            is_hero: Whether this is the local player.
        """
        self._known_players[seat_index] = PlayerInfo(
            seat_index=seat_index,
            name=name,
            confidence=1.0,
            is_hero=is_hero,
        )

    def clear(self) -> None:
        """Clear all known player information."""
        self._known_players.clear()
