"""Hand phase tracking based on community card count.

Tracks the progression of a poker hand through streets based
on the number of community cards detected.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.detection.card import Card
from src.engine.game_state import Street

logger = logging.getLogger(__name__)


class HandPhaseTracker:
    """Tracks the current street of a poker hand.

    Determines the hand phase based on the number of community
    cards visible on the board.
    """

    def __init__(self) -> None:
        self._current_street = Street.PREFLOP
        self._previous_community_count = 0

    @property
    def current_street(self) -> Street:
        """The current street."""
        return self._current_street

    def update(self, community_cards: list[Card]) -> Optional[Street]:
        """Update the hand phase based on detected community cards.

        Args:
            community_cards: Currently visible community cards.

        Returns:
            The new Street if a transition occurred, None otherwise.
        """
        count = len(community_cards)
        new_street = self._street_from_card_count(count)

        if new_street != self._current_street:
            old_street = self._current_street
            self._current_street = new_street
            self._previous_community_count = count
            logger.info(
                "Street transition: %s -> %s (%d community cards)",
                old_street.value,
                new_street.value,
                count,
            )
            return new_street

        self._previous_community_count = count
        return None

    def reset(self) -> None:
        """Reset to preflop for a new hand."""
        self._current_street = Street.PREFLOP
        self._previous_community_count = 0

    @staticmethod
    def _street_from_card_count(count: int) -> Street:
        """Determine the street from the number of community cards.

        Args:
            count: Number of community cards visible.

        Returns:
            The corresponding Street.
        """
        if count == 0:
            return Street.PREFLOP
        elif count == 3:
            return Street.FLOP
        elif count == 4:
            return Street.TURN
        elif count >= 5:
            return Street.RIVER
        else:
            # 1 or 2 cards could be transitional — stay at current
            return Street.PREFLOP
