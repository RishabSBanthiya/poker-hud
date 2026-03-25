"""Game state coordinator — central hub for game state management.

Receives detection results, updates the current hand state, tracks
street transitions, and notifies subscribers of state changes and
completed hands.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Callable, Optional

from src.detection.card import Card
from src.detection.detection_pipeline import DetectionResult
from src.engine.game_state import (
    ActionType,
    HandState,
    PlayerAction,
    PlayerState,
)
from src.engine.hand_history import HandHistoryParser, HandRecord
from src.engine.hand_phase_tracker import HandPhaseTracker

logger = logging.getLogger(__name__)


class GameStateCoordinator:
    """Central coordinator for game state tracking.

    Processes detection results into game state updates, tracks
    hand phases, and notifies callbacks on state changes.

    Args:
        hand_phase_tracker: Tracker for street transitions.
    """

    def __init__(self, hand_phase_tracker: HandPhaseTracker) -> None:
        self._phase_tracker = hand_phase_tracker
        self._current_hand: Optional[HandState] = None
        self._hand_history_parser = HandHistoryParser()
        self._on_state_change: Optional[
            Callable[[HandState], None]
        ] = None
        self._on_hand_complete: Optional[
            Callable[[HandRecord], None]
        ] = None

    @property
    def current_hand(self) -> Optional[HandState]:
        """The current hand being tracked."""
        return self._current_hand

    def set_state_change_callback(
        self, callback: Callable[[HandState], None]
    ) -> None:
        """Set callback for game state changes.

        Args:
            callback: Function receiving the updated HandState.
        """
        self._on_state_change = callback

    def set_hand_complete_callback(
        self, callback: Callable[[HandRecord], None]
    ) -> None:
        """Set callback for completed hands.

        Args:
            callback: Function receiving a HandRecord.
        """
        self._on_hand_complete = callback

    def process_detection(self, detection: DetectionResult) -> None:
        """Process a detection result and update game state.

        Args:
            detection: Detection result from the detection pipeline.
        """
        card_result = detection.card_result
        community_cards = [d.card for d in card_result.community_cards]
        hole_cards = [d.card for d in card_result.hole_cards]

        # Check if we need to start a new hand
        if self._current_hand is None or self._should_start_new_hand(
            community_cards
        ):
            self._start_new_hand(detection)

        hand = self._current_hand
        if hand is None:
            return

        # Update community cards
        if community_cards:
            hand.community_cards = community_cards

        # Update hole cards for hero
        if hole_cards:
            hero = next(
                (p for p in hand.players if p.is_hero), None
            )
            if hero is not None:
                hero.hole_cards = hole_cards

        # Update players from detection
        for player_info in detection.players:
            existing = hand.get_player(player_info.name)
            if existing is None:
                hand.players.append(
                    PlayerState(
                        name=player_info.name,
                        seat_index=player_info.seat_index,
                        is_hero=player_info.is_hero,
                    )
                )

        # Track street transitions
        new_street = self._phase_tracker.update(community_cards)
        if new_street is not None:
            hand.street = new_street

        if self._on_state_change is not None:
            self._on_state_change(hand)

    def record_action(
        self,
        player_name: str,
        action_type: ActionType,
        amount: float = 0.0,
    ) -> None:
        """Record a player action in the current hand.

        Args:
            player_name: Name of the acting player.
            action_type: Type of action.
            amount: Bet/raise amount.
        """
        if self._current_hand is None:
            return

        action = PlayerAction(
            player_name=player_name,
            action_type=action_type,
            amount=amount,
            street=self._current_hand.street,
        )
        self._current_hand.actions.append(action)
        self._current_hand.pot += amount

        # Mark folded players as inactive
        if action_type == ActionType.FOLD:
            player = self._current_hand.get_player(player_name)
            if player is not None:
                player.is_active = False

        if self._on_state_change is not None:
            self._on_state_change(self._current_hand)

    def complete_hand(self, winner_name: Optional[str] = None) -> Optional[HandRecord]:
        """Mark the current hand as complete.

        Args:
            winner_name: Name of the winning player.

        Returns:
            A HandRecord of the completed hand, or None if no hand active.
        """
        if self._current_hand is None:
            return None

        self._current_hand.is_complete = True
        self._current_hand.winner_name = winner_name

        record = self._hand_history_parser.hand_state_to_record(
            self._current_hand, timestamp=time.time()
        )

        if self._on_hand_complete is not None:
            self._on_hand_complete(record)

        logger.info(
            "Hand %s completed — pot=%.1f, winner=%s",
            record.hand_id,
            record.pot,
            record.winner_name,
        )

        self._current_hand = None
        self._phase_tracker.reset()

        return record

    def _should_start_new_hand(self, community_cards: list[Card]) -> bool:
        """Determine if a new hand should be started.

        A new hand is detected when community cards reset to empty
        after having been non-empty (hand finished and new deal).
        """
        if self._current_hand is None:
            return True
        if self._current_hand.is_complete:
            return True
        if (
            len(community_cards) == 0
            and len(self._current_hand.community_cards) > 0
        ):
            return True
        return False

    def _start_new_hand(self, detection: DetectionResult) -> None:
        """Initialize a new hand from detection data.

        Args:
            detection: The detection result triggering the new hand.
        """
        # Complete previous hand if still open
        if self._current_hand is not None and not self._current_hand.is_complete:
            self.complete_hand()

        hand_id = str(uuid.uuid4())[:8]
        players = [
            PlayerState(
                name=pi.name,
                seat_index=pi.seat_index,
                is_hero=pi.is_hero,
            )
            for pi in detection.players
        ]

        self._current_hand = HandState(
            hand_id=hand_id,
            players=players,
        )
        self._phase_tracker.reset()

        logger.info("New hand started: %s with %d players", hand_id, len(players))
