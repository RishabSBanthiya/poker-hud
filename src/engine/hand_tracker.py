"""Hand phase tracking via community card count changes.

Monitors ``DetectionResult`` updates to determine the current street,
detect street transitions, and signal new-hand resets.  Designed to work
with both the live-capture detection pipeline and replayed hand histories.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.detection.card import Card
from src.detection.validation import DetectionResult
from src.engine.game_state import (
    STREET_COMMUNITY_CARDS,
    GameState,
    Street,
)

logger = logging.getLogger(__name__)

# Maps community-card count to the corresponding street.
_CARD_COUNT_TO_STREET: dict[int, Street] = {
    v: k for k, v in STREET_COMMUNITY_CARDS.items() if k != Street.SHOWDOWN
}

# Valid community card counts during a hand.
_VALID_COMMUNITY_COUNTS: frozenset[int] = frozenset(
    STREET_COMMUNITY_CARDS.values()
)

# Callback type for street-transition events.
StreetTransitionCallback = Callable[[Street, Street], None]

# Callback type for new-hand events.
NewHandCallback = Callable[[], None]


@dataclass
class _TrackerState:
    """Internal mutable state for the hand phase tracker.

    Attributes:
        current_street: The street we believe we are on.
        previous_community: Community cards seen in the last update.
        betting_round: Tracks the number of betting rounds within the
            current street (resets on each street transition).
        hand_active: Whether a hand is currently in progress.
        last_update_ts: Timestamp of the most recent ``update`` call.
        consecutive_reset_frames: Number of consecutive frames with zero
            community cards after a hand was active (used to debounce
            new-hand detection).
    """

    current_street: Street = Street.PREFLOP
    previous_community: list[Card] = field(default_factory=list)
    betting_round: int = 0
    hand_active: bool = False
    last_update_ts: float = 0.0
    consecutive_reset_frames: int = 0


class HandPhaseTracker:
    """Tracks the current betting street by observing community card changes.

    Street transitions are inferred from the number of community cards
    reported in each ``DetectionResult``:

    * 0 cards -> PREFLOP
    * 3 cards -> FLOP
    * 4 cards -> TURN
    * 5 cards -> RIVER

    The tracker also detects new-hand starts (community cards drop back
    to zero after having been dealt) and fires registered callbacks on
    both street transitions and new-hand events.

    Args:
        game_state: Optional ``GameState`` to keep synchronised with the
            tracked street.
        reset_frame_threshold: Number of consecutive zero-community-card
            frames required before a new hand is declared.  Prevents
            spurious resets from a single missed frame.
    """

    def __init__(
        self,
        game_state: Optional[GameState] = None,
        reset_frame_threshold: int = 2,
    ) -> None:
        self._game_state = game_state
        self._reset_frame_threshold = max(1, reset_frame_threshold)
        self._state = _TrackerState()
        self._transition_callbacks: list[StreetTransitionCallback] = []
        self._new_hand_callbacks: list[NewHandCallback] = []

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def current_street(self) -> Street:
        """Return the currently tracked street."""
        return self._state.current_street

    @property
    def betting_round(self) -> int:
        """Return the current betting round within the street."""
        return self._state.betting_round

    @property
    def hand_active(self) -> bool:
        """Return whether a hand is currently in progress."""
        return self._state.hand_active

    @property
    def previous_community_cards(self) -> list[Card]:
        """Return the community cards from the last update."""
        return list(self._state.previous_community)

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_street_transition(self, callback: StreetTransitionCallback) -> None:
        """Register a callback for street transitions.

        Args:
            callback: Called with ``(old_street, new_street)`` whenever a
                transition is detected.
        """
        self._transition_callbacks.append(callback)

    def on_new_hand(self, callback: NewHandCallback) -> None:
        """Register a callback for new-hand events.

        Args:
            callback: Called (with no arguments) when a new hand starts.
        """
        self._new_hand_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Core update logic
    # ------------------------------------------------------------------

    def update(self, detection_result: DetectionResult) -> Optional[Street]:
        """Process a detection result and return the new street, if changed.

        This is the main entry point called once per frame / detection
        cycle.  It inspects the community card count to decide whether
        the street has advanced or a new hand has started.

        Args:
            detection_result: The latest validated detection output.

        Returns:
            The new ``Street`` if a transition occurred, otherwise ``None``.
        """
        community = detection_result.get_community_cards()
        count = len(community)
        now = detection_result.timestamp or time.time()
        self._state.last_update_ts = now

        # ------ New-hand detection ------
        if self._state.hand_active and count == 0:
            self._state.consecutive_reset_frames += 1
            if (
                self._state.consecutive_reset_frames
                >= self._reset_frame_threshold
            ):
                self._handle_new_hand()
                return Street.PREFLOP
            # Not enough consecutive resets yet; no transition.
            return None

        # If we see community cards, reset the consecutive-reset counter.
        self._state.consecutive_reset_frames = 0

        # ------ Determine target street from card count ------
        target_street = _CARD_COUNT_TO_STREET.get(count)

        if target_street is None:
            # Invalid count (e.g. 1, 2) -- likely a detection artefact.
            logger.debug(
                "Ignoring invalid community card count %d", count
            )
            return None

        # ------ First cards seen in a new hand ------
        if not self._state.hand_active:
            self._state.hand_active = True

        # ------ Check for street transition ------
        old_street = self._state.current_street

        if target_street == old_street:
            # Same street; check if the cards changed within the same
            # count (unlikely but could happen with misdetections).
            self._state.previous_community = community
            return None

        if target_street.value <= old_street.value:
            # Going backwards is invalid during a hand -- could be a
            # new hand that wasn't caught via zero-card reset (e.g. we
            # jumped straight from RIVER to FLOP).
            if count < len(self._state.previous_community):
                self._handle_new_hand()
                # After reset we're on PREFLOP; re-evaluate.
                return self._apply_transition(
                    Street.PREFLOP, target_street, community
                )
            logger.warning(
                "Street regression detected (%s -> %s) without card "
                "count decrease; ignoring.",
                old_street.name,
                target_street.name,
            )
            return None

        return self._apply_transition(old_street, target_street, community)

    def is_new_hand(self) -> bool:
        """Return whether the most recent update triggered a new hand.

        Convenience alias: returns ``True`` if we are on PREFLOP with
        no community cards and the hand was previously active.
        """
        return (
            self._state.current_street == Street.PREFLOP
            and len(self._state.previous_community) == 0
        )

    def reset(self) -> None:
        """Fully reset the tracker state (e.g. at application start)."""
        self._state = _TrackerState()
        if self._game_state is not None:
            self._game_state.reset_for_new_hand()

    def advance_betting_round(self) -> int:
        """Manually signal that a betting round has completed.

        Returns:
            The new betting-round count for the current street.
        """
        self._state.betting_round += 1
        return self._state.betting_round

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_transition(
        self,
        old_street: Street,
        new_street: Street,
        community: list[Card],
    ) -> Optional[Street]:
        """Apply a street transition and fire callbacks.

        Handles the case where multiple streets are skipped (e.g.
        PREFLOP -> TURN due to a missed FLOP detection) by emitting
        intermediate transitions.

        Args:
            old_street: The street we are transitioning from.
            new_street: The street we are transitioning to.
            community: The current community cards.

        Returns:
            The final new street after all transitions.
        """
        # Walk through each intermediate street in order.
        current = old_street
        while current != new_street:
            next_street = self._next_street(current)
            if next_street is None:
                break
            self._fire_transition(current, next_street)
            current = next_street

        self._state.current_street = new_street
        self._state.previous_community = community
        self._state.betting_round = 0
        return new_street

    def _fire_transition(self, old: Street, new: Street) -> None:
        """Fire transition callbacks and sync GameState."""
        logger.info("Street transition: %s -> %s", old.name, new.name)

        if self._game_state is not None:
            try:
                self._game_state.advance_street()
            except ValueError:
                logger.warning(
                    "GameState could not advance from %s", old.name
                )

        for cb in self._transition_callbacks:
            try:
                cb(old, new)
            except Exception:
                logger.exception("Error in street-transition callback")

    def _handle_new_hand(self) -> None:
        """Reset state for a new hand and fire callbacks."""
        logger.info("New hand detected")
        self._state = _TrackerState()
        self._state.hand_active = False

        if self._game_state is not None:
            self._game_state.reset_for_new_hand()

        for cb in self._new_hand_callbacks:
            try:
                cb()
            except Exception:
                logger.exception("Error in new-hand callback")

    @staticmethod
    def _next_street(street: Street) -> Optional[Street]:
        """Return the next street in order, or None at SHOWDOWN."""
        order = [
            Street.PREFLOP,
            Street.FLOP,
            Street.TURN,
            Street.RIVER,
            Street.SHOWDOWN,
        ]
        try:
            idx = order.index(street)
        except ValueError:
            return None
        if idx + 1 < len(order):
            return order[idx + 1]
        return None
