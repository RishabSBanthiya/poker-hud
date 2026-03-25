"""Game state engine coordinator.

Central orchestrator that receives detection results (cards, OCR data,
player identifications) and maintains the authoritative GameState. Detects
player actions by observing state changes between frames and emits events
on significant state transitions.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

from src.common.logging import get_logger
from src.common.performance import LatencyTracker, PerfTimer
from src.detection.card import Card
from src.detection.ocr_engine import OCRResult
from src.detection.player_identifier import PlayerMatch
from src.detection.validation import DetectionResult
from src.engine.game_state import (
    STREET_COMMUNITY_CARDS,
    ActionType,
    GameState,
    Player,
    PlayerAction,
    Street,
)
from src.engine.state_validator import GameStateValidator

logger = get_logger("engine.coordinator")


class StateChangeType(Enum):
    """Types of state changes that can be emitted as events."""

    NEW_HAND = auto()
    STREET_CHANGE = auto()
    PLAYER_ACTION = auto()
    COMMUNITY_CARDS_UPDATED = auto()
    HOLE_CARDS_DETECTED = auto()
    STATE_UPDATED = auto()


@dataclass(frozen=True)
class StateChangeEvent:
    """Event emitted when the game state changes significantly.

    Attributes:
        change_type: The kind of state change.
        state: Snapshot of the game state after the change.
        details: Additional structured context about the change.
        timestamp: Unix timestamp when the event was created.
    """

    change_type: StateChangeType
    state: GameState
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class _PlayerSnapshot:
    """Snapshot of a player's observable state for change detection.

    Used to compare states between frames and infer player actions.
    """

    seat: int
    name: str
    stack_size: float
    current_bet: float
    is_active: bool
    has_cards: bool


# Type alias for event callbacks.
EventCallback = Callable[[StateChangeEvent], None]


class GameStateCoordinator:
    """Central orchestrator for game state management.

    Receives detection results from the vision pipeline and maintains the
    current GameState. Coordinates updates from multiple sources:

    - Card detections update community/hole cards
    - OCR results update bet amounts and stack sizes
    - Player identification maps seats to player names

    Detects player actions by observing state changes between frames:

    - Stack decreased + bet appeared -> player bet/raised
    - Cards gone + no bet -> player folded
    - Stack unchanged + existing bet matched -> player called

    Emits events on significant state changes (new hand, street change,
    player action) via registered callbacks.

    Args:
        big_blind: Big blind amount for the table.
        small_blind: Small blind amount for the table.
        num_seats: Number of seats at the table.
        hero_seat: Seat number of the hero (user).
        validator: Optional GameStateValidator instance. If None, a default
            validator is created.
    """

    def __init__(
        self,
        big_blind: float = 2.0,
        small_blind: float = 1.0,
        num_seats: int = 6,
        hero_seat: int = 0,
        validator: Optional[GameStateValidator] = None,
    ) -> None:
        self._state = GameState(
            small_blind=small_blind,
            big_blind=big_blind,
            hero_seat=hero_seat,
        )
        self._num_seats = num_seats
        self._validator = validator or GameStateValidator(auto_correct=True)

        # Previous frame snapshots for change detection.
        self._prev_snapshots: dict[int, _PlayerSnapshot] = {}
        self._prev_community_cards: list[Card] = []
        self._prev_street: Street = Street.PREFLOP

        # Event callbacks.
        self._on_state_change: list[EventCallback] = []
        self._on_new_hand: list[EventCallback] = []
        self._on_action: list[EventCallback] = []

        # Performance tracking.
        self._latency_tracker = LatencyTracker("coordinator.process_frame")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        detection_result: DetectionResult,
        ocr_results: Optional[dict[int, OCRResult]] = None,
        player_ids: Optional[dict[int, PlayerMatch]] = None,
    ) -> GameState:
        """Process a single frame of detection data and update game state.

        This is the main entry point called once per captured frame. It
        integrates card detections, OCR results, and player identifications,
        then infers actions and validates the resulting state.

        Args:
            detection_result: Card detection results from the vision pipeline.
            ocr_results: Mapping of seat number to OCR result (bet amounts
                or stack sizes). Keys may also include special sentinel values
                (e.g., -1 for pot display).
            player_ids: Mapping of seat number to player identification result.

        Returns:
            The updated GameState after processing this frame.
        """
        with PerfTimer(
            "process_frame", logger=logger, tracker=self._latency_tracker
        ):
            ocr_results = ocr_results or {}
            player_ids = player_ids or {}

            # Step 1: Update player identities.
            self._update_player_identities(player_ids)

            # Step 2: Detect new hand.
            new_hand_detected = self._detect_new_hand(detection_result)
            if new_hand_detected:
                self._handle_new_hand()

            # Step 3: Update community cards and detect street changes.
            self._update_community_cards(detection_result)

            # Step 4: Update player state from OCR.
            self._update_from_ocr(ocr_results)

            # Step 5: Update hole cards.
            self._update_hole_cards(detection_result)

            # Step 6: Infer player actions from state changes.
            self._infer_actions()

            # Step 7: Take new snapshots.
            self._take_snapshots()

            # Step 8: Validate state.
            validation_errors = self._validator.validate(self._state)
            if validation_errors:
                logger.debug(
                    "validation_after_frame",
                    error_count=len(validation_errors),
                )

            # Step 9: Emit general state change event.
            self._emit_event(
                StateChangeType.STATE_UPDATED,
                details={"validation_errors": len(validation_errors)},
            )

            self._state.timestamp = time.time()

        return self.get_current_state()

    def get_current_state(self) -> GameState:
        """Return a deep copy of the current game state.

        Returns:
            A copy of the current GameState, safe to mutate externally.
        """
        return copy.deepcopy(self._state)

    def on_state_change(self, callback: EventCallback) -> None:
        """Register a callback for any state change event.

        Args:
            callback: Function called with a StateChangeEvent on each update.
        """
        self._on_state_change.append(callback)

    def on_new_hand(self, callback: EventCallback) -> None:
        """Register a callback for new hand events.

        Args:
            callback: Function called with a StateChangeEvent when a new
                hand is detected.
        """
        self._on_new_hand.append(callback)

    def on_action(self, callback: EventCallback) -> None:
        """Register a callback for player action events.

        Args:
            callback: Function called with a StateChangeEvent when a player
                action is inferred.
        """
        self._on_action.append(callback)

    def reset(self) -> None:
        """Reset the coordinator to its initial state.

        Clears the game state, snapshots, and all tracking data.
        """
        self._state = GameState(
            small_blind=self._state.small_blind,
            big_blind=self._state.big_blind,
            hero_seat=self._state.hero_seat,
        )
        self._prev_snapshots.clear()
        self._prev_community_cards.clear()
        self._prev_street = Street.PREFLOP
        self._latency_tracker.reset()

    # ------------------------------------------------------------------
    # Internal: player identity updates
    # ------------------------------------------------------------------

    def _update_player_identities(
        self, player_ids: dict[int, PlayerMatch]
    ) -> None:
        """Map seat numbers to player names from identification results."""
        for seat, match in player_ids.items():
            if seat < 0 or seat >= self._num_seats:
                continue

            player = self._state.get_player_by_seat(seat)
            if player is None:
                # New player detected at this seat.
                new_player = Player(
                    name=match.name,
                    seat_number=seat,
                    stack_size=0.0,
                )
                self._state.players.append(new_player)
                logger.info(
                    "player_detected",
                    seat=seat,
                    name=match.name,
                    confidence=match.confidence,
                )
            elif player.name != match.name and match.confidence > 0.8:
                # Name changed — update if high confidence.
                logger.info(
                    "player_name_updated",
                    seat=seat,
                    old_name=player.name,
                    new_name=match.name,
                )
                player.name = match.name

    # ------------------------------------------------------------------
    # Internal: new hand detection
    # ------------------------------------------------------------------

    def _detect_new_hand(self, detection_result: DetectionResult) -> bool:
        """Detect whether a new hand has started.

        Heuristics:
        - Community cards went from >0 to 0 (board cleared).
        - We previously had community cards but now have none and we're
          seeing hole cards dealt.
        """
        current_community = detection_result.get_community_cards()

        if (
            len(self._prev_community_cards) > 0
            and len(current_community) == 0
        ):
            return True

        return False

    def _handle_new_hand(self) -> None:
        """Reset state for a new hand and emit event."""
        logger.info(
            "new_hand_detected",
            hand_number=self._state.hand_number + 1,
        )
        self._state.reset_for_new_hand()
        self._prev_snapshots.clear()
        self._prev_community_cards.clear()
        self._prev_street = Street.PREFLOP

        self._emit_event(
            StateChangeType.NEW_HAND,
            details={"hand_number": self._state.hand_number},
        )

    # ------------------------------------------------------------------
    # Internal: community card and street updates
    # ------------------------------------------------------------------

    def _update_community_cards(
        self, detection_result: DetectionResult
    ) -> None:
        """Update community cards and detect street changes."""
        detected_cards = detection_result.get_community_cards()

        if not detected_cards:
            return

        current_count = len(self._state.community_cards)
        detected_count = len(detected_cards)

        if detected_count <= current_count:
            # No new cards or fewer cards (possible detection miss).
            return

        # Determine the target street based on detected card count.
        target_street = self._street_for_card_count(detected_count)
        if target_street is None:
            logger.warning(
                "invalid_community_card_count",
                count=detected_count,
            )
            return

        # Advance street if needed.
        while (
            self._state.current_street != target_street
            and self._state.current_street != Street.SHOWDOWN
        ):
            old_street = self._state.current_street
            try:
                self._state.advance_street()
            except ValueError:
                break

            logger.info(
                "street_advanced",
                from_street=old_street.name,
                to_street=self._state.current_street.name,
            )
            self._emit_event(
                StateChangeType.STREET_CHANGE,
                details={
                    "from_street": old_street.name,
                    "to_street": self._state.current_street.name,
                },
            )

        # Set community cards directly to the detected ones.
        self._state.community_cards = list(detected_cards)
        self._prev_community_cards = list(detected_cards)

        self._emit_event(
            StateChangeType.COMMUNITY_CARDS_UPDATED,
            details={
                "card_count": detected_count,
                "street": self._state.current_street.name,
            },
        )

    @staticmethod
    def _street_for_card_count(count: int) -> Optional[Street]:
        """Return the street corresponding to a community card count.

        Args:
            count: Number of community cards detected.

        Returns:
            The corresponding Street, or None if the count is invalid.
        """
        for street, expected in STREET_COMMUNITY_CARDS.items():
            if expected == count:
                return street
        return None

    # ------------------------------------------------------------------
    # Internal: OCR updates
    # ------------------------------------------------------------------

    def _update_from_ocr(
        self, ocr_results: dict[int, OCRResult]
    ) -> None:
        """Update player stacks and bets from OCR results.

        Args:
            ocr_results: Mapping of seat number to OCR result. Negative
                seat numbers are reserved for special regions (e.g., pot).
        """
        for seat, result in ocr_results.items():
            if result.value is None:
                continue

            if seat == -1:
                # Special: pot display.
                # We don't overwrite pot_size from OCR directly since it's
                # tracked via actions, but we can use it for validation.
                continue

            player = self._state.get_player_by_seat(seat)
            if player is None:
                continue

            # Use OCR value as stack size update.
            if result.confidence >= 0.7:
                player.stack_size = result.value

    # ------------------------------------------------------------------
    # Internal: hole card updates
    # ------------------------------------------------------------------

    def _update_hole_cards(self, detection_result: DetectionResult) -> None:
        """Update player hole cards from detection results."""
        for seat, cards in detection_result.player_cards.items():
            if len(cards) != 2:
                continue

            player = self._state.get_player_by_seat(seat)
            if player is None:
                continue

            if player.hole_cards != cards:
                player.hole_cards = list(cards)
                logger.info(
                    "hole_cards_detected",
                    seat=seat,
                    cards=[str(c) for c in cards],
                )
                self._emit_event(
                    StateChangeType.HOLE_CARDS_DETECTED,
                    details={
                        "seat": seat,
                        "cards": [str(c) for c in cards],
                    },
                )

    # ------------------------------------------------------------------
    # Internal: action inference
    # ------------------------------------------------------------------

    def _infer_actions(self) -> None:
        """Infer player actions by comparing current state to previous snapshots.

        Heuristics:
        - Stack decreased + new bet appeared -> player bet/raised.
        - Player was active but now inactive (no cards) -> player folded.
        - Stack decreased + bet matches existing bet -> player called.
        """
        if not self._prev_snapshots:
            return

        for player in self._state.players:
            prev = self._prev_snapshots.get(player.seat_number)
            if prev is None:
                continue

            action = self._detect_action(prev, player)
            if action is not None:
                self._state.record_action(player.seat_number, action)
                logger.info(
                    "action_inferred",
                    seat=player.seat_number,
                    player=player.name,
                    action=action.action_type.value,
                    amount=action.amount,
                )
                self._emit_event(
                    StateChangeType.PLAYER_ACTION,
                    details={
                        "seat": player.seat_number,
                        "player": player.name,
                        "action": action.action_type.value,
                        "amount": action.amount,
                    },
                )

    def _detect_action(
        self, prev: _PlayerSnapshot, current: Player
    ) -> Optional[PlayerAction]:
        """Detect a single player action by comparing snapshots.

        Args:
            prev: Previous frame snapshot for this player.
            current: Current player state.

        Returns:
            Inferred PlayerAction, or None if no action detected.
        """
        # Player was active but is now inactive (folded).
        if prev.is_active and not current.is_active:
            return PlayerAction(
                action_type=ActionType.FOLD,
                amount=0.0,
                street=self._state.current_street,
            )

        # Stack decreased — player put chips in.
        stack_diff = prev.stack_size - current.stack_size
        if stack_diff > _CHIP_TOLERANCE:
            if current.stack_size <= _CHIP_TOLERANCE:
                # All-in.
                return PlayerAction(
                    action_type=ActionType.ALL_IN,
                    amount=stack_diff,
                    street=self._state.current_street,
                )

            # Determine if this is a bet, raise, or call.
            max_existing_bet = self._get_max_current_bet(
                exclude_seat=current.seat_number
            )

            if max_existing_bet <= _CHIP_TOLERANCE:
                # No existing bet — this is an opening bet.
                return PlayerAction(
                    action_type=ActionType.BET,
                    amount=stack_diff,
                    street=self._state.current_street,
                )

            if current.current_bet > max_existing_bet + _CHIP_TOLERANCE:
                # Bet exceeds the current max — this is a raise.
                return PlayerAction(
                    action_type=ActionType.RAISE,
                    amount=stack_diff,
                    street=self._state.current_street,
                )

            # Bet matches existing — this is a call.
            return PlayerAction(
                action_type=ActionType.CALL,
                amount=stack_diff,
                street=self._state.current_street,
            )

        return None

    def _get_max_current_bet(self, exclude_seat: int = -1) -> float:
        """Return the maximum current bet among all players.

        Args:
            exclude_seat: Seat number to exclude from the search.

        Returns:
            Maximum current bet amount.
        """
        max_bet = 0.0
        for player in self._state.players:
            if player.seat_number == exclude_seat:
                continue
            if player.current_bet > max_bet:
                max_bet = player.current_bet
        return max_bet

    # ------------------------------------------------------------------
    # Internal: snapshot management
    # ------------------------------------------------------------------

    def _take_snapshots(self) -> None:
        """Capture current player states for next-frame comparison."""
        self._prev_snapshots.clear()
        for player in self._state.players:
            self._prev_snapshots[player.seat_number] = _PlayerSnapshot(
                seat=player.seat_number,
                name=player.name,
                stack_size=player.stack_size,
                current_bet=player.current_bet,
                is_active=player.is_active,
                has_cards=player.hole_cards is not None,
            )
        self._prev_community_cards = list(self._state.community_cards)
        self._prev_street = self._state.current_street

    # ------------------------------------------------------------------
    # Internal: event emission
    # ------------------------------------------------------------------

    def _emit_event(
        self,
        change_type: StateChangeType,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Create and dispatch a state change event to registered callbacks.

        Args:
            change_type: The type of state change.
            details: Additional context about the change.
        """
        event = StateChangeEvent(
            change_type=change_type,
            state=copy.deepcopy(self._state),
            details=details or {},
        )

        # Dispatch to type-specific callbacks.
        if change_type == StateChangeType.NEW_HAND:
            for cb in self._on_new_hand:
                self._safe_call(cb, event)
        elif change_type == StateChangeType.PLAYER_ACTION:
            for cb in self._on_action:
                self._safe_call(cb, event)

        # Always dispatch to general callbacks.
        for cb in self._on_state_change:
            self._safe_call(cb, event)

    @staticmethod
    def _safe_call(callback: EventCallback, event: StateChangeEvent) -> None:
        """Call a callback, catching and logging any exceptions.

        Args:
            callback: The callback to invoke.
            event: The event to pass.
        """
        try:
            callback(event)
        except Exception:
            logger.exception(
                "event_callback_error",
                change_type=event.change_type.name,
            )


# Tolerance for floating-point chip comparisons.
_CHIP_TOLERANCE = 0.01
