"""Unit tests for src.engine.coordinator module."""

from __future__ import annotations

from src.detection.card import Card, Rank, Suit
from src.detection.ocr_engine import OCRResult
from src.detection.player_identifier import PlayerMatch
from src.detection.validation import DetectionResult
from src.engine.coordinator import (
    GameStateCoordinator,
    StateChangeEvent,
    StateChangeType,
)
from src.engine.game_state import ActionType, Street
from src.engine.state_validator import GameStateValidator

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

ACE_SPADES = Card(Rank.ACE, Suit.SPADES)
KING_HEARTS = Card(Rank.KING, Suit.HEARTS)
QUEEN_DIAMONDS = Card(Rank.QUEEN, Suit.DIAMONDS)
JACK_CLUBS = Card(Rank.JACK, Suit.CLUBS)
TEN_SPADES = Card(Rank.TEN, Suit.SPADES)
FIVE_HEARTS = Card(Rank.FIVE, Suit.HEARTS)
TWO_CLUBS = Card(Rank.TWO, Suit.CLUBS)


def _empty_detection() -> DetectionResult:
    """Return an empty DetectionResult with no cards detected."""
    return DetectionResult()


def _flop_detection(
    community: list[Card] | None = None,
    player_cards: dict[int, list[Card]] | None = None,
) -> DetectionResult:
    """Return a DetectionResult with given cards."""
    return DetectionResult(
        community_cards=community or [],
        player_cards=player_cards or {},
    )


def _make_coordinator(**kwargs) -> GameStateCoordinator:
    """Create a coordinator with sensible test defaults."""
    defaults = {
        "big_blind": 2.0,
        "small_blind": 1.0,
        "num_seats": 6,
        "hero_seat": 0,
    }
    defaults.update(kwargs)
    return GameStateCoordinator(**defaults)


# -----------------------------------------------------------------------
# Tests: basic construction
# -----------------------------------------------------------------------


class TestCoordinatorInit:
    def test_default_state(self) -> None:
        coord = _make_coordinator()
        state = coord.get_current_state()
        assert state.small_blind == 1.0
        assert state.big_blind == 2.0
        assert state.hero_seat == 0
        assert state.current_street == Street.PREFLOP
        assert state.players == []

    def test_get_current_state_returns_copy(self) -> None:
        coord = _make_coordinator()
        state1 = coord.get_current_state()
        state2 = coord.get_current_state()
        assert state1 is not state2


# -----------------------------------------------------------------------
# Tests: player identity updates
# -----------------------------------------------------------------------


class TestPlayerIdentityUpdates:
    def test_new_player_registered(self) -> None:
        coord = _make_coordinator()
        player_ids = {
            0: PlayerMatch(name="Hero", raw_text="Hero", confidence=0.95),
            1: PlayerMatch(name="Villain", raw_text="Villain", confidence=0.90),
        }
        coord.process_frame(_empty_detection(), player_ids=player_ids)

        state = coord.get_current_state()
        assert len(state.players) == 2
        assert state.get_player_by_seat(0) is not None
        assert state.get_player_by_seat(0).name == "Hero"
        assert state.get_player_by_seat(1).name == "Villain"

    def test_player_name_updated_high_confidence(self) -> None:
        coord = _make_coordinator()
        # First frame: detect player.
        ids1 = {0: PlayerMatch(name="OldName", raw_text="OldName", confidence=0.9)}
        coord.process_frame(_empty_detection(), player_ids=ids1)

        # Second frame: higher confidence name update.
        ids2 = {0: PlayerMatch(name="NewName", raw_text="NewName", confidence=0.95)}
        coord.process_frame(_empty_detection(), player_ids=ids2)

        state = coord.get_current_state()
        assert state.get_player_by_seat(0).name == "NewName"

    def test_player_name_not_updated_low_confidence(self) -> None:
        coord = _make_coordinator()
        ids1 = {
            0: PlayerMatch(
                name="CorrectName", raw_text="CorrectName", confidence=0.9
            ),
        }
        coord.process_frame(_empty_detection(), player_ids=ids1)

        ids2 = {0: PlayerMatch(name="WrongName", raw_text="WrongName", confidence=0.5)}
        coord.process_frame(_empty_detection(), player_ids=ids2)

        state = coord.get_current_state()
        assert state.get_player_by_seat(0).name == "CorrectName"

    def test_invalid_seat_ignored(self) -> None:
        coord = _make_coordinator(num_seats=6)
        ids = {99: PlayerMatch(name="Ghost", raw_text="Ghost", confidence=0.9)}
        coord.process_frame(_empty_detection(), player_ids=ids)

        state = coord.get_current_state()
        assert len(state.players) == 0


# -----------------------------------------------------------------------
# Tests: community card and street detection
# -----------------------------------------------------------------------


class TestCommunityCardUpdates:
    def test_flop_detected(self) -> None:
        coord = _make_coordinator()
        detection = _flop_detection(
            community=[ACE_SPADES, KING_HEARTS, QUEEN_DIAMONDS]
        )
        coord.process_frame(detection)

        state = coord.get_current_state()
        assert len(state.community_cards) == 3
        assert state.current_street == Street.FLOP

    def test_turn_detected(self) -> None:
        coord = _make_coordinator()
        # First: flop.
        coord.process_frame(
            _flop_detection(community=[ACE_SPADES, KING_HEARTS, QUEEN_DIAMONDS])
        )
        # Then: turn.
        turn_board = [ACE_SPADES, KING_HEARTS, QUEEN_DIAMONDS, JACK_CLUBS]
        coord.process_frame(_flop_detection(community=turn_board))

        state = coord.get_current_state()
        assert len(state.community_cards) == 4
        assert state.current_street == Street.TURN

    def test_river_detected(self) -> None:
        coord = _make_coordinator()
        flop = [ACE_SPADES, KING_HEARTS, QUEEN_DIAMONDS]
        turn = flop + [JACK_CLUBS]
        river = turn + [TEN_SPADES]
        coord.process_frame(_flop_detection(community=flop))
        coord.process_frame(_flop_detection(community=turn))
        coord.process_frame(_flop_detection(community=river))

        state = coord.get_current_state()
        assert len(state.community_cards) == 5
        assert state.current_street == Street.RIVER

    def test_no_new_cards_does_not_advance(self) -> None:
        coord = _make_coordinator()
        coord.process_frame(
            _flop_detection(community=[ACE_SPADES, KING_HEARTS, QUEEN_DIAMONDS])
        )
        # Same cards again.
        coord.process_frame(
            _flop_detection(community=[ACE_SPADES, KING_HEARTS, QUEEN_DIAMONDS])
        )

        state = coord.get_current_state()
        assert state.current_street == Street.FLOP

    def test_empty_detection_no_change(self) -> None:
        coord = _make_coordinator()
        coord.process_frame(_empty_detection())
        state = coord.get_current_state()
        assert state.current_street == Street.PREFLOP
        assert state.community_cards == []


# -----------------------------------------------------------------------
# Tests: new hand detection
# -----------------------------------------------------------------------


class TestNewHandDetection:
    def test_board_cleared_triggers_new_hand(self) -> None:
        coord = _make_coordinator()
        # First hand: flop.
        coord.process_frame(
            _flop_detection(community=[ACE_SPADES, KING_HEARTS, QUEEN_DIAMONDS])
        )
        state1 = coord.get_current_state()
        hand_num1 = state1.hand_number

        # Board clears -> new hand.
        coord.process_frame(_empty_detection())

        state2 = coord.get_current_state()
        assert state2.hand_number == hand_num1 + 1
        assert state2.current_street == Street.PREFLOP
        assert state2.community_cards == []

    def test_new_hand_callback_called(self) -> None:
        coord = _make_coordinator()
        events: list[StateChangeEvent] = []
        coord.on_new_hand(events.append)

        coord.process_frame(
            _flop_detection(community=[ACE_SPADES, KING_HEARTS, QUEEN_DIAMONDS])
        )
        coord.process_frame(_empty_detection())

        new_hand_events = [
            e for e in events
            if e.change_type == StateChangeType.NEW_HAND
        ]
        assert len(new_hand_events) == 1


# -----------------------------------------------------------------------
# Tests: OCR updates
# -----------------------------------------------------------------------


class TestOCRUpdates:
    def test_stack_size_updated_from_ocr(self) -> None:
        coord = _make_coordinator()
        # Register a player first.
        ids = {0: PlayerMatch(name="Hero", raw_text="Hero", confidence=0.9)}
        coord.process_frame(_empty_detection(), player_ids=ids)

        # OCR provides stack size.
        ocr = {0: OCRResult(raw_text="$150.00", value=150.0, confidence=0.85)}
        coord.process_frame(_empty_detection(), ocr_results=ocr)

        state = coord.get_current_state()
        assert state.get_player_by_seat(0).stack_size == 150.0

    def test_low_confidence_ocr_ignored(self) -> None:
        coord = _make_coordinator()
        ids = {0: PlayerMatch(name="Hero", raw_text="Hero", confidence=0.9)}
        coord.process_frame(_empty_detection(), player_ids=ids)

        # Set initial stack.
        ocr_good = {0: OCRResult(raw_text="$100", value=100.0, confidence=0.9)}
        coord.process_frame(_empty_detection(), ocr_results=ocr_good)

        # Low confidence OCR should not update.
        ocr_bad = {0: OCRResult(raw_text="$999", value=999.0, confidence=0.3)}
        coord.process_frame(_empty_detection(), ocr_results=ocr_bad)

        state = coord.get_current_state()
        assert state.get_player_by_seat(0).stack_size == 100.0

    def test_ocr_no_value_ignored(self) -> None:
        coord = _make_coordinator()
        ids = {0: PlayerMatch(name="Hero", raw_text="Hero", confidence=0.9)}
        coord.process_frame(_empty_detection(), player_ids=ids)

        ocr = {0: OCRResult(raw_text="???", value=None, confidence=0.5)}
        coord.process_frame(_empty_detection(), ocr_results=ocr)

        state = coord.get_current_state()
        # Stack stays at default 0 since no valid OCR.
        assert state.get_player_by_seat(0).stack_size == 0.0


# -----------------------------------------------------------------------
# Tests: hole card updates
# -----------------------------------------------------------------------


class TestHoleCardUpdates:
    def test_hole_cards_detected(self) -> None:
        coord = _make_coordinator()
        ids = {0: PlayerMatch(name="Hero", raw_text="Hero", confidence=0.9)}
        coord.process_frame(_empty_detection(), player_ids=ids)

        detection = _flop_detection(
            player_cards={0: [ACE_SPADES, KING_HEARTS]}
        )
        coord.process_frame(detection)

        state = coord.get_current_state()
        hero = state.get_player_by_seat(0)
        assert hero.hole_cards is not None
        assert len(hero.hole_cards) == 2
        assert hero.hole_cards[0] == ACE_SPADES

    def test_single_hole_card_ignored(self) -> None:
        coord = _make_coordinator()
        ids = {0: PlayerMatch(name="Hero", raw_text="Hero", confidence=0.9)}
        coord.process_frame(_empty_detection(), player_ids=ids)

        detection = _flop_detection(
            player_cards={0: [ACE_SPADES]}  # Only 1 card -- invalid.
        )
        coord.process_frame(detection)

        state = coord.get_current_state()
        assert state.get_player_by_seat(0).hole_cards is None


# -----------------------------------------------------------------------
# Tests: action inference
# -----------------------------------------------------------------------


class TestActionInference:
    def _setup_players(self, coord: GameStateCoordinator) -> None:
        """Register players and set up initial stacks."""
        ids = {
            0: PlayerMatch(name="Hero", raw_text="Hero", confidence=0.9),
            1: PlayerMatch(name="Villain", raw_text="Villain", confidence=0.9),
        }
        ocr = {
            0: OCRResult(raw_text="$100", value=100.0, confidence=0.9),
            1: OCRResult(raw_text="$100", value=100.0, confidence=0.9),
        }
        coord.process_frame(_empty_detection(), ocr_results=ocr, player_ids=ids)

    def test_bet_inferred_from_stack_decrease(self) -> None:
        coord = _make_coordinator()
        self._setup_players(coord)

        # Player 0 stack decreases, current_bet increases -> bet.
        ocr = {0: OCRResult(raw_text="$90", value=90.0, confidence=0.9)}
        coord.process_frame(_empty_detection(), ocr_results=ocr)

        state = coord.get_current_state()
        # Action should have been inferred.
        assert len(state.action_history) >= 1
        last_action = state.action_history[-1]
        assert last_action.action_type == ActionType.BET
        assert last_action.amount == 10.0

    def test_fold_inferred(self) -> None:
        coord = _make_coordinator()
        self._setup_players(coord)

        # Manually mark player as inactive between frames.
        # Access internal state to simulate fold detection.
        coord._state.players[1].is_active = False

        coord.process_frame(_empty_detection())

        state = coord.get_current_state()
        fold_actions = [
            a for a in state.action_history
            if a.action_type == ActionType.FOLD
        ]
        assert len(fold_actions) == 1

    def test_all_in_inferred_from_zero_stack(self) -> None:
        coord = _make_coordinator()
        self._setup_players(coord)

        # Stack goes to zero -> all-in.
        ocr = {0: OCRResult(raw_text="$0", value=0.0, confidence=0.9)}
        coord.process_frame(_empty_detection(), ocr_results=ocr)

        state = coord.get_current_state()
        all_in_actions = [
            a for a in state.action_history
            if a.action_type == ActionType.ALL_IN
        ]
        assert len(all_in_actions) == 1
        assert all_in_actions[0].amount == 100.0

    def test_call_inferred_when_matching_existing_bet(self) -> None:
        coord = _make_coordinator()
        self._setup_players(coord)

        # Player 0 bets.
        ocr0 = {0: OCRResult(raw_text="$90", value=90.0, confidence=0.9)}
        coord.process_frame(_empty_detection(), ocr_results=ocr0)

        # Player 1 matches the bet (call).
        ocr1 = {1: OCRResult(raw_text="$90", value=90.0, confidence=0.9)}
        coord.process_frame(_empty_detection(), ocr_results=ocr1)

        state = coord.get_current_state()
        call_actions = [
            a for a in state.action_history
            if a.action_type == ActionType.CALL
        ]
        assert len(call_actions) == 1

    def test_no_action_when_no_change(self) -> None:
        coord = _make_coordinator()
        self._setup_players(coord)

        # Same state — no change.
        ocr = {
            0: OCRResult(raw_text="$100", value=100.0, confidence=0.9),
            1: OCRResult(raw_text="$100", value=100.0, confidence=0.9),
        }
        coord.process_frame(_empty_detection(), ocr_results=ocr)

        state = coord.get_current_state()
        assert len(state.action_history) == 0


# -----------------------------------------------------------------------
# Tests: event callbacks
# -----------------------------------------------------------------------


class TestEventCallbacks:
    def test_on_state_change_called(self) -> None:
        coord = _make_coordinator()
        events: list[StateChangeEvent] = []
        coord.on_state_change(events.append)

        coord.process_frame(_empty_detection())
        # At minimum, STATE_UPDATED should fire.
        state_events = [
            e for e in events
            if e.change_type == StateChangeType.STATE_UPDATED
        ]
        assert len(state_events) >= 1

    def test_on_action_callback(self) -> None:
        coord = _make_coordinator()
        action_events: list[StateChangeEvent] = []
        coord.on_action(action_events.append)

        # Register players with stacks.
        ids = {0: PlayerMatch(name="Hero", raw_text="Hero", confidence=0.9)}
        ocr = {0: OCRResult(raw_text="$100", value=100.0, confidence=0.9)}
        coord.process_frame(_empty_detection(), ocr_results=ocr, player_ids=ids)

        # Trigger a bet.
        ocr2 = {0: OCRResult(raw_text="$90", value=90.0, confidence=0.9)}
        coord.process_frame(_empty_detection(), ocr_results=ocr2)

        action_events_filtered = [
            e for e in action_events
            if e.change_type == StateChangeType.PLAYER_ACTION
        ]
        assert len(action_events_filtered) >= 1

    def test_street_change_event(self) -> None:
        coord = _make_coordinator()
        events: list[StateChangeEvent] = []
        coord.on_state_change(events.append)

        coord.process_frame(
            _flop_detection(community=[ACE_SPADES, KING_HEARTS, QUEEN_DIAMONDS])
        )

        street_events = [
            e for e in events
            if e.change_type == StateChangeType.STREET_CHANGE
        ]
        assert len(street_events) >= 1

    def test_callback_exception_does_not_propagate(self) -> None:
        coord = _make_coordinator()

        def bad_callback(event: StateChangeEvent) -> None:
            raise RuntimeError("callback error")

        coord.on_state_change(bad_callback)
        # Should not raise.
        coord.process_frame(_empty_detection())

    def test_multiple_callbacks(self) -> None:
        coord = _make_coordinator()
        calls_a: list[StateChangeEvent] = []
        calls_b: list[StateChangeEvent] = []

        coord.on_state_change(calls_a.append)
        coord.on_state_change(calls_b.append)

        coord.process_frame(_empty_detection())
        assert len(calls_a) > 0
        assert len(calls_b) > 0


# -----------------------------------------------------------------------
# Tests: reset
# -----------------------------------------------------------------------


class TestReset:
    def test_reset_clears_state(self) -> None:
        coord = _make_coordinator()
        ids = {0: PlayerMatch(name="Hero", raw_text="Hero", confidence=0.9)}
        coord.process_frame(
            _flop_detection(community=[ACE_SPADES, KING_HEARTS, QUEEN_DIAMONDS]),
            player_ids=ids,
        )

        coord.reset()

        state = coord.get_current_state()
        assert state.players == []
        assert state.community_cards == []
        assert state.current_street == Street.PREFLOP


# -----------------------------------------------------------------------
# Tests: integration scenario
# -----------------------------------------------------------------------


class TestIntegrationScenario:
    """Walk through a multi-frame sequence simulating a real hand."""

    def test_full_hand_flow(self) -> None:
        coord = _make_coordinator()
        all_events: list[StateChangeEvent] = []
        coord.on_state_change(all_events.append)

        # Frame 1: Players detected.
        ids = {
            0: PlayerMatch(name="Hero", raw_text="Hero", confidence=0.95),
            1: PlayerMatch(name="Villain1", raw_text="Villain1", confidence=0.90),
            2: PlayerMatch(name="Villain2", raw_text="Villain2", confidence=0.88),
        }
        ocr = {
            0: OCRResult(raw_text="$200", value=200.0, confidence=0.9),
            1: OCRResult(raw_text="$150", value=150.0, confidence=0.9),
            2: OCRResult(raw_text="$300", value=300.0, confidence=0.9),
        }
        coord.process_frame(_empty_detection(), ocr_results=ocr, player_ids=ids)

        state = coord.get_current_state()
        assert len(state.players) == 3
        assert state.get_player_by_seat(0).stack_size == 200.0

        # Frame 2: Flop appears with hero hole cards.
        flop = [ACE_SPADES, KING_HEARTS, QUEEN_DIAMONDS]
        detection = _flop_detection(
            community=flop,
            player_cards={0: [TWO_CLUBS, FIVE_HEARTS]},
        )
        coord.process_frame(detection, ocr_results=ocr)

        state = coord.get_current_state()
        assert state.current_street == Street.FLOP
        assert len(state.community_cards) == 3
        hero = state.get_player_by_seat(0)
        assert hero.hole_cards is not None
        assert len(hero.hole_cards) == 2

        # Frame 3: Turn card appears.
        turn_board = flop + [JACK_CLUBS]
        detection = _flop_detection(community=turn_board)
        coord.process_frame(detection, ocr_results=ocr)

        state = coord.get_current_state()
        assert state.current_street == Street.TURN
        assert len(state.community_cards) == 4

        # Verify events were emitted.
        street_events = [
            e for e in all_events
            if e.change_type == StateChangeType.STREET_CHANGE
        ]
        assert len(street_events) >= 2  # PREFLOP->FLOP, FLOP->TURN

    def test_validator_integration(self) -> None:
        """Verify the validator runs on each frame."""
        validator = GameStateValidator(auto_correct=True)
        coord = _make_coordinator(validator=validator)

        ids = {0: PlayerMatch(name="Hero", raw_text="Hero", confidence=0.9)}
        ocr = {0: OCRResult(raw_text="$100", value=100.0, confidence=0.9)}
        coord.process_frame(_empty_detection(), ocr_results=ocr, player_ids=ids)

        # State should be valid after processing.
        state = coord.get_current_state()
        errors = validator.validate(state)
        # No hard errors expected for a clean state.
        hard_errors = [
            e for e in errors
            if e.severity.name == "ERROR"
        ]
        assert len(hard_errors) == 0
