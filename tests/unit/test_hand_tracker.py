"""Unit tests for HandPhaseTracker (S3-02)."""

from __future__ import annotations

from src.detection.card import Card, Rank, Suit
from src.detection.validation import DetectionResult
from src.engine.game_state import GameState, Player, Street
from src.engine.hand_tracker import HandPhaseTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _card(rank_ch: str, suit_ch: str) -> Card:
    """Shorthand card factory: _card('A', 'h') -> Ace of Hearts."""
    rank_map = {
        "2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE,
        "6": Rank.SIX, "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE,
        "T": Rank.TEN, "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING,
        "A": Rank.ACE,
    }
    suit_map = {
        "h": Suit.HEARTS, "d": Suit.DIAMONDS,
        "c": Suit.CLUBS, "s": Suit.SPADES,
    }
    return Card(rank=rank_map[rank_ch], suit=suit_map[suit_ch])


def _detection(cards: list[Card]) -> DetectionResult:
    """Build a DetectionResult with the given community cards."""
    return DetectionResult(community_cards=cards)


# Reusable card sets.
FLOP = [_card("J", "s"), _card("7", "h"), _card("2", "c")]
TURN = FLOP + [_card("9", "d")]
RIVER = TURN + [_card("3", "s")]


# ---------------------------------------------------------------------------
# Tests: basic street transitions
# ---------------------------------------------------------------------------


class TestStreetTransitions:
    """Test that community card count changes trigger correct transitions."""

    def test_initial_state_is_preflop(self) -> None:
        tracker = HandPhaseTracker()
        assert tracker.current_street == Street.PREFLOP
        assert not tracker.hand_active

    def test_preflop_to_flop(self) -> None:
        tracker = HandPhaseTracker()
        result = tracker.update(_detection(FLOP))
        assert result == Street.FLOP
        assert tracker.current_street == Street.FLOP

    def test_flop_to_turn(self) -> None:
        tracker = HandPhaseTracker()
        tracker.update(_detection(FLOP))
        result = tracker.update(_detection(TURN))
        assert result == Street.TURN
        assert tracker.current_street == Street.TURN

    def test_turn_to_river(self) -> None:
        tracker = HandPhaseTracker()
        tracker.update(_detection(FLOP))
        tracker.update(_detection(TURN))
        result = tracker.update(_detection(RIVER))
        assert result == Street.RIVER
        assert tracker.current_street == Street.RIVER

    def test_full_hand_progression(self) -> None:
        tracker = HandPhaseTracker()
        assert tracker.update(_detection(FLOP)) == Street.FLOP
        assert tracker.update(_detection(TURN)) == Street.TURN
        assert tracker.update(_detection(RIVER)) == Street.RIVER

    def test_same_street_returns_none(self) -> None:
        tracker = HandPhaseTracker()
        tracker.update(_detection(FLOP))
        result = tracker.update(_detection(FLOP))
        assert result is None
        assert tracker.current_street == Street.FLOP

    def test_preflop_no_cards_stays_preflop(self) -> None:
        tracker = HandPhaseTracker()
        result = tracker.update(_detection([]))
        # Not yet active, no transition.
        assert result is None
        assert tracker.current_street == Street.PREFLOP


# ---------------------------------------------------------------------------
# Tests: new hand detection
# ---------------------------------------------------------------------------


class TestNewHandDetection:
    """Test detection of new hand starts."""

    def test_new_hand_after_community_reset(self) -> None:
        tracker = HandPhaseTracker(reset_frame_threshold=2)
        tracker.update(_detection(FLOP))  # Activates hand.
        # Two consecutive empty frames trigger reset.
        tracker.update(_detection([]))
        result = tracker.update(_detection([]))
        assert result == Street.PREFLOP

    def test_single_empty_frame_does_not_reset(self) -> None:
        tracker = HandPhaseTracker(reset_frame_threshold=2)
        tracker.update(_detection(FLOP))
        result = tracker.update(_detection([]))
        assert result is None  # Not enough consecutive resets.
        assert tracker.current_street == Street.FLOP

    def test_reset_threshold_configurable(self) -> None:
        tracker = HandPhaseTracker(reset_frame_threshold=3)
        tracker.update(_detection(FLOP))
        tracker.update(_detection([]))
        tracker.update(_detection([]))
        assert tracker.current_street == Street.FLOP  # Still waiting.
        tracker.update(_detection([]))
        assert tracker.current_street == Street.PREFLOP  # Now reset.

    def test_is_new_hand(self) -> None:
        tracker = HandPhaseTracker(reset_frame_threshold=1)
        tracker.update(_detection(FLOP))
        tracker.update(_detection([]))
        assert tracker.is_new_hand()


# ---------------------------------------------------------------------------
# Tests: callbacks
# ---------------------------------------------------------------------------


class TestCallbacks:
    """Test callback registration and invocation."""

    def test_street_transition_callback_fires(self) -> None:
        transitions: list[tuple[Street, Street]] = []
        tracker = HandPhaseTracker()
        tracker.on_street_transition(
            lambda old, new: transitions.append((old, new))
        )
        tracker.update(_detection(FLOP))
        assert transitions == [(Street.PREFLOP, Street.FLOP)]

    def test_multiple_callbacks(self) -> None:
        calls_a: list[Street] = []
        calls_b: list[Street] = []
        tracker = HandPhaseTracker()
        tracker.on_street_transition(lambda o, n: calls_a.append(n))
        tracker.on_street_transition(lambda o, n: calls_b.append(n))
        tracker.update(_detection(FLOP))
        assert calls_a == [Street.FLOP]
        assert calls_b == [Street.FLOP]

    def test_new_hand_callback_fires(self) -> None:
        new_hand_count = [0]
        tracker = HandPhaseTracker(reset_frame_threshold=1)
        tracker.on_new_hand(
            lambda: new_hand_count.__setitem__(0, new_hand_count[0] + 1)
        )
        tracker.update(_detection(FLOP))
        tracker.update(_detection([]))
        assert new_hand_count[0] == 1

    def test_callback_exception_does_not_crash(self) -> None:
        """A misbehaving callback should not prevent further processing."""
        tracker = HandPhaseTracker()
        def _bad_callback(o: Street, n: Street) -> None:
            raise RuntimeError("boom")

        tracker.on_street_transition(_bad_callback)
        # Should not raise.
        tracker.update(_detection(FLOP))
        assert tracker.current_street == Street.FLOP


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases: missed frames, skipped streets, invalid counts."""

    def test_skipped_street_emits_intermediate_transitions(self) -> None:
        """If FLOP is missed, going from 0 to 4 cards should fire both
        PREFLOP->FLOP and FLOP->TURN transitions."""
        transitions: list[tuple[Street, Street]] = []
        tracker = HandPhaseTracker()
        tracker.on_street_transition(
            lambda old, new: transitions.append((old, new))
        )
        tracker.update(_detection(TURN))
        assert transitions == [
            (Street.PREFLOP, Street.FLOP),
            (Street.FLOP, Street.TURN),
        ]
        assert tracker.current_street == Street.TURN

    def test_invalid_card_count_ignored(self) -> None:
        """1 or 2 community cards is not a valid state; should be ignored."""
        tracker = HandPhaseTracker()
        one_card = [_card("A", "h")]
        result = tracker.update(_detection(one_card))
        assert result is None
        assert tracker.current_street == Street.PREFLOP

    def test_two_cards_ignored(self) -> None:
        tracker = HandPhaseTracker()
        two_cards = [_card("A", "h"), _card("K", "d")]
        result = tracker.update(_detection(two_cards))
        assert result is None

    def test_betting_round_tracking(self) -> None:
        tracker = HandPhaseTracker()
        assert tracker.betting_round == 0
        tracker.advance_betting_round()
        assert tracker.betting_round == 1
        tracker.update(_detection(FLOP))  # Resets betting round.
        assert tracker.betting_round == 0

    def test_reset_clears_state(self) -> None:
        tracker = HandPhaseTracker()
        tracker.update(_detection(FLOP))
        tracker.reset()
        assert tracker.current_street == Street.PREFLOP
        assert not tracker.hand_active
        assert tracker.previous_community_cards == []


# ---------------------------------------------------------------------------
# Tests: GameState integration
# ---------------------------------------------------------------------------


class TestGameStateIntegration:
    """Test that HandPhaseTracker syncs with a GameState."""

    def _make_game_state(self) -> GameState:
        return GameState(
            players=[
                Player(name="Hero", seat_number=1),
                Player(name="Villain", seat_number=2),
            ],
        )

    def test_advance_street_syncs_game_state(self) -> None:
        gs = self._make_game_state()
        tracker = HandPhaseTracker(game_state=gs)
        tracker.update(_detection(FLOP))
        assert gs.current_street == Street.FLOP

    def test_new_hand_resets_game_state(self) -> None:
        gs = self._make_game_state()
        tracker = HandPhaseTracker(
            game_state=gs, reset_frame_threshold=1
        )
        tracker.update(_detection(FLOP))
        assert gs.current_street == Street.FLOP
        tracker.update(_detection([]))
        assert gs.current_street == Street.PREFLOP

    def test_full_progression_with_game_state(self) -> None:
        gs = self._make_game_state()
        tracker = HandPhaseTracker(game_state=gs)
        tracker.update(_detection(FLOP))
        assert gs.current_street == Street.FLOP
        tracker.update(_detection(TURN))
        assert gs.current_street == Street.TURN
        tracker.update(_detection(RIVER))
        assert gs.current_street == Street.RIVER
