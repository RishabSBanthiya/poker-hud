"""Unit tests for the game state engine subsystem."""

from __future__ import annotations

from src.detection.card import Card, Rank, Suit
from src.engine.game_state import (
    ActionType,
    HandState,
    PlayerAction,
    PlayerState,
    Street,
)
from src.engine.hand_history import HandHistoryParser
from src.engine.hand_phase_tracker import HandPhaseTracker


class TestHandPhaseTracker:
    """Tests for street tracking based on community card count."""

    def test_initial_state_is_preflop(self) -> None:
        tracker = HandPhaseTracker()
        assert tracker.current_street == Street.PREFLOP

    def test_three_cards_is_flop(self) -> None:
        tracker = HandPhaseTracker()
        cards = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.DIAMONDS),
            Card(Rank.QUEEN, Suit.SPADES),
        ]
        result = tracker.update(cards)
        assert result == Street.FLOP
        assert tracker.current_street == Street.FLOP

    def test_four_cards_is_turn(self) -> None:
        tracker = HandPhaseTracker()
        cards = [Card(Rank.ACE, Suit.HEARTS)] * 4
        tracker.update(cards[:3])  # Flop first
        result = tracker.update(cards)
        assert result == Street.TURN

    def test_five_cards_is_river(self) -> None:
        tracker = HandPhaseTracker()
        cards = [Card(Rank.ACE, Suit.HEARTS)] * 5
        tracker.update(cards[:3])
        tracker.update(cards[:4])
        result = tracker.update(cards)
        assert result == Street.RIVER

    def test_no_transition_returns_none(self) -> None:
        tracker = HandPhaseTracker()
        result = tracker.update([])  # Still preflop
        assert result is None

    def test_reset(self) -> None:
        tracker = HandPhaseTracker()
        tracker.update([Card(Rank.ACE, Suit.HEARTS)] * 3)
        assert tracker.current_street == Street.FLOP
        tracker.reset()
        assert tracker.current_street == Street.PREFLOP


class TestHandState:
    """Tests for the HandState data model."""

    def test_get_player_found(self) -> None:
        hand = HandState(
            players=[
                PlayerState(name="Alice", seat_index=0),
                PlayerState(name="Bob", seat_index=1),
            ]
        )
        assert hand.get_player("Alice") is not None
        assert hand.get_player("Alice").name == "Alice"

    def test_get_player_not_found(self) -> None:
        hand = HandState(players=[])
        assert hand.get_player("Nobody") is None

    def test_get_active_players(self) -> None:
        hand = HandState(
            players=[
                PlayerState(name="Alice", seat_index=0, is_active=True),
                PlayerState(name="Bob", seat_index=1, is_active=False),
                PlayerState(name="Charlie", seat_index=2, is_active=True),
            ]
        )
        active = hand.get_active_players()
        assert len(active) == 2
        names = {p.name for p in active}
        assert names == {"Alice", "Charlie"}

    def test_get_actions_for_street(self) -> None:
        hand = HandState(
            actions=[
                PlayerAction("Alice", ActionType.RAISE, 3.0, Street.PREFLOP),
                PlayerAction("Bob", ActionType.CALL, 3.0, Street.PREFLOP),
                PlayerAction("Alice", ActionType.BET, 5.0, Street.FLOP),
            ]
        )
        preflop = hand.get_actions_for_street(Street.PREFLOP)
        assert len(preflop) == 2
        flop = hand.get_actions_for_street(Street.FLOP)
        assert len(flop) == 1


class TestHandHistoryParser:
    """Tests for hand history serialization."""

    def test_parse_card_str_valid(self) -> None:
        card = HandHistoryParser._parse_card_str("Ah")
        assert card is not None
        assert card.rank == Rank.ACE
        assert card.suit == Suit.HEARTS

    def test_parse_card_str_ten(self) -> None:
        card = HandHistoryParser._parse_card_str("10d")
        assert card is not None
        assert card.rank == Rank.TEN
        assert card.suit == Suit.DIAMONDS

    def test_parse_card_str_invalid(self) -> None:
        assert HandHistoryParser._parse_card_str("") is None
        assert HandHistoryParser._parse_card_str("X") is None
        assert HandHistoryParser._parse_card_str("Ax") is None
