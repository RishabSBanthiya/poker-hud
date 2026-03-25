"""Unit tests for src.engine.game_state module."""

from __future__ import annotations

import pytest
from src.detection.card import Card, Rank, Suit
from src.engine.game_state import (
    _STREET_NEXT,
    POSITION_ORDER_FULL,
    STREET_COMMUNITY_CARDS,
    ActionType,
    GameState,
    HandStrength,
    Player,
    PlayerAction,
    Position,
    SidePot,
    Street,
)

# -----------------------------------------------------------------------
# Helpers / fixtures
# -----------------------------------------------------------------------

def _card(rank: Rank, suit: Suit) -> Card:
    return Card(rank=rank, suit=suit)


ACE_SPADES = _card(Rank.ACE, Suit.SPADES)
KING_HEARTS = _card(Rank.KING, Suit.HEARTS)
QUEEN_DIAMONDS = _card(Rank.QUEEN, Suit.DIAMONDS)
JACK_CLUBS = _card(Rank.JACK, Suit.CLUBS)
TEN_SPADES = _card(Rank.TEN, Suit.SPADES)
FIVE_HEARTS = _card(Rank.FIVE, Suit.HEARTS)


def _make_player(name: str = "Alice", seat: int = 0, **kwargs) -> Player:
    return Player(name=name, seat_number=seat, **kwargs)


def _make_game(
    num_players: int = 3,
    hero_seat: int = 0,
    sb: float = 1.0,
    bb: float = 2.0,
) -> GameState:
    """Build a GameState with ``num_players`` seated players."""
    names = ["Hero", "Villain1", "Villain2", "Villain3", "Villain4",
             "Villain5", "Villain6", "Villain7", "Villain8", "Villain9"]
    players = [
        Player(name=names[i], seat_number=i, stack_size=100.0)
        for i in range(num_players)
    ]
    return GameState(
        table_name="Test Table",
        small_blind=sb,
        big_blind=bb,
        players=players,
        hero_seat=hero_seat,
    )


# -----------------------------------------------------------------------
# Enum tests
# -----------------------------------------------------------------------

class TestPosition:
    def test_all_positions_exist(self) -> None:
        expected = {"BTN", "SB", "BB", "UTG", "UTG1", "MP", "MP1", "CO", "HJ", "LJ"}
        assert {p.value for p in Position} == expected

    def test_position_order_length(self) -> None:
        assert len(POSITION_ORDER_FULL) == 10


class TestStreet:
    def test_street_order(self) -> None:
        streets = list(Street)
        names = [s.name for s in streets]
        assert names == ["PREFLOP", "FLOP", "TURN", "RIVER", "SHOWDOWN"]

    def test_community_card_counts(self) -> None:
        assert STREET_COMMUNITY_CARDS[Street.PREFLOP] == 0
        assert STREET_COMMUNITY_CARDS[Street.FLOP] == 3
        assert STREET_COMMUNITY_CARDS[Street.TURN] == 4
        assert STREET_COMMUNITY_CARDS[Street.RIVER] == 5
        assert STREET_COMMUNITY_CARDS[Street.SHOWDOWN] == 5

    def test_street_transitions(self) -> None:
        assert _STREET_NEXT[Street.PREFLOP] == Street.FLOP
        assert _STREET_NEXT[Street.FLOP] == Street.TURN
        assert _STREET_NEXT[Street.TURN] == Street.RIVER
        assert _STREET_NEXT[Street.RIVER] == Street.SHOWDOWN
        assert Street.SHOWDOWN not in _STREET_NEXT


class TestActionType:
    def test_all_action_types(self) -> None:
        expected = {"fold", "check", "call", "bet", "raise", "all_in", "post_blind"}
        assert {a.value for a in ActionType} == expected


class TestHandStrength:
    def test_ordering(self) -> None:
        # auto() assigns increasing ints so we can compare .value
        assert HandStrength.HIGH_CARD.value < HandStrength.PAIR.value
        assert HandStrength.PAIR.value < HandStrength.TWO_PAIR.value
        assert HandStrength.STRAIGHT_FLUSH.value < HandStrength.ROYAL_FLUSH.value


# -----------------------------------------------------------------------
# PlayerAction tests
# -----------------------------------------------------------------------

class TestPlayerAction:
    def test_defaults(self) -> None:
        action = PlayerAction(
            action_type=ActionType.FOLD,
            amount=0.0,
            street=Street.PREFLOP,
        )
        assert action.player_name == ""
        assert action.timestamp > 0

    def test_str_with_amount(self) -> None:
        action = PlayerAction(
            action_type=ActionType.BET,
            amount=10.0,
            street=Street.FLOP,
            player_name="Bob",
        )
        assert "Bob" in str(action)
        assert "bet" in str(action)
        assert "$10.00" in str(action)

    def test_str_without_amount(self) -> None:
        action = PlayerAction(
            action_type=ActionType.CHECK,
            amount=0.0,
            street=Street.FLOP,
            player_name="Bob",
        )
        result = str(action)
        assert "check" in result
        assert "$" not in result


# -----------------------------------------------------------------------
# Player tests
# -----------------------------------------------------------------------

class TestPlayer:
    def test_basic_creation(self) -> None:
        player = _make_player("Alice", seat=3, stack_size=500.0)
        assert player.name == "Alice"
        assert player.seat_number == 3
        assert player.stack_size == 500.0
        assert player.is_active is True
        assert player.is_all_in is False
        assert player.hole_cards is None
        assert player.actions == []
        assert player.current_bet == 0.0

    def test_add_action_fold(self) -> None:
        player = _make_player()
        action = PlayerAction(ActionType.FOLD, 0.0, Street.PREFLOP)
        player.add_action(action)
        assert not player.is_active
        assert len(player.actions) == 1

    def test_add_action_bet(self) -> None:
        player = _make_player(stack_size=100.0)
        action = PlayerAction(ActionType.BET, 20.0, Street.FLOP)
        player.add_action(action)
        assert player.current_bet == 20.0
        assert player.stack_size == 80.0
        assert player.is_active

    def test_add_action_raise(self) -> None:
        player = _make_player(stack_size=100.0)
        player.add_action(PlayerAction(ActionType.CALL, 10.0, Street.PREFLOP))
        player.add_action(PlayerAction(ActionType.RAISE, 30.0, Street.PREFLOP))
        assert player.current_bet == 40.0
        assert player.stack_size == 60.0

    def test_add_action_all_in(self) -> None:
        player = _make_player(stack_size=50.0)
        action = PlayerAction(ActionType.ALL_IN, 50.0, Street.PREFLOP)
        player.add_action(action)
        assert player.is_all_in
        assert player.stack_size == 0.0
        assert player.current_bet == 50.0

    def test_add_action_post_blind(self) -> None:
        player = _make_player(stack_size=100.0)
        action = PlayerAction(ActionType.POST_BLIND, 2.0, Street.PREFLOP)
        player.add_action(action)
        assert player.current_bet == 2.0
        assert player.stack_size == 98.0

    def test_reset_for_new_street(self) -> None:
        player = _make_player(stack_size=100.0)
        player.add_action(PlayerAction(ActionType.BET, 10.0, Street.FLOP))
        assert player.current_bet == 10.0
        player.reset_for_new_street()
        assert player.current_bet == 0.0

    def test_str(self) -> None:
        player = _make_player("Bob", seat=2, stack_size=75.0, position=Position.BTN)
        result = str(player)
        assert "Bob" in result
        assert "BTN" in result
        assert "75.00" in result

    def test_str_no_position(self) -> None:
        player = _make_player("Bob", seat=2, stack_size=75.0)
        assert "?" in str(player)

    def test_hole_cards(self) -> None:
        player = _make_player()
        player.hole_cards = [ACE_SPADES, KING_HEARTS]
        assert len(player.hole_cards) == 2
        assert player.hole_cards[0] == ACE_SPADES


# -----------------------------------------------------------------------
# SidePot tests
# -----------------------------------------------------------------------

class TestSidePot:
    def test_creation(self) -> None:
        sp = SidePot(amount=150.0, eligible_players=[0, 2])
        assert sp.amount == 150.0
        assert sp.eligible_players == [0, 2]

    def test_default_eligible(self) -> None:
        sp = SidePot(amount=50.0)
        assert sp.eligible_players == []


# -----------------------------------------------------------------------
# GameState tests
# -----------------------------------------------------------------------

class TestGameStateCreation:
    def test_defaults(self) -> None:
        gs = GameState()
        assert gs.current_street == Street.PREFLOP
        assert gs.pot_size == 0.0
        assert gs.community_cards == []
        assert gs.players == []
        assert gs.action_history == []
        assert gs.side_pots == []
        assert gs.ante == 0.0
        assert gs.hand_number == 0
        assert len(gs.hand_id) > 0
        assert len(gs.table_id) > 0

    def test_custom_values(self) -> None:
        gs = _make_game(num_players=6, sb=5.0, bb=10.0)
        assert gs.small_blind == 5.0
        assert gs.big_blind == 10.0
        assert len(gs.players) == 6


class TestGameStateQueries:
    def test_get_hero(self) -> None:
        gs = _make_game(hero_seat=0)
        hero = gs.get_hero()
        assert hero is not None
        assert hero.name == "Hero"

    def test_get_hero_not_found(self) -> None:
        gs = _make_game(hero_seat=99)
        assert gs.get_hero() is None

    def test_get_player_by_seat(self) -> None:
        gs = _make_game()
        p = gs.get_player_by_seat(1)
        assert p is not None
        assert p.name == "Villain1"

    def test_get_player_by_seat_missing(self) -> None:
        gs = _make_game()
        assert gs.get_player_by_seat(99) is None

    def test_get_player_by_name(self) -> None:
        gs = _make_game()
        p = gs.get_player_by_name("Villain2")
        assert p is not None
        assert p.seat_number == 2

    def test_get_player_by_name_missing(self) -> None:
        gs = _make_game()
        assert gs.get_player_by_name("Ghost") is None

    def test_get_active_players(self) -> None:
        gs = _make_game()
        assert len(gs.get_active_players()) == 3

        # Fold one player
        gs.players[1].is_active = False
        active = gs.get_active_players()
        assert len(active) == 2
        assert all(p.is_active for p in active)

    def test_get_current_pot_no_side_pots(self) -> None:
        gs = _make_game()
        gs.pot_size = 50.0
        assert gs.get_current_pot() == 50.0

    def test_get_current_pot_with_side_pots(self) -> None:
        gs = _make_game()
        gs.pot_size = 50.0
        gs.side_pots = [SidePot(30.0, [0, 1]), SidePot(20.0, [0])]
        assert gs.get_current_pot() == 100.0

    def test_get_num_players(self) -> None:
        gs = _make_game(num_players=5)
        assert gs.get_num_players() == 5


class TestGameStateMutations:
    def test_advance_street_preflop_to_flop(self) -> None:
        gs = _make_game()
        gs.players[0].current_bet = 10.0
        gs.advance_street()
        assert gs.current_street == Street.FLOP
        # Bets should be reset
        assert gs.players[0].current_bet == 0.0

    def test_advance_street_full_sequence(self) -> None:
        gs = _make_game()
        gs.advance_street()
        assert gs.current_street == Street.FLOP
        gs.advance_street()
        assert gs.current_street == Street.TURN
        gs.advance_street()
        assert gs.current_street == Street.RIVER
        gs.advance_street()
        assert gs.current_street == Street.SHOWDOWN

    def test_advance_street_past_showdown_raises(self) -> None:
        gs = _make_game()
        gs.current_street = Street.SHOWDOWN
        with pytest.raises(ValueError, match="Cannot advance past SHOWDOWN"):
            gs.advance_street()

    def test_add_community_cards_flop(self) -> None:
        gs = _make_game()
        gs.current_street = Street.FLOP
        flop = [ACE_SPADES, KING_HEARTS, QUEEN_DIAMONDS]
        gs.add_community_cards(flop)
        assert gs.community_cards == flop

    def test_add_community_cards_turn(self) -> None:
        gs = _make_game()
        gs.current_street = Street.TURN
        gs.community_cards = [ACE_SPADES, KING_HEARTS, QUEEN_DIAMONDS]
        gs.add_community_cards([JACK_CLUBS])
        assert len(gs.community_cards) == 4

    def test_add_community_cards_exceeds_five(self) -> None:
        gs = _make_game()
        gs.current_street = Street.RIVER
        gs.community_cards = [
            ACE_SPADES, KING_HEARTS, QUEEN_DIAMONDS,
            JACK_CLUBS, TEN_SPADES,
        ]
        with pytest.raises(ValueError, match="max 5"):
            gs.add_community_cards([FIVE_HEARTS])

    def test_add_community_cards_wrong_count_for_street(self) -> None:
        gs = _make_game()
        gs.current_street = Street.PREFLOP
        with pytest.raises(ValueError, match="PREFLOP expects at most 0"):
            gs.add_community_cards([ACE_SPADES])

    def test_add_community_cards_empty_list(self) -> None:
        gs = _make_game()
        gs.add_community_cards([])
        assert gs.community_cards == []

    def test_record_action(self) -> None:
        gs = _make_game()
        action = PlayerAction(ActionType.BET, 10.0, Street.PREFLOP)
        gs.record_action(0, action)

        assert len(gs.action_history) == 1
        assert gs.action_history[0].player_name == "Hero"
        assert gs.pot_size == 10.0
        assert gs.players[0].stack_size == 90.0

    def test_record_action_fold(self) -> None:
        gs = _make_game()
        action = PlayerAction(ActionType.FOLD, 0.0, Street.PREFLOP)
        gs.record_action(1, action)
        # Fold adds nothing to pot
        assert gs.pot_size == 0.0
        assert not gs.players[1].is_active

    def test_record_action_check(self) -> None:
        gs = _make_game()
        action = PlayerAction(ActionType.CHECK, 0.0, Street.FLOP)
        gs.record_action(0, action)
        assert gs.pot_size == 0.0

    def test_record_action_invalid_seat(self) -> None:
        gs = _make_game()
        action = PlayerAction(ActionType.BET, 10.0, Street.PREFLOP)
        with pytest.raises(ValueError, match="No player at seat 99"):
            gs.record_action(99, action)

    def test_record_action_post_blind(self) -> None:
        gs = _make_game(sb=1.0, bb=2.0)
        gs.record_action(0, PlayerAction(ActionType.POST_BLIND, 1.0, Street.PREFLOP))
        gs.record_action(1, PlayerAction(ActionType.POST_BLIND, 2.0, Street.PREFLOP))
        assert gs.pot_size == 3.0

    def test_reset_for_new_hand(self) -> None:
        gs = _make_game()
        old_hand_id = gs.hand_id

        # Simulate some play
        gs.pot_size = 100.0
        gs.current_street = Street.RIVER
        gs.community_cards = [
            ACE_SPADES, KING_HEARTS, QUEEN_DIAMONDS,
            JACK_CLUBS, TEN_SPADES,
        ]
        gs.players[0].hole_cards = [ACE_SPADES, KING_HEARTS]
        gs.players[1].is_active = False
        gs.players[2].is_all_in = True
        gs.action_history.append(
            PlayerAction(ActionType.BET, 50.0, Street.RIVER, player_name="Hero")
        )

        gs.reset_for_new_hand()

        # Check reset state
        assert gs.hand_id != old_hand_id
        assert gs.hand_number == 1
        assert gs.current_street == Street.PREFLOP
        assert gs.community_cards == []
        assert gs.pot_size == 0.0
        assert gs.side_pots == []
        assert gs.action_history == []

        # Players should be reset
        for player in gs.players:
            assert player.hole_cards is None
            assert player.is_active is True
            assert player.is_all_in is False
            assert player.current_bet == 0.0
            assert player.actions == []

        # Stacks should be preserved (not reset)
        assert gs.players[0].stack_size == 100.0
        assert gs.table_name == "Test Table"


class TestGameStateStr:
    def test_str_empty_board(self) -> None:
        gs = _make_game()
        result = str(gs)
        assert "PREFLOP" in result
        assert "---" in result  # No board cards
        assert "$0.00" in result

    def test_str_with_board(self) -> None:
        gs = _make_game()
        gs.current_street = Street.FLOP
        gs.community_cards = [ACE_SPADES, KING_HEARTS, QUEEN_DIAMONDS]
        gs.pot_size = 25.0
        result = str(gs)
        assert "FLOP" in result
        assert "$25.00" in result


# -----------------------------------------------------------------------
# Integration-style tests (multi-step scenarios)
# -----------------------------------------------------------------------

class TestHandScenario:
    """Walk through a realistic hand to verify state transitions."""

    def test_full_preflop_to_showdown(self) -> None:
        gs = _make_game(num_players=3, sb=1.0, bb=2.0)

        # Post blinds
        gs.record_action(0, PlayerAction(ActionType.POST_BLIND, 1.0, Street.PREFLOP))
        gs.record_action(1, PlayerAction(ActionType.POST_BLIND, 2.0, Street.PREFLOP))
        assert gs.pot_size == 3.0

        # Preflop action
        gs.record_action(2, PlayerAction(ActionType.CALL, 2.0, Street.PREFLOP))
        gs.record_action(0, PlayerAction(ActionType.CALL, 1.0, Street.PREFLOP))
        gs.record_action(1, PlayerAction(ActionType.CHECK, 0.0, Street.PREFLOP))
        assert gs.pot_size == 6.0

        # Advance to flop
        gs.advance_street()
        assert gs.current_street == Street.FLOP
        gs.add_community_cards([ACE_SPADES, KING_HEARTS, QUEEN_DIAMONDS])

        # Flop action
        gs.record_action(0, PlayerAction(ActionType.BET, 4.0, Street.FLOP))
        gs.record_action(1, PlayerAction(ActionType.CALL, 4.0, Street.FLOP))
        gs.record_action(2, PlayerAction(ActionType.FOLD, 0.0, Street.FLOP))
        assert gs.pot_size == 14.0
        assert len(gs.get_active_players()) == 2

        # Advance to turn
        gs.advance_street()
        assert gs.current_street == Street.TURN
        gs.add_community_cards([JACK_CLUBS])
        assert len(gs.community_cards) == 4

        # Turn action
        gs.record_action(0, PlayerAction(ActionType.CHECK, 0.0, Street.TURN))
        gs.record_action(1, PlayerAction(ActionType.CHECK, 0.0, Street.TURN))

        # Advance to river
        gs.advance_street()
        assert gs.current_street == Street.RIVER
        gs.add_community_cards([TEN_SPADES])
        assert len(gs.community_cards) == 5

        # River action
        gs.record_action(0, PlayerAction(ActionType.BET, 10.0, Street.RIVER))
        gs.record_action(1, PlayerAction(ActionType.CALL, 10.0, Street.RIVER))
        assert gs.pot_size == 34.0

        # Advance to showdown
        gs.advance_street()
        assert gs.current_street == Street.SHOWDOWN

        # Verify action history has all actions:
        # 2 blinds + 3 preflop + 3 flop + 2 turn + 2 river = 12
        assert len(gs.action_history) == 12

    def test_heads_up_all_in(self) -> None:
        """Two players, one goes all-in preflop."""
        gs = _make_game(num_players=2, sb=1.0, bb=2.0)

        gs.record_action(0, PlayerAction(ActionType.POST_BLIND, 1.0, Street.PREFLOP))
        gs.record_action(1, PlayerAction(ActionType.POST_BLIND, 2.0, Street.PREFLOP))
        gs.record_action(0, PlayerAction(ActionType.ALL_IN, 99.0, Street.PREFLOP))
        gs.record_action(1, PlayerAction(ActionType.CALL, 98.0, Street.PREFLOP))

        assert gs.players[0].is_all_in
        assert gs.players[0].stack_size == 0.0
        assert gs.pot_size == 200.0

    def test_new_hand_after_reset(self) -> None:
        gs = _make_game()
        gs.record_action(0, PlayerAction(ActionType.POST_BLIND, 1.0, Street.PREFLOP))
        gs.reset_for_new_hand()

        # Should be able to play again
        gs.record_action(0, PlayerAction(ActionType.POST_BLIND, 1.0, Street.PREFLOP))
        assert gs.pot_size == 1.0
        assert gs.hand_number == 1
