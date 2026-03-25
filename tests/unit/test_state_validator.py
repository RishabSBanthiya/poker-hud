"""Unit tests for src.engine.state_validator module."""

from __future__ import annotations

from src.detection.card import Card, Rank, Suit
from src.engine.game_state import (
    ActionType,
    GameState,
    Player,
    PlayerAction,
    Street,
)
from src.engine.state_validator import (
    GameStateValidator,
    Severity,
    ValidationError,
)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

ACE_SPADES = Card(Rank.ACE, Suit.SPADES)
KING_HEARTS = Card(Rank.KING, Suit.HEARTS)
QUEEN_DIAMONDS = Card(Rank.QUEEN, Suit.DIAMONDS)
JACK_CLUBS = Card(Rank.JACK, Suit.CLUBS)
TEN_SPADES = Card(Rank.TEN, Suit.SPADES)
FIVE_HEARTS = Card(Rank.FIVE, Suit.HEARTS)


def _make_game(
    num_players: int = 3,
    sb: float = 1.0,
    bb: float = 2.0,
    stack: float = 100.0,
) -> GameState:
    """Build a GameState with seated players."""
    names = ["Hero", "Villain1", "Villain2", "Villain3", "Villain4", "Villain5"]
    players = [
        Player(name=names[i], seat_number=i, stack_size=stack)
        for i in range(num_players)
    ]
    return GameState(
        table_name="Test Table",
        small_blind=sb,
        big_blind=bb,
        players=players,
        hero_seat=0,
    )


# -----------------------------------------------------------------------
# Tests: ValidationError dataclass
# -----------------------------------------------------------------------


class TestValidationError:
    def test_str_without_context(self) -> None:
        err = ValidationError(
            severity=Severity.ERROR,
            message="Something went wrong",
        )
        assert "[ERROR]" in str(err)
        assert "Something went wrong" in str(err)

    def test_str_with_context(self) -> None:
        err = ValidationError(
            severity=Severity.WARNING,
            message="Minor issue",
            context={"pot": 10.0},
        )
        result = str(err)
        assert "[WARNING]" in result
        assert "pot=10.0" in result


# -----------------------------------------------------------------------
# Tests: pot consistency
# -----------------------------------------------------------------------


class TestPotConsistency:
    def test_valid_pot(self) -> None:
        gs = _make_game()
        gs.record_action(0, PlayerAction(ActionType.POST_BLIND, 1.0, Street.PREFLOP))
        gs.record_action(1, PlayerAction(ActionType.POST_BLIND, 2.0, Street.PREFLOP))

        validator = GameStateValidator(auto_correct=False)
        errors = validator.validate(gs)
        pot_errors = [e for e in errors if "Pot size" in e.message]
        assert len(pot_errors) == 0

    def test_pot_mismatch_error(self) -> None:
        gs = _make_game()
        gs.record_action(0, PlayerAction(ActionType.POST_BLIND, 1.0, Street.PREFLOP))
        gs.record_action(1, PlayerAction(ActionType.POST_BLIND, 2.0, Street.PREFLOP))
        # Manually corrupt the pot.
        gs.pot_size = 100.0

        validator = GameStateValidator(auto_correct=False)
        errors = validator.validate(gs)
        pot_errors = [e for e in errors if "Pot size" in e.message]
        assert len(pot_errors) == 1
        assert pot_errors[0].severity == Severity.ERROR

    def test_pot_auto_correct_small_discrepancy(self) -> None:
        gs = _make_game()
        gs.record_action(0, PlayerAction(ActionType.POST_BLIND, 1.0, Street.PREFLOP))
        gs.record_action(1, PlayerAction(ActionType.POST_BLIND, 2.0, Street.PREFLOP))
        # Small rounding error.
        gs.pot_size = 3.5

        validator = GameStateValidator(auto_correct=True)
        errors = validator.validate(gs)
        pot_errors = [e for e in errors if "auto-corrected" in e.message]
        assert len(pot_errors) == 1
        assert pot_errors[0].severity == Severity.WARNING
        # Should have been corrected.
        assert abs(gs.pot_size - 3.0) < 0.01

    def test_pot_auto_correct_disabled(self) -> None:
        gs = _make_game()
        gs.record_action(0, PlayerAction(ActionType.POST_BLIND, 1.0, Street.PREFLOP))
        gs.pot_size = 1.5  # Small mismatch

        validator = GameStateValidator(auto_correct=False)
        errors = validator.validate(gs)
        pot_errors = [e for e in errors if "Pot size" in e.message]
        assert len(pot_errors) == 1
        assert pot_errors[0].severity == Severity.ERROR
        # Not corrected.
        assert gs.pot_size == 1.5


# -----------------------------------------------------------------------
# Tests: stack sizes
# -----------------------------------------------------------------------


class TestStackSizes:
    def test_valid_stacks(self) -> None:
        gs = _make_game(stack=100.0)
        validator = GameStateValidator()
        errors = validator.validate(gs)
        stack_errors = [e for e in errors if "negative stack" in e.message]
        assert len(stack_errors) == 0

    def test_negative_stack(self) -> None:
        gs = _make_game()
        gs.players[0].stack_size = -10.0

        validator = GameStateValidator()
        errors = validator.validate(gs)
        stack_errors = [e for e in errors if "negative stack" in e.message]
        assert len(stack_errors) == 1
        assert stack_errors[0].severity == Severity.ERROR
        assert "Hero" in stack_errors[0].message

    def test_zero_stack_is_valid(self) -> None:
        gs = _make_game()
        gs.players[0].stack_size = 0.0

        validator = GameStateValidator()
        errors = validator.validate(gs)
        stack_errors = [e for e in errors if "negative stack" in e.message]
        assert len(stack_errors) == 0


# -----------------------------------------------------------------------
# Tests: active player count
# -----------------------------------------------------------------------


class TestActivePlayerCount:
    def test_consistent_active_count(self) -> None:
        gs = _make_game(num_players=3)
        gs.record_action(2, PlayerAction(ActionType.FOLD, 0.0, Street.PREFLOP))

        validator = GameStateValidator(auto_correct=False)
        errors = validator.validate(gs)
        active_errors = [e for e in errors if "Active player count" in e.message]
        assert len(active_errors) == 0

    def test_inconsistent_active_count(self) -> None:
        gs = _make_game(num_players=3)
        # Record a fold in history but don't actually deactivate.
        gs.action_history.append(
            PlayerAction(ActionType.FOLD, 0.0, Street.PREFLOP, player_name="Villain2")
        )
        # All players still active -> mismatch.

        validator = GameStateValidator(auto_correct=False)
        errors = validator.validate(gs)
        active_errors = [e for e in errors if "Active player count" in e.message]
        assert len(active_errors) == 1
        assert active_errors[0].severity == Severity.WARNING

    def test_no_active_players_is_error(self) -> None:
        gs = _make_game(num_players=2)
        # Fold both players (inconsistent state).
        gs.record_action(0, PlayerAction(ActionType.FOLD, 0.0, Street.PREFLOP))
        gs.record_action(1, PlayerAction(ActionType.FOLD, 0.0, Street.PREFLOP))

        validator = GameStateValidator(auto_correct=False)
        errors = validator.validate(gs)
        no_active = [e for e in errors if "No active players" in e.message]
        assert len(no_active) == 1
        assert no_active[0].severity == Severity.ERROR


# -----------------------------------------------------------------------
# Tests: betting rules
# -----------------------------------------------------------------------


class TestBettingRules:
    def test_valid_bet(self) -> None:
        gs = _make_game()
        gs.current_street = Street.FLOP
        gs.record_action(
            0, PlayerAction(ActionType.BET, 4.0, Street.FLOP)
        )

        validator = GameStateValidator(auto_correct=False)
        errors = validator.validate(gs)
        bet_errors = [
            e for e in errors
            if "bet" in e.message.lower() and "below" in e.message.lower()
        ]
        assert len(bet_errors) == 0

    def test_bet_below_big_blind(self) -> None:
        gs = _make_game(bb=2.0)
        gs.current_street = Street.FLOP
        gs.record_action(
            0, PlayerAction(ActionType.BET, 0.5, Street.FLOP)
        )

        validator = GameStateValidator(auto_correct=False)
        errors = validator.validate(gs)
        bet_errors = [e for e in errors if "below big blind" in e.message]
        assert len(bet_errors) == 1
        assert bet_errors[0].severity == Severity.WARNING

    def test_raise_below_min_raise(self) -> None:
        gs = _make_game(bb=2.0)
        # Bet 10, then raise only 5 (should be at least 10).
        gs.record_action(
            0, PlayerAction(ActionType.BET, 10.0, Street.FLOP)
        )
        gs.record_action(
            1, PlayerAction(ActionType.RAISE, 5.0, Street.FLOP)
        )

        validator = GameStateValidator(auto_correct=False)
        errors = validator.validate(gs)
        raise_errors = [e for e in errors if "min raise" in e.message]
        assert len(raise_errors) == 1


# -----------------------------------------------------------------------
# Tests: blind structure
# -----------------------------------------------------------------------


class TestBlindStructure:
    def test_valid_blinds(self) -> None:
        gs = _make_game(sb=1.0, bb=2.0)
        validator = GameStateValidator()
        errors = validator.validate(gs)
        blind_errors = [
            e for e in errors
            if "blind" in e.message.lower()
            and (
                "negative" in e.message.lower()
                or "greater" in e.message.lower()
            )
        ]
        assert len(blind_errors) == 0

    def test_negative_small_blind(self) -> None:
        gs = _make_game(sb=-1.0, bb=2.0)
        validator = GameStateValidator()
        errors = validator.validate(gs)
        blind_errors = [e for e in errors if "Small blind is negative" in e.message]
        assert len(blind_errors) == 1
        assert blind_errors[0].severity == Severity.ERROR

    def test_negative_big_blind(self) -> None:
        gs = _make_game(sb=1.0, bb=-2.0)
        validator = GameStateValidator()
        errors = validator.validate(gs)
        blind_errors = [e for e in errors if "Big blind is negative" in e.message]
        assert len(blind_errors) == 1

    def test_small_blind_exceeds_big_blind(self) -> None:
        gs = _make_game(sb=5.0, bb=2.0)
        validator = GameStateValidator()
        errors = validator.validate(gs)
        blind_errors = [e for e in errors if "greater than big blind" in e.message]
        assert len(blind_errors) == 1
        assert blind_errors[0].severity == Severity.ERROR


# -----------------------------------------------------------------------
# Tests: community card count
# -----------------------------------------------------------------------


class TestCommunityCardCount:
    def test_valid_flop(self) -> None:
        gs = _make_game()
        gs.current_street = Street.FLOP
        gs.community_cards = [ACE_SPADES, KING_HEARTS, QUEEN_DIAMONDS]

        validator = GameStateValidator()
        errors = validator.validate(gs)
        cc_errors = [e for e in errors if "community cards" in e.message]
        assert len(cc_errors) == 0

    def test_too_many_cards_for_street(self) -> None:
        gs = _make_game()
        gs.current_street = Street.FLOP
        gs.community_cards = [ACE_SPADES, KING_HEARTS, QUEEN_DIAMONDS, JACK_CLUBS]

        validator = GameStateValidator()
        errors = validator.validate(gs)
        cc_errors = [e for e in errors if "Too many community cards" in e.message]
        assert len(cc_errors) == 1
        assert cc_errors[0].severity == Severity.ERROR

    def test_preflop_no_cards(self) -> None:
        gs = _make_game()
        gs.current_street = Street.PREFLOP
        gs.community_cards = []

        validator = GameStateValidator()
        errors = validator.validate(gs)
        cc_errors = [e for e in errors if "community cards" in e.message]
        assert len(cc_errors) == 0


# -----------------------------------------------------------------------
# Tests: full validation pass
# -----------------------------------------------------------------------


class TestFullValidation:
    def test_clean_state_no_errors(self) -> None:
        gs = _make_game()
        validator = GameStateValidator()
        errors = validator.validate(gs)
        assert len(errors) == 0

    def test_multiple_errors_reported(self) -> None:
        gs = _make_game(sb=-1.0, bb=-2.0)
        gs.players[0].stack_size = -50.0

        validator = GameStateValidator(auto_correct=False)
        errors = validator.validate(gs)
        # Should have at least: negative SB, negative BB, negative stack.
        assert len(errors) >= 3
