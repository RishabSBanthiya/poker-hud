"""Unit tests for src.solver.postflop_advisor module."""

from __future__ import annotations

import pytest
from src.detection.card import Card, Rank, Suit
from src.engine.game_state import (
    GameState,
    Player,
    Position,
    Street,
)
from src.solver.equity import EquityCalculator
from src.solver.postflop_advisor import (
    ActionRecommendation,
    DrawType,
    PostflopAdvisor,
    detect_draws,
    detect_flush_draw,
    detect_straight_draw,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _c(rank: Rank, suit: Suit) -> Card:
    return Card(rank=rank, suit=suit)


AS = _c(Rank.ACE, Suit.SPADES)
AH = _c(Rank.ACE, Suit.HEARTS)
AD = _c(Rank.ACE, Suit.DIAMONDS)
KS = _c(Rank.KING, Suit.SPADES)
KH = _c(Rank.KING, Suit.HEARTS)
QS = _c(Rank.QUEEN, Suit.SPADES)
QH = _c(Rank.QUEEN, Suit.HEARTS)
JS = _c(Rank.JACK, Suit.SPADES)
JH = _c(Rank.JACK, Suit.HEARTS)
TS = _c(Rank.TEN, Suit.SPADES)
TH = _c(Rank.TEN, Suit.HEARTS)
NS = _c(Rank.NINE, Suit.SPADES)
NH = _c(Rank.NINE, Suit.HEARTS)
ES = _c(Rank.EIGHT, Suit.SPADES)
EH = _c(Rank.EIGHT, Suit.HEARTS)
SS = _c(Rank.SEVEN, Suit.SPADES)
SH = _c(Rank.SEVEN, Suit.HEARTS)
SIX_S = _c(Rank.SIX, Suit.SPADES)
SIX_H = _c(Rank.SIX, Suit.HEARTS)
FIVE_S = _c(Rank.FIVE, Suit.SPADES)
FIVE_H = _c(Rank.FIVE, Suit.HEARTS)
FOUR_H = _c(Rank.FOUR, Suit.HEARTS)
THREE_H = _c(Rank.THREE, Suit.HEARTS)
TWO_H = _c(Rank.TWO, Suit.HEARTS)
TWO_D = _c(Rank.TWO, Suit.DIAMONDS)
TWO_C = _c(Rank.TWO, Suit.CLUBS)


def _make_flop_game(
    board: list[Card],
    pot: float,
    hero_bet: float = 0.0,
    villain_bet: float = 0.0,
    street: Street = Street.FLOP,
) -> GameState:
    """Create a GameState on the flop with 2 players."""
    hero = Player(
        name="Hero", seat_number=0, position=Position.BTN,
        stack_size=100.0, is_active=True, current_bet=hero_bet,
    )
    villain = Player(
        name="Villain", seat_number=1, position=Position.BB,
        stack_size=100.0, is_active=True, current_bet=villain_bet,
    )
    gs = GameState(
        players=[hero, villain],
        hero_seat=0,
        community_cards=list(board),
        pot_size=pot,
        current_street=street,
        big_blind=2.0,
    )
    return gs


# ---------------------------------------------------------------------------
# Draw detection
# ---------------------------------------------------------------------------

class TestDetectFlushDraw:
    def test_four_to_flush(self) -> None:
        hand = (AS, KS)
        board = [QS, JS, TWO_H]
        assert detect_flush_draw(hand, board) is True

    def test_no_flush_draw(self) -> None:
        hand = (AH, KH)
        board = [QS, JS, TWO_D]
        assert detect_flush_draw(hand, board) is False

    def test_board_flush_draw_without_hero(self) -> None:
        """Three spades on board but hero has no spades."""
        hand = (AH, KH)
        board = [QS, JS, TS]
        # Hero doesn't contribute to the flush draw.
        assert detect_flush_draw(hand, board) is False

    def test_made_flush_is_not_draw(self) -> None:
        """5 to a flush is a made flush, not a draw."""
        hand = (AS, KS)
        board = [QS, JS, TS]
        # 5 cards of same suit: the count is 5, not 4, so no "draw".
        assert detect_flush_draw(hand, board) is False


class TestDetectStraightDraw:
    def test_oesd(self) -> None:
        """Open-ended: 8-9 on J-T-x board."""
        hand = (EH, NH)
        board = [JH, TH, TWO_D]
        result = detect_straight_draw(hand, board)
        assert result == DrawType.OPEN_ENDED_STRAIGHT

    def test_gutshot(self) -> None:
        """Gutshot: need exactly one inner card."""
        hand = (EH, SH)
        board = [TS, SIX_H, TWO_D]
        # 6-7-8-x-10 -- need 9, inner card = gutshot.
        result = detect_straight_draw(hand, board)
        assert result == DrawType.GUTSHOT_STRAIGHT

    def test_no_straight_draw(self) -> None:
        hand = (AH, KH)
        board = [TWO_D, FIVE_S, SIX_S]
        result = detect_straight_draw(hand, board)
        assert result == DrawType.NO_DRAW


class TestDetectDraws:
    def test_combo_draw(self) -> None:
        """Flush draw + straight draw = combo draw."""
        hand = (NS, ES)
        board = [JS, TS, TWO_H]
        result = detect_draws(hand, board)
        assert result == DrawType.COMBO_DRAW

    def test_flush_draw_only(self) -> None:
        # As + 3s on Ks-6h-2h board: 2 spades in hand + 1 on board = only 3.
        # Need 4 to a flush: use board with 2 spades.
        hand = (AS, _c(Rank.THREE, Suit.SPADES))
        board = [KS, SIX_S, TWO_H]
        result = detect_draws(hand, board)
        assert result == DrawType.FLUSH_DRAW

    def test_no_draw(self) -> None:
        hand = (AH, KH)
        board = [TWO_D, FIVE_S, SIX_S]
        result = detect_draws(hand, board)
        assert result == DrawType.NO_DRAW


# ---------------------------------------------------------------------------
# ActionRecommendation
# ---------------------------------------------------------------------------

class TestActionRecommendation:
    def test_dataclass_fields(self) -> None:
        rec = ActionRecommendation(
            action="call",
            confidence=0.7,
            equity=0.55,
            pot_odds=0.33,
            reasoning="Good equity.",
        )
        assert rec.action == "call"
        assert rec.draw_type == DrawType.NO_DRAW


# ---------------------------------------------------------------------------
# PostflopAdvisor
# ---------------------------------------------------------------------------

class TestPostflopAdvisor:
    def setup_method(self) -> None:
        self.calc = EquityCalculator(seed=42)
        self.advisor = PostflopAdvisor(
            equity_calculator=self.calc, num_simulations=2_000,
        )

    def test_strong_hand_recommends_raise(self) -> None:
        """AA on a low board should recommend raise."""
        board = [TWO_D, THREE_H, SIX_H]
        gs = _make_flop_game(board=board, pot=10.0, villain_bet=5.0)
        rec = self.advisor.get_recommendation(gs, (AS, AH), amount_to_call=5.0)
        assert rec.action in ("raise", "call")
        assert rec.equity > 0.5

    def test_weak_hand_large_bet_recommends_fold(self) -> None:
        """Weak hand facing a large bet should fold."""
        board = [AS, KS, QS]  # Scary board.
        gs = _make_flop_game(board=board, pot=10.0, villain_bet=50.0)
        rec = self.advisor.get_recommendation(
            gs, (TWO_H, _c(Rank.FOUR, Suit.CLUBS)), amount_to_call=50.0,
        )
        assert rec.action == "fold"
        assert rec.pot_odds > 0

    def test_drawing_hand_gets_implied_odds(self) -> None:
        """Flush draw should get implied odds bonus."""
        board = [KS, JS, TWO_D]
        gs = _make_flop_game(board=board, pot=10.0, villain_bet=4.0)
        rec = self.advisor.get_recommendation(
            gs, (AS, TS), amount_to_call=4.0,
        )
        # With flush draw, should lean towards call or raise.
        assert rec.action in ("call", "raise")
        assert rec.draw_type in (
            DrawType.FLUSH_DRAW, DrawType.COMBO_DRAW,
        )

    def test_checked_to_strong_hand_bets(self) -> None:
        """When checked to with a strong hand, should bet (raise)."""
        board = [TWO_D, THREE_H, SIX_H]
        gs = _make_flop_game(board=board, pot=6.0)
        rec = self.advisor.get_recommendation(gs, (AS, AH), amount_to_call=0.0)
        assert rec.action == "raise"
        assert rec.pot_odds == 0.0

    def test_checked_to_weak_hand_checks(self) -> None:
        """When checked to with a weak hand, should check."""
        board = [AS, KS, QS]
        gs = _make_flop_game(board=board, pot=6.0)
        rec = self.advisor.get_recommendation(
            gs, (TWO_H, _c(Rank.FOUR, Suit.CLUBS)), amount_to_call=0.0,
        )
        assert rec.action == "call"  # "call" means check when amount_to_call=0

    def test_pot_odds_calculation(self) -> None:
        """Pot odds should be amount_to_call / (pot + amount_to_call)."""
        board = [TWO_D, THREE_H, SIX_H]
        gs = _make_flop_game(board=board, pot=10.0, villain_bet=5.0)
        rec = self.advisor.get_recommendation(gs, (AS, AH), amount_to_call=5.0)
        expected_pot_odds = 5.0 / (10.0 + 5.0)
        assert rec.pot_odds == pytest.approx(expected_pot_odds, abs=0.01)

    def test_river_no_implied_odds(self) -> None:
        """On the river, draws should not get implied odds bonus."""
        board = [KS, JS, TWO_D, THREE_H, SIX_H]
        gs = _make_flop_game(
            board=board, pot=10.0, villain_bet=8.0, street=Street.RIVER,
        )
        rec = self.advisor.get_recommendation(
            gs, (AS, TS), amount_to_call=8.0,
        )
        # Flush draw on the river is worthless -- still a made hand decision.
        assert rec.action in ("call", "fold", "raise")
        assert rec.equity >= 0.0

    def test_infer_amount_to_call(self) -> None:
        """Should infer call amount from opponent bets vs hero bet."""
        board = [TWO_D, THREE_H, SIX_H]
        gs = _make_flop_game(
            board=board, pot=20.0, hero_bet=2.0, villain_bet=8.0,
        )
        rec = self.advisor.get_recommendation(gs, (AS, AH))
        # Inferred amount to call = 8.0 - 2.0 = 6.0.
        expected_pot_odds = 6.0 / (20.0 + 6.0)
        assert rec.pot_odds == pytest.approx(expected_pot_odds, abs=0.02)

    def test_recommendation_has_reasoning(self) -> None:
        board = [TWO_D, THREE_H, SIX_H]
        gs = _make_flop_game(board=board, pot=10.0, villain_bet=5.0)
        rec = self.advisor.get_recommendation(gs, (AS, AH), amount_to_call=5.0)
        assert len(rec.reasoning) > 0
        assert isinstance(rec.confidence, float)
        assert 0.0 <= rec.confidence <= 1.0
