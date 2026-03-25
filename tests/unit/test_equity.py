"""Unit tests for src.solver.equity module."""

from __future__ import annotations

import time

import pytest
from src.detection.card import Card, Rank, Suit
from src.engine.game_state import HandStrength
from src.solver.equity import (
    EquityCalculator,
    EquityResult,
    compare_hands,
    evaluate_hand,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _c(rank: Rank, suit: Suit) -> Card:
    return Card(rank=rank, suit=suit)


AS = _c(Rank.ACE, Suit.SPADES)
AH = _c(Rank.ACE, Suit.HEARTS)
AD = _c(Rank.ACE, Suit.DIAMONDS)
AC = _c(Rank.ACE, Suit.CLUBS)
KS = _c(Rank.KING, Suit.SPADES)
KH = _c(Rank.KING, Suit.HEARTS)
KD = _c(Rank.KING, Suit.DIAMONDS)
KC = _c(Rank.KING, Suit.CLUBS)
QS = _c(Rank.QUEEN, Suit.SPADES)
QH = _c(Rank.QUEEN, Suit.HEARTS)
JS = _c(Rank.JACK, Suit.SPADES)
JH = _c(Rank.JACK, Suit.HEARTS)
TS = _c(Rank.TEN, Suit.SPADES)
TH = _c(Rank.TEN, Suit.HEARTS)
NS = _c(Rank.NINE, Suit.SPADES)
ES = _c(Rank.EIGHT, Suit.SPADES)
EH = _c(Rank.EIGHT, Suit.HEARTS)
SS = _c(Rank.SEVEN, Suit.SPADES)
SH = _c(Rank.SEVEN, Suit.HEARTS)
SIX_S = _c(Rank.SIX, Suit.SPADES)
FIVE_S = _c(Rank.FIVE, Suit.SPADES)
FIVE_H = _c(Rank.FIVE, Suit.HEARTS)
FOUR_S = _c(Rank.FOUR, Suit.SPADES)
THREE_S = _c(Rank.THREE, Suit.SPADES)
THREE_H = _c(Rank.THREE, Suit.HEARTS)
TWO_S = _c(Rank.TWO, Suit.SPADES)
TWO_H = _c(Rank.TWO, Suit.HEARTS)
TWO_D = _c(Rank.TWO, Suit.DIAMONDS)


# ---------------------------------------------------------------------------
# evaluate_hand — 5 cards
# ---------------------------------------------------------------------------

class TestEvaluateHand5Cards:
    def test_royal_flush(self) -> None:
        cards = [AS, KS, QS, JS, TS]
        strength, kickers = evaluate_hand(cards)
        assert strength == HandStrength.ROYAL_FLUSH

    def test_straight_flush(self) -> None:
        cards = [NS, ES, SS, SIX_S, FIVE_S]
        strength, kickers = evaluate_hand(cards)
        assert strength == HandStrength.STRAIGHT_FLUSH
        assert kickers[0] == 7  # 9 high

    def test_four_of_a_kind(self) -> None:
        cards = [AS, AH, AD, AC, KS]
        strength, kickers = evaluate_hand(cards)
        assert strength == HandStrength.FOUR_OF_A_KIND
        assert kickers[0] == 12  # Aces

    def test_full_house(self) -> None:
        cards = [AS, AH, AD, KS, KH]
        strength, kickers = evaluate_hand(cards)
        assert strength == HandStrength.FULL_HOUSE
        assert kickers == [12, 11]

    def test_flush(self) -> None:
        cards = [AS, KS, QS, JS, NS]
        strength, kickers = evaluate_hand(cards)
        assert strength == HandStrength.FLUSH

    def test_straight(self) -> None:
        cards = [AS, KH, QS, JH, TS]
        strength, kickers = evaluate_hand(cards)
        assert strength == HandStrength.STRAIGHT
        assert kickers[0] == 12  # Ace high

    def test_wheel_straight(self) -> None:
        """A-2-3-4-5 straight (wheel)."""
        cards = [AS, TWO_H, THREE_S, FOUR_S, FIVE_H]
        strength, kickers = evaluate_hand(cards)
        assert strength == HandStrength.STRAIGHT
        assert kickers[0] == 3  # 5 high

    def test_three_of_a_kind(self) -> None:
        cards = [AS, AH, AD, KS, QH]
        strength, kickers = evaluate_hand(cards)
        assert strength == HandStrength.THREE_OF_A_KIND

    def test_two_pair(self) -> None:
        cards = [AS, AH, KS, KH, QS]
        strength, kickers = evaluate_hand(cards)
        assert strength == HandStrength.TWO_PAIR
        assert kickers[0] == 12  # Aces
        assert kickers[1] == 11  # Kings

    def test_one_pair(self) -> None:
        cards = [AS, AH, KS, QH, JS]
        strength, kickers = evaluate_hand(cards)
        assert strength == HandStrength.PAIR
        assert kickers[0] == 12

    def test_high_card(self) -> None:
        cards = [AS, KH, QS, JH, NS]
        strength, kickers = evaluate_hand(cards)
        assert strength == HandStrength.HIGH_CARD
        assert kickers[0] == 12


# ---------------------------------------------------------------------------
# evaluate_hand — 7 cards (best 5)
# ---------------------------------------------------------------------------

class TestEvaluateHand7Cards:
    def test_finds_best_from_7(self) -> None:
        """Should find a flush among 7 cards."""
        cards = [AS, KS, QS, JS, NS, TWO_H, THREE_H]
        strength, _ = evaluate_hand(cards)
        assert strength == HandStrength.FLUSH

    def test_full_house_from_7(self) -> None:
        cards = [AS, AH, AD, KS, KH, TWO_S, THREE_S]
        strength, kickers = evaluate_hand(cards)
        assert strength == HandStrength.FULL_HOUSE

    def test_too_few_cards_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 5"):
            evaluate_hand([AS, AH, KS])


# ---------------------------------------------------------------------------
# compare_hands
# ---------------------------------------------------------------------------

class TestCompareHands:
    def test_higher_hand_wins(self) -> None:
        board = [QS, JH, TS, TWO_H, THREE_H]
        # AA vs KK
        result = compare_hands([AS, AH], [KS, KH], board)
        assert result == 1

    def test_lower_hand_loses(self) -> None:
        board = [QS, JH, TS, TWO_H, THREE_H]
        result = compare_hands([KS, KH], [AS, AH], board)
        assert result == -1

    def test_tie(self) -> None:
        # Same board, both have same pair (board pair).
        board = [AS, KS, QS, JS, TS]
        four_h = _c(Rank.FOUR, Suit.HEARTS)
        result = compare_hands([TWO_H, THREE_H], [TWO_D, four_h], board)
        assert result == 0  # Both play the board


# ---------------------------------------------------------------------------
# EquityCalculator
# ---------------------------------------------------------------------------

class TestEquityCalculator:
    def setup_method(self) -> None:
        self.calc = EquityCalculator(seed=42)

    def test_aa_vs_kk_approximately_80_pct(self) -> None:
        """AA vs KK preflop should be roughly 80% equity."""
        equity = self.calc.calculate_equity(
            hand=(AS, AH),
            board=[],
            num_opponents=1,
            num_simulations=10_000,
        )
        # Should be between 75% and 90%.
        assert 0.75 <= equity <= 0.90, f"AA vs random opponent equity: {equity:.2%}"

    def test_aa_preflop_high_equity(self) -> None:
        """AA preflop vs 1 random opponent should be >75%."""
        equity = self.calc.calculate_equity(
            hand=(AS, AH), board=[], num_opponents=1, num_simulations=5_000,
        )
        assert equity > 0.75

    def test_72o_preflop_low_equity(self) -> None:
        """72o preflop should have low equity."""
        equity = self.calc.calculate_equity(
            hand=(SH, TWO_H), board=[], num_opponents=1, num_simulations=5_000,
        )
        assert equity < 0.45

    def test_equity_with_board(self) -> None:
        """After a favorable flop, equity should increase."""
        # AA on a low board.
        equity = self.calc.calculate_equity(
            hand=(AS, AH),
            board=[TWO_S, THREE_S, SIX_S],
            num_opponents=1,
            num_simulations=5_000,
        )
        assert equity > 0.6

    def test_equity_multiple_opponents(self) -> None:
        """Equity decreases with more opponents."""
        eq_1 = self.calc.calculate_equity(
            hand=(AS, AH), board=[], num_opponents=1, num_simulations=5_000,
        )
        eq_3 = self.calc.calculate_equity(
            hand=(AS, AH), board=[], num_opponents=3, num_simulations=5_000,
        )
        assert eq_3 < eq_1

    def test_detailed_result_structure(self) -> None:
        result = self.calc.calculate_equity_detailed(
            hand=(AS, AH), board=[], num_opponents=1, num_simulations=1_000,
        )
        assert isinstance(result, EquityResult)
        assert result.total_simulations == 1_000
        assert result.win_count + result.tie_count <= 1_000
        assert 0.0 <= result.equity <= 1.0

    def test_board_too_many_cards_raises(self) -> None:
        with pytest.raises(ValueError, match="more than 5"):
            self.calc.calculate_equity(
                hand=(AS, AH),
                board=[KS, QS, JS, TS, NS, ES],
                num_opponents=1,
            )

    def test_no_opponents_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            self.calc.calculate_equity(
                hand=(AS, AH), board=[], num_opponents=0,
            )

    def test_river_equity_deterministic(self) -> None:
        """With all 5 board cards dealt, equity is near 0 or 1."""
        # AA on board with no help for opponents.
        board = [
            _c(Rank.TWO, Suit.DIAMONDS),
            _c(Rank.THREE, Suit.CLUBS),
            _c(Rank.SIX, Suit.HEARTS),
            _c(Rank.NINE, Suit.DIAMONDS),
            _c(Rank.JACK, Suit.CLUBS),
        ]
        equity = self.calc.calculate_equity(
            hand=(AS, AH), board=board, num_opponents=1, num_simulations=5_000,
        )
        assert equity > 0.7

    def test_performance_10k_simulations(self) -> None:
        """10,000 simulations should complete in under 2 seconds."""
        start = time.time()
        self.calc.calculate_equity(
            hand=(AS, AH), board=[], num_opponents=1, num_simulations=10_000,
        )
        elapsed = time.time() - start
        assert elapsed < 2.0, f"10k simulations took {elapsed:.2f}s"

    def test_seed_reproducibility(self) -> None:
        calc1 = EquityCalculator(seed=123)
        calc2 = EquityCalculator(seed=123)
        kwargs = dict(
            hand=(AS, AH), board=[], num_opponents=1, num_simulations=1_000
        )
        eq1 = calc1.calculate_equity(**kwargs)
        eq2 = calc2.calculate_equity(**kwargs)
        assert eq1 == eq2
