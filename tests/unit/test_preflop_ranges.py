"""Unit tests for src.solver.preflop_ranges module."""

from __future__ import annotations

import pytest
from src.detection.card import Card, Rank, Suit
from src.engine.game_state import Position
from src.solver.preflop_ranges import (
    HandRange,
    PreflopRangeTable,
    hand_to_notation,
    notation_to_matrix_pos,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _card(rank: Rank, suit: Suit) -> Card:
    return Card(rank=rank, suit=suit)


AS = _card(Rank.ACE, Suit.SPADES)
AH = _card(Rank.ACE, Suit.HEARTS)
AD = _card(Rank.ACE, Suit.DIAMONDS)
KS = _card(Rank.KING, Suit.SPADES)
KH = _card(Rank.KING, Suit.HEARTS)
QS = _card(Rank.QUEEN, Suit.SPADES)
QH = _card(Rank.QUEEN, Suit.HEARTS)
JS = _card(Rank.JACK, Suit.SPADES)
TS = _card(Rank.TEN, Suit.SPADES)
TH = _card(Rank.TEN, Suit.HEARTS)
NINE_S = _card(Rank.NINE, Suit.SPADES)
TWO_H = _card(Rank.TWO, Suit.HEARTS)
TWO_C = _card(Rank.TWO, Suit.CLUBS)
SEVEN_D = _card(Rank.SEVEN, Suit.DIAMONDS)


# ---------------------------------------------------------------------------
# hand_to_notation
# ---------------------------------------------------------------------------

class TestHandToNotation:
    def test_pocket_pair(self) -> None:
        assert hand_to_notation(AS, AH) == "AA"

    def test_suited(self) -> None:
        assert hand_to_notation(AS, KS) == "AKs"

    def test_offsuit(self) -> None:
        assert hand_to_notation(AS, KH) == "AKo"

    def test_order_independent(self) -> None:
        assert hand_to_notation(KH, AS) == "AKo"
        assert hand_to_notation(KS, AS) == "AKs"

    def test_low_cards(self) -> None:
        assert hand_to_notation(TWO_H, TWO_C) == "22"


# ---------------------------------------------------------------------------
# notation_to_matrix_pos
# ---------------------------------------------------------------------------

class TestNotationToMatrixPos:
    def test_pocket_aces_top_left(self) -> None:
        row, col = notation_to_matrix_pos("AA")
        assert row == 0
        assert col == 0

    def test_pocket_twos_bottom_right(self) -> None:
        row, col = notation_to_matrix_pos("22")
        assert row == 12
        assert col == 12

    def test_suited_above_diagonal(self) -> None:
        row, col = notation_to_matrix_pos("AKs")
        assert row < col  # suited is above diagonal

    def test_offsuit_below_diagonal(self) -> None:
        row, col = notation_to_matrix_pos("AKo")
        assert row > col  # offsuit is below diagonal


# ---------------------------------------------------------------------------
# HandRange
# ---------------------------------------------------------------------------

class TestHandRange:
    def test_add_and_contains(self) -> None:
        hr = HandRange()
        hr.add_hand("AA", 1.0)
        assert hr.contains("AA")
        assert not hr.contains("KK")

    def test_weight_validation(self) -> None:
        hr = HandRange()
        with pytest.raises(ValueError, match="Weight must be"):
            hr.add_hand("AA", 1.5)
        with pytest.raises(ValueError, match="Weight must be"):
            hr.add_hand("AA", -0.1)

    def test_remove_hand(self) -> None:
        hr = HandRange()
        hr.add_hand("AA", 1.0)
        hr.remove_hand("AA")
        assert not hr.contains("AA")

    def test_remove_nonexistent_no_error(self) -> None:
        hr = HandRange()
        hr.remove_hand("AA")  # Should not raise

    def test_size(self) -> None:
        hr = HandRange()
        hr.add_hand("AA", 1.0)
        hr.add_hand("KK", 1.0)
        hr.add_hand("QQ", 0.0)  # Weight 0 should not count
        assert hr.size == 2

    def test_total_combos(self) -> None:
        hr = HandRange()
        hr.add_hand("AA", 1.0)  # 6 combos
        hr.add_hand("AKs", 1.0)  # 4 combos
        hr.add_hand("AKo", 0.5)  # 12 * 0.5 = 6 combos
        assert hr.total_combos == pytest.approx(16.0)

    def test_len_and_in(self) -> None:
        hr = HandRange()
        hr.add_hand("AA")
        hr.add_hand("KK")
        assert len(hr) == 2
        assert "AA" in hr

    def test_get_weight_default(self) -> None:
        hr = HandRange()
        assert hr.get_weight("AA") == 0.0

    def test_get_weight_set(self) -> None:
        hr = HandRange()
        hr.add_hand("AA", 0.75)
        assert hr.get_weight("AA") == 0.75


# ---------------------------------------------------------------------------
# PreflopRangeTable
# ---------------------------------------------------------------------------

class TestPreflopRangeTable:
    def setup_method(self) -> None:
        self.table = PreflopRangeTable()

    def test_utg_range_tightest(self) -> None:
        utg = self.table.get_open_range(Position.UTG)
        btn = self.table.get_open_range(Position.BTN)
        assert utg.size < btn.size

    def test_btn_range_widest(self) -> None:
        ranges = {
            pos: self.table.get_open_range(pos).size
            for pos in [Position.UTG, Position.CO, Position.BTN]
        }
        assert ranges[Position.BTN] > ranges[Position.CO] > ranges[Position.UTG]

    def test_open_range_contains_premium(self) -> None:
        """Every position should open AA, KK, AKs."""
        for pos in [Position.UTG, Position.CO, Position.BTN, Position.SB]:
            hr = self.table.get_open_range(pos)
            assert "AA" in hr
            assert "KK" in hr
            assert "AKs" in hr

    def test_utg_does_not_open_trash(self) -> None:
        utg = self.table.get_open_range(Position.UTG)
        assert "72o" not in utg
        assert "83o" not in utg

    def test_three_bet_range(self) -> None:
        hr = self.table.get_three_bet_range(Position.BTN, Position.UTG)
        assert "AA" in hr
        assert "KK" in hr

    def test_three_bet_wider_lp_vs_ep(self) -> None:
        wide = self.table.get_three_bet_range(Position.BTN, Position.UTG)
        tight = self.table.get_three_bet_range(Position.UTG, Position.UTG1)
        assert wide.size >= tight.size

    def test_call_range(self) -> None:
        hr = self.table.get_call_range(Position.BB, Position.BTN)
        assert hr.size > 0

    def test_bb_call_range_widest(self) -> None:
        bb = self.table.get_call_range(Position.BB, Position.BTN)
        co = self.table.get_call_range(Position.CO, Position.BTN)
        assert bb.size >= co.size

    def test_four_bet_range_narrow(self) -> None:
        hr = self.table.get_four_bet_range()
        assert "AA" in hr
        assert "KK" in hr
        assert hr.size < 10

    def test_is_in_range(self) -> None:
        open_range = self.table.get_open_range(Position.BTN)
        assert self.table.is_in_range((AS, AH), open_range)
        assert not self.table.is_in_range((TWO_H, SEVEN_D), open_range)

    def test_recommendation_open_raise(self) -> None:
        rec = self.table.get_recommendation(
            (AS, AH), Position.UTG, action_facing="open"
        )
        assert rec == "raise"

    def test_recommendation_open_fold_trash(self) -> None:
        rec = self.table.get_recommendation(
            (TWO_H, SEVEN_D), Position.UTG, action_facing="open"
        )
        assert rec == "fold"

    def test_recommendation_facing_raise_call(self) -> None:
        # JTs should be in BB call range vs UTG open (BB is widest caller).
        rec = self.table.get_recommendation(
            (JS, TS), Position.BB, action_facing="raise", vs_position=Position.UTG
        )
        assert rec in ("call", "raise")

    def test_recommendation_facing_raise_fold_trash(self) -> None:
        rec = self.table.get_recommendation(
            (TWO_H, SEVEN_D), Position.CO, action_facing="raise",
            vs_position=Position.UTG,
        )
        assert rec == "fold"

    def test_recommendation_facing_3bet_premium(self) -> None:
        rec = self.table.get_recommendation(
            (AS, AH), Position.UTG, action_facing="three_bet",
            vs_position=Position.BTN,
        )
        assert rec == "raise"

    def test_caching(self) -> None:
        """Repeated calls should return cached results."""
        r1 = self.table.get_open_range(Position.BTN)
        r2 = self.table.get_open_range(Position.BTN)
        assert r1 is r2
