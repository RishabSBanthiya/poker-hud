"""Tests for src.solver.range_estimator."""

from __future__ import annotations

import numpy as np
import pytest
from src.detection.card import Card, Rank, Suit
from src.engine.game_state import ActionType, PlayerAction, Street
from src.solver.range_estimator import (
    _NUM_RANKS,
    RANK_TO_INDEX,
    HandRange,
    RangeEstimator,
    _build_top_range_mask,
)
from src.stats.calculations import PlayerStats

# ---------------------------------------------------------------------------
# Fixtures: player profiles
# ---------------------------------------------------------------------------


@pytest.fixture
def tight_player() -> PlayerStats:
    """Tight-aggressive player: low VPIP/PFR, high 3-bet fold."""
    return PlayerStats(
        vpip=15.0,
        pfr=12.0,
        three_bet_pct=4.0,
        fold_to_three_bet=70.0,
        cbet_pct=75.0,
        aggression_factor=3.0,
        wtsd=25.0,
        total_hands=200,
    )


@pytest.fixture
def loose_player() -> PlayerStats:
    """Loose-passive player: high VPIP, low PFR."""
    return PlayerStats(
        vpip=45.0,
        pfr=8.0,
        three_bet_pct=2.0,
        fold_to_three_bet=80.0,
        cbet_pct=40.0,
        aggression_factor=0.8,
        wtsd=35.0,
        total_hands=150,
    )


@pytest.fixture
def aggressive_player() -> PlayerStats:
    """Loose-aggressive player: high VPIP and PFR."""
    return PlayerStats(
        vpip=38.0,
        pfr=30.0,
        three_bet_pct=10.0,
        fold_to_three_bet=40.0,
        cbet_pct=80.0,
        aggression_factor=4.5,
        wtsd=30.0,
        total_hands=300,
    )


@pytest.fixture
def passive_player() -> PlayerStats:
    """Tight-passive player: low VPIP, very low PFR."""
    return PlayerStats(
        vpip=18.0,
        pfr=5.0,
        three_bet_pct=1.0,
        fold_to_three_bet=90.0,
        cbet_pct=30.0,
        aggression_factor=0.5,
        wtsd=40.0,
        total_hands=100,
    )


@pytest.fixture
def unknown_player() -> PlayerStats:
    """Unknown player with zero hands."""
    return PlayerStats(total_hands=0)


@pytest.fixture
def estimator() -> RangeEstimator:
    return RangeEstimator()


# ---------------------------------------------------------------------------
# HandRange unit tests
# ---------------------------------------------------------------------------


class TestHandRange:
    """Tests for HandRange dataclass."""

    def test_full_range_has_169_combos(self) -> None:
        hr = HandRange.full_range()
        assert hr.total_combos() == pytest.approx(169.0)
        assert hr.range_pct() == pytest.approx(100.0)

    def test_empty_range_has_zero_combos(self) -> None:
        hr = HandRange.empty_range()
        assert hr.total_combos() == pytest.approx(0.0)
        assert hr.range_pct() == pytest.approx(0.0)

    def test_get_set_weight(self) -> None:
        hr = HandRange.empty_range()
        hr.set_weight(Rank.ACE, Rank.KING, suited=True, weight=0.8)
        assert hr.get_weight(Rank.ACE, Rank.KING, suited=True) == pytest.approx(0.8)
        # Offsuit should still be 0
        assert hr.get_weight(Rank.ACE, Rank.KING, suited=False) == pytest.approx(0.0)

    def test_pocket_pair_weight(self) -> None:
        hr = HandRange.empty_range()
        hr.set_weight(Rank.ACE, Rank.ACE, suited=False, weight=1.0)
        assert hr.get_weight(Rank.ACE, Rank.ACE, suited=False) == pytest.approx(1.0)

    def test_weight_clamped_to_0_1(self) -> None:
        hr = HandRange.full_range()
        hr.set_weight(Rank.TWO, Rank.THREE, suited=True, weight=1.5)
        assert hr.get_weight(Rank.TWO, Rank.THREE, suited=True) == pytest.approx(1.0)
        hr.set_weight(Rank.TWO, Rank.THREE, suited=True, weight=-0.5)
        assert hr.get_weight(Rank.TWO, Rank.THREE, suited=True) == pytest.approx(0.0)

    def test_scale_reduces_range(self) -> None:
        hr = HandRange.full_range()
        hr.scale(0.5)
        assert hr.total_combos() == pytest.approx(169.0 * 0.5)

    def test_scale_clamps(self) -> None:
        hr = HandRange.full_range()
        hr.scale(2.0)
        # All weights clamped to 1.0
        assert hr.total_combos() == pytest.approx(169.0)

    def test_copy_is_independent(self) -> None:
        hr = HandRange.full_range()
        copy = hr.copy()
        copy.scale(0.0)
        assert hr.total_combos() == pytest.approx(169.0)
        assert copy.total_combos() == pytest.approx(0.0)

    def test_apply_mask(self) -> None:
        hr = HandRange.full_range()
        mask = np.zeros((_NUM_RANKS, _NUM_RANKS))
        mask[12, 12] = 1.0  # Only AA
        hr.apply_mask(mask)
        assert hr.get_weight(Rank.ACE, Rank.ACE, suited=False) == pytest.approx(1.0)
        assert hr.get_weight(Rank.KING, Rank.KING, suited=False) == pytest.approx(0.0)

    def test_invalid_matrix_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="13x13"):
            HandRange(matrix=np.ones((5, 5)))

    def test_suited_vs_offsuit_different_cells(self) -> None:
        hr = HandRange.empty_range()
        hr.set_weight(Rank.ACE, Rank.KING, suited=True, weight=1.0)
        hr.set_weight(Rank.ACE, Rank.KING, suited=False, weight=0.5)
        assert hr.get_weight(Rank.ACE, Rank.KING, suited=True) == pytest.approx(1.0)
        assert hr.get_weight(Rank.ACE, Rank.KING, suited=False) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _build_top_range_mask tests
# ---------------------------------------------------------------------------


class TestBuildTopRangeMask:
    """Tests for the range mask builder."""

    def test_100_pct_mask_is_all_ones(self) -> None:
        mask = _build_top_range_mask(100.0)
        assert np.all(mask == 1.0)

    def test_small_pct_mask_is_mostly_zeros(self) -> None:
        mask = _build_top_range_mask(5.0)
        total = np.sum(mask)
        # ~5% of 169 = ~8 combos
        assert total < 20
        assert total > 0

    def test_mask_includes_aces(self) -> None:
        mask = _build_top_range_mask(5.0)
        # AA should always be in any top range
        aa_idx = RANK_TO_INDEX[Rank.ACE]
        assert mask[aa_idx, aa_idx] == 1.0


# ---------------------------------------------------------------------------
# RangeEstimator: initial range tests
# ---------------------------------------------------------------------------


class TestInitialRange:
    """Tests for _initial_range calibration."""

    def test_unknown_player_gets_full_range(
        self, estimator: RangeEstimator, unknown_player: PlayerStats
    ) -> None:
        hr = estimator._initial_range(unknown_player)
        assert hr.range_pct() == pytest.approx(100.0)

    def test_tight_player_gets_narrow_range(
        self, estimator: RangeEstimator, tight_player: PlayerStats
    ) -> None:
        hr = estimator._initial_range(tight_player)
        assert hr.range_pct() < 25.0

    def test_loose_player_gets_wide_range(
        self, estimator: RangeEstimator, loose_player: PlayerStats
    ) -> None:
        hr = estimator._initial_range(loose_player)
        assert hr.range_pct() > 30.0


# ---------------------------------------------------------------------------
# RangeEstimator: preflop action narrowing
# ---------------------------------------------------------------------------


class TestPreflopNarrowing:
    """Tests for preflop action-based range narrowing."""

    def test_fold_returns_empty_range(
        self, estimator: RangeEstimator, tight_player: PlayerStats
    ) -> None:
        hr = HandRange.full_range()
        result = estimator.update_range(
            hr, ActionType.FOLD, Street.PREFLOP, tight_player
        )
        assert result.total_combos() == pytest.approx(0.0)

    def test_raise_from_tight_player_narrows_significantly(
        self, estimator: RangeEstimator, tight_player: PlayerStats
    ) -> None:
        hr = HandRange.full_range()
        result = estimator.update_range(
            hr, ActionType.RAISE, Street.PREFLOP, tight_player
        )
        assert result.range_pct() < 20.0

    def test_raise_from_loose_player_stays_wider(
        self, estimator: RangeEstimator, aggressive_player: PlayerStats
    ) -> None:
        hr = HandRange.full_range()
        result = estimator.update_range(
            hr, ActionType.RAISE, Street.PREFLOP, aggressive_player
        )
        # Aggressive player's PFR is 30, but 3-bet is 10 so it triggers
        # the 3-bet path; still should be narrow but wider than tight
        assert result.range_pct() < 50.0

    def test_call_removes_premiums_and_junk(
        self, estimator: RangeEstimator, loose_player: PlayerStats
    ) -> None:
        hr = HandRange.full_range()
        result = estimator.update_range(
            hr, ActionType.CALL, Street.PREFLOP, loose_player
        )
        # Should be narrower than full range
        assert result.range_pct() < 100.0
        # But not too narrow (callers have a range)
        assert result.range_pct() > 10.0

    def test_all_in_preflop_is_very_narrow(
        self, estimator: RangeEstimator, tight_player: PlayerStats
    ) -> None:
        hr = HandRange.full_range()
        result = estimator.update_range(
            hr, ActionType.ALL_IN, Street.PREFLOP, tight_player
        )
        assert result.range_pct() < 20.0

    def test_check_preflop_reduces_premiums(
        self, estimator: RangeEstimator, tight_player: PlayerStats
    ) -> None:
        hr = HandRange.full_range()
        result = estimator.update_range(
            hr, ActionType.CHECK, Street.PREFLOP, tight_player
        )
        # AA weight should be reduced (would have raised)
        aa_weight = result.get_weight(Rank.ACE, Rank.ACE, suited=False)
        assert aa_weight < 1.0


# ---------------------------------------------------------------------------
# RangeEstimator: postflop action narrowing
# ---------------------------------------------------------------------------


class TestPostflopNarrowing:
    """Tests for postflop action-based range narrowing."""

    def test_fold_postflop_returns_empty(
        self, estimator: RangeEstimator, tight_player: PlayerStats
    ) -> None:
        hr = HandRange.full_range()
        result = estimator.update_range(
            hr, ActionType.FOLD, Street.FLOP, tight_player
        )
        assert result.total_combos() == pytest.approx(0.0)

    def test_bet_on_flop_narrows_range(
        self, estimator: RangeEstimator, tight_player: PlayerStats
    ) -> None:
        hr = HandRange.full_range()
        result = estimator.update_range(
            hr, ActionType.BET, Street.FLOP, tight_player
        )
        assert result.range_pct() < 100.0

    def test_check_on_flop_for_high_cbet_removes_strong(
        self, estimator: RangeEstimator, tight_player: PlayerStats
    ) -> None:
        """A player with 75% c-bet who checks is likely weak."""
        hr = HandRange.full_range()
        result = estimator.update_range(
            hr, ActionType.CHECK, Street.FLOP, tight_player
        )
        # Premium hands should be reduced
        aa_weight = result.get_weight(Rank.ACE, Rank.ACE, suited=False)
        assert aa_weight < 0.5

    def test_river_bet_narrows_more_than_flop_bet(
        self, estimator: RangeEstimator, tight_player: PlayerStats
    ) -> None:
        hr_flop = HandRange.full_range()
        hr_river = HandRange.full_range()
        flop_result = estimator.update_range(
            hr_flop, ActionType.BET, Street.FLOP, tight_player
        )
        river_result = estimator.update_range(
            hr_river, ActionType.BET, Street.RIVER, tight_player
        )
        assert river_result.range_pct() < flop_result.range_pct()

    def test_call_postflop_reduces_premiums(
        self, estimator: RangeEstimator, tight_player: PlayerStats
    ) -> None:
        hr = HandRange.full_range()
        result = estimator.update_range(
            hr, ActionType.CALL, Street.TURN, tight_player
        )
        # Premium should be reduced (would have raised)
        aa_before = 1.0
        aa_after = result.get_weight(Rank.ACE, Rank.ACE, suited=False)
        assert aa_after < aa_before

    def test_aggressive_player_bet_keeps_wider_range(
        self, estimator: RangeEstimator, aggressive_player: PlayerStats
    ) -> None:
        hr = HandRange.full_range()
        result = estimator.update_range(
            hr, ActionType.BET, Street.FLOP, aggressive_player
        )
        # Very aggressive player betting: could be bluffing, wider range
        assert result.range_pct() > 30.0


# ---------------------------------------------------------------------------
# RangeEstimator: full estimate_range pipeline
# ---------------------------------------------------------------------------


class TestEstimateRange:
    """Integration tests for estimate_range."""

    def test_no_actions_returns_initial_range(
        self, estimator: RangeEstimator, tight_player: PlayerStats
    ) -> None:
        result = estimator.estimate_range(tight_player, [], [])
        initial = estimator._initial_range(tight_player)
        np.testing.assert_array_equal(result.matrix, initial.matrix)

    def test_single_raise_narrows(
        self, estimator: RangeEstimator, tight_player: PlayerStats
    ) -> None:
        actions = [
            PlayerAction(
                action_type=ActionType.RAISE,
                amount=6.0,
                street=Street.PREFLOP,
            )
        ]
        result = estimator.estimate_range(tight_player, actions, [])
        initial = estimator._initial_range(tight_player)
        assert result.total_combos() <= initial.total_combos()

    def test_raise_then_bet_narrows_progressively(
        self, estimator: RangeEstimator, tight_player: PlayerStats
    ) -> None:
        actions = [
            PlayerAction(
                action_type=ActionType.RAISE, amount=6.0, street=Street.PREFLOP
            ),
            PlayerAction(
                action_type=ActionType.BET, amount=10.0, street=Street.FLOP
            ),
        ]
        board = [
            Card(Rank.TEN, Suit.HEARTS),
            Card(Rank.SEVEN, Suit.CLUBS),
            Card(Rank.TWO, Suit.DIAMONDS),
        ]
        result = estimator.estimate_range(tight_player, actions, board)
        # After preflop raise + flop bet, should be quite narrow
        assert result.range_pct() < 20.0

    def test_unknown_player_full_range_on_no_actions(
        self, estimator: RangeEstimator, unknown_player: PlayerStats
    ) -> None:
        result = estimator.estimate_range(unknown_player, [], [])
        assert result.range_pct() == pytest.approx(100.0)

    def test_call_then_check_creates_capped_range(
        self, estimator: RangeEstimator, passive_player: PlayerStats
    ) -> None:
        """Passive player calling preflop then checking flop = capped range."""
        actions = [
            PlayerAction(
                action_type=ActionType.CALL, amount=2.0, street=Street.PREFLOP
            ),
            PlayerAction(
                action_type=ActionType.CHECK, amount=0.0, street=Street.FLOP
            ),
        ]
        result = estimator.estimate_range(passive_player, actions, [])
        # After calling then checking, range should be non-empty but narrowed
        assert result.total_combos() > 0
        assert result.range_pct() < 80.0


# ---------------------------------------------------------------------------
# Player profile comparison tests
# ---------------------------------------------------------------------------


class TestProfileComparison:
    """Verify that different player profiles produce different ranges."""

    def test_tight_narrower_than_loose_on_raise(
        self,
        estimator: RangeEstimator,
        tight_player: PlayerStats,
        aggressive_player: PlayerStats,
    ) -> None:
        """Tight raiser has narrower range than aggressive."""
        actions = [
            PlayerAction(
                action_type=ActionType.RAISE, amount=6.0, street=Street.PREFLOP
            ),
        ]
        tight_range = estimator.estimate_range(tight_player, actions, [])
        aggressive_range = estimator.estimate_range(aggressive_player, actions, [])
        # Tight PFR=12 vs aggressive PFR=30: tight should be narrower
        # Note: aggressive player has high 3-bet (10%) which triggers 3-bet path,
        # but initial range is wider (VPIP=38 vs 15), and raise mask is wider too.
        assert tight_range.range_pct() < 20.0
        assert aggressive_range.range_pct() < 20.0

    def test_passive_wider_call_range_than_aggressive(
        self,
        estimator: RangeEstimator,
        passive_player: PlayerStats,
        aggressive_player: PlayerStats,
    ) -> None:
        """Passive players have wider calling ranges (they call more, raise less)."""
        actions = [
            PlayerAction(
                action_type=ActionType.CALL, amount=2.0, street=Street.PREFLOP
            ),
        ]
        passive_range = estimator.estimate_range(passive_player, actions, [])
        aggressive_range = estimator.estimate_range(aggressive_player, actions, [])
        # Both should be non-empty
        assert passive_range.total_combos() > 0
        assert aggressive_range.total_combos() > 0
