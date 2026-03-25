"""Unit tests for src.stats.calculations module."""

from __future__ import annotations

import pytest
from src.engine.game_state import ActionType, Street
from src.stats.calculations import (
    ConfidenceLevel,
    HandActionRecord,
    PlayerStats,
    StatCalculator,
    confidence_from_sample_size,
)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

_HAND = 0  # auto-increment hand counter


def _next_hand_id() -> str:
    global _HAND
    _HAND += 1
    return f"hand-{_HAND}"


def _action(
    hand_id: str,
    street: Street,
    action_type: ActionType,
    amount: float = 0.0,
    player: str = "Hero",
    is_blind: bool = False,
) -> HandActionRecord:
    return HandActionRecord(
        hand_id=hand_id,
        player_name=player,
        street=street,
        action_type=action_type,
        amount=amount,
        is_blind=is_blind,
    )


@pytest.fixture(autouse=True)
def _reset_hand_counter() -> None:
    global _HAND
    _HAND = 0


# -----------------------------------------------------------------------
# Confidence level
# -----------------------------------------------------------------------


class TestConfidenceLevel:
    def test_very_low(self) -> None:
        assert confidence_from_sample_size(0) == ConfidenceLevel.VERY_LOW
        assert confidence_from_sample_size(9) == ConfidenceLevel.VERY_LOW

    def test_low(self) -> None:
        assert confidence_from_sample_size(10) == ConfidenceLevel.LOW
        assert confidence_from_sample_size(49) == ConfidenceLevel.LOW

    def test_medium(self) -> None:
        assert confidence_from_sample_size(50) == ConfidenceLevel.MEDIUM
        assert confidence_from_sample_size(199) == ConfidenceLevel.MEDIUM

    def test_high(self) -> None:
        assert confidence_from_sample_size(200) == ConfidenceLevel.HIGH
        assert confidence_from_sample_size(999) == ConfidenceLevel.HIGH

    def test_very_high(self) -> None:
        assert confidence_from_sample_size(1000) == ConfidenceLevel.VERY_HIGH
        assert confidence_from_sample_size(50000) == ConfidenceLevel.VERY_HIGH


# -----------------------------------------------------------------------
# VPIP
# -----------------------------------------------------------------------


class TestVPIP:
    def setup_method(self) -> None:
        self.calc = StatCalculator()

    def test_empty_actions(self) -> None:
        assert self.calc.vpip([]) == 0.0

    def test_all_folds(self) -> None:
        """Player folds every hand preflop -- VPIP should be 0."""
        actions = []
        for _ in range(3):
            hid = _next_hand_id()
            actions.append(_action(
                hid, Street.PREFLOP, ActionType.POST_BLIND,
                1.0, is_blind=True,
            ))
            actions.append(_action(hid, Street.PREFLOP, ActionType.FOLD))
        assert self.calc.vpip(actions) == 0.0

    def test_all_calls(self) -> None:
        """Player calls every hand -- VPIP should be 100."""
        actions = []
        for _ in range(3):
            hid = _next_hand_id()
            actions.append(_action(
                hid, Street.PREFLOP, ActionType.POST_BLIND,
                2.0, is_blind=True,
            ))
            actions.append(_action(hid, Street.PREFLOP, ActionType.CALL, 2.0))
        assert self.calc.vpip(actions) == 100.0

    def test_mixed_actions(self) -> None:
        """2 calls + 1 fold out of 3 hands = 66.67%."""
        actions = []
        h1 = _next_hand_id()
        actions.append(_action(h1, Street.PREFLOP, ActionType.CALL, 2.0))
        h2 = _next_hand_id()
        actions.append(_action(h2, Street.PREFLOP, ActionType.RAISE, 6.0))
        h3 = _next_hand_id()
        actions.append(_action(h3, Street.PREFLOP, ActionType.FOLD))
        assert self.calc.vpip(actions) == pytest.approx(66.667, rel=1e-2)

    def test_raise_counts_as_vpip(self) -> None:
        hid = _next_hand_id()
        actions = [_action(hid, Street.PREFLOP, ActionType.RAISE, 6.0)]
        assert self.calc.vpip(actions) == 100.0

    def test_blind_not_counted(self) -> None:
        """A blind post alone should NOT count as VPIP."""
        hid = _next_hand_id()
        actions = [
            _action(hid, Street.PREFLOP, ActionType.POST_BLIND, 2.0, is_blind=True),
            _action(hid, Street.PREFLOP, ActionType.CHECK),
        ]
        assert self.calc.vpip(actions) == 0.0


# -----------------------------------------------------------------------
# PFR
# -----------------------------------------------------------------------


class TestPFR:
    def setup_method(self) -> None:
        self.calc = StatCalculator()

    def test_empty(self) -> None:
        assert self.calc.pfr([]) == 0.0

    def test_raises_every_hand(self) -> None:
        actions = []
        for _ in range(4):
            hid = _next_hand_id()
            actions.append(_action(hid, Street.PREFLOP, ActionType.RAISE, 6.0))
        assert self.calc.pfr(actions) == 100.0

    def test_never_raises(self) -> None:
        actions = []
        for _ in range(4):
            hid = _next_hand_id()
            actions.append(_action(hid, Street.PREFLOP, ActionType.CALL, 2.0))
        assert self.calc.pfr(actions) == 0.0

    def test_mixed(self) -> None:
        """1 raise out of 4 hands = 25%."""
        h1 = _next_hand_id()
        h2 = _next_hand_id()
        h3 = _next_hand_id()
        h4 = _next_hand_id()
        actions = [
            _action(h1, Street.PREFLOP, ActionType.RAISE, 6.0),
            _action(h2, Street.PREFLOP, ActionType.CALL, 2.0),
            _action(h3, Street.PREFLOP, ActionType.FOLD),
            _action(h4, Street.PREFLOP, ActionType.CALL, 2.0),
        ]
        assert self.calc.pfr(actions) == 25.0

    def test_all_in_counts_as_pfr(self) -> None:
        hid = _next_hand_id()
        actions = [_action(hid, Street.PREFLOP, ActionType.ALL_IN, 100.0)]
        assert self.calc.pfr(actions) == 100.0


# -----------------------------------------------------------------------
# Aggression Factor
# -----------------------------------------------------------------------


class TestAggressionFactor:
    def setup_method(self) -> None:
        self.calc = StatCalculator()

    def test_empty(self) -> None:
        assert self.calc.aggression_factor([]) == 0.0

    def test_all_aggressive(self) -> None:
        """Only bets/raises, no calls -> inf."""
        hid = _next_hand_id()
        actions = [
            _action(hid, Street.FLOP, ActionType.BET, 10.0),
            _action(hid, Street.TURN, ActionType.RAISE, 30.0),
        ]
        assert self.calc.aggression_factor(actions) == float("inf")

    def test_all_passive(self) -> None:
        """Only calls -> 0 aggressive / N calls = 0."""
        hid = _next_hand_id()
        actions = [
            _action(hid, Street.FLOP, ActionType.CALL, 10.0),
            _action(hid, Street.TURN, ActionType.CALL, 20.0),
        ]
        assert self.calc.aggression_factor(actions) == 0.0

    def test_balanced(self) -> None:
        """2 aggressive + 2 passive = AF of 1.0."""
        hid = _next_hand_id()
        actions = [
            _action(hid, Street.FLOP, ActionType.BET, 10.0),
            _action(hid, Street.TURN, ActionType.RAISE, 20.0),
            _action(hid, Street.PREFLOP, ActionType.CALL, 2.0),
            _action(hid, Street.RIVER, ActionType.CALL, 40.0),
        ]
        assert self.calc.aggression_factor(actions) == 1.0

    def test_blinds_excluded(self) -> None:
        """Blind posts should not count as aggressive or passive."""
        hid = _next_hand_id()
        actions = [
            _action(hid, Street.PREFLOP, ActionType.POST_BLIND, 2.0, is_blind=True),
            _action(hid, Street.FLOP, ActionType.BET, 10.0),
            _action(hid, Street.TURN, ActionType.CALL, 20.0),
        ]
        assert self.calc.aggression_factor(actions) == 1.0

    def test_all_in_counts_as_aggressive(self) -> None:
        hid = _next_hand_id()
        actions = [
            _action(hid, Street.FLOP, ActionType.ALL_IN, 100.0),
            _action(hid, Street.PREFLOP, ActionType.CALL, 2.0),
        ]
        assert self.calc.aggression_factor(actions) == 1.0


# -----------------------------------------------------------------------
# C-Bet%
# -----------------------------------------------------------------------


class TestCBet:
    def setup_method(self) -> None:
        self.calc = StatCalculator()

    def test_empty(self) -> None:
        assert self.calc.cbet_pct([]) == 0.0

    def test_always_cbets(self) -> None:
        """Player raises preflop and bets flop every time."""
        actions = []
        for _ in range(3):
            hid = _next_hand_id()
            actions.append(_action(hid, Street.PREFLOP, ActionType.RAISE, 6.0))
            actions.append(_action(hid, Street.FLOP, ActionType.BET, 8.0))
        assert self.calc.cbet_pct(actions) == 100.0

    def test_never_cbets(self) -> None:
        """Player raises preflop but checks flop."""
        actions = []
        for _ in range(3):
            hid = _next_hand_id()
            actions.append(_action(hid, Street.PREFLOP, ActionType.RAISE, 6.0))
            actions.append(_action(hid, Street.FLOP, ActionType.CHECK))
        assert self.calc.cbet_pct(actions) == 0.0

    def test_no_preflop_raise(self) -> None:
        """Player only calls preflop -- no cbet opportunities."""
        hid = _next_hand_id()
        actions = [
            _action(hid, Street.PREFLOP, ActionType.CALL, 2.0),
            _action(hid, Street.FLOP, ActionType.BET, 5.0),
        ]
        assert self.calc.cbet_pct(actions) == 0.0

    def test_mixed(self) -> None:
        """Raised preflop 2 hands, bet flop in 1 = 50%."""
        h1 = _next_hand_id()
        h2 = _next_hand_id()
        actions = [
            _action(h1, Street.PREFLOP, ActionType.RAISE, 6.0),
            _action(h1, Street.FLOP, ActionType.BET, 8.0),
            _action(h2, Street.PREFLOP, ActionType.RAISE, 6.0),
            _action(h2, Street.FLOP, ActionType.CHECK),
        ]
        assert self.calc.cbet_pct(actions) == 50.0

    def test_hand_ended_preflop_no_opportunity(self) -> None:
        """Player raised preflop but hand ended before flop -- not an opportunity."""
        hid = _next_hand_id()
        actions = [_action(hid, Street.PREFLOP, ActionType.RAISE, 6.0)]
        assert self.calc.cbet_pct(actions) == 0.0


# -----------------------------------------------------------------------
# WTSD
# -----------------------------------------------------------------------


class TestWTSD:
    def setup_method(self) -> None:
        self.calc = StatCalculator()

    def test_empty(self) -> None:
        assert self.calc.wtsd([]) == 0.0

    def test_always_goes_to_showdown(self) -> None:
        actions = []
        for _ in range(3):
            hid = _next_hand_id()
            actions.append(_action(hid, Street.PREFLOP, ActionType.CALL, 2.0))
            actions.append(_action(hid, Street.FLOP, ActionType.CALL, 5.0))
            actions.append(_action(hid, Street.TURN, ActionType.CALL, 10.0))
            actions.append(_action(hid, Street.RIVER, ActionType.CALL, 20.0))
            actions.append(_action(hid, Street.SHOWDOWN, ActionType.CHECK))
        assert self.calc.wtsd(actions) == 100.0

    def test_never_goes_to_showdown(self) -> None:
        """Player sees flop but always folds on flop."""
        actions = []
        for _ in range(3):
            hid = _next_hand_id()
            actions.append(_action(hid, Street.PREFLOP, ActionType.CALL, 2.0))
            actions.append(_action(hid, Street.FLOP, ActionType.FOLD))
        assert self.calc.wtsd(actions) == 0.0

    def test_never_sees_flop(self) -> None:
        """Player folds preflop -- no WTSD opportunity."""
        actions = []
        for _ in range(3):
            hid = _next_hand_id()
            actions.append(_action(hid, Street.PREFLOP, ActionType.FOLD))
        assert self.calc.wtsd(actions) == 0.0

    def test_mixed(self) -> None:
        """2 saw flop, 1 went to showdown = 50%."""
        h1 = _next_hand_id()
        h2 = _next_hand_id()
        actions = [
            # Hand 1: saw flop, went to showdown
            _action(h1, Street.PREFLOP, ActionType.CALL, 2.0),
            _action(h1, Street.FLOP, ActionType.CALL, 5.0),
            _action(h1, Street.RIVER, ActionType.CALL, 20.0),
            _action(h1, Street.SHOWDOWN, ActionType.CHECK),
            # Hand 2: saw flop, folded on turn
            _action(h2, Street.PREFLOP, ActionType.CALL, 2.0),
            _action(h2, Street.FLOP, ActionType.CALL, 5.0),
            _action(h2, Street.TURN, ActionType.FOLD),
        ]
        assert self.calc.wtsd(actions) == 50.0

    def test_river_no_fold_counts_as_showdown(self) -> None:
        """Player calls river without explicit SHOWDOWN action."""
        hid = _next_hand_id()
        actions = [
            _action(hid, Street.PREFLOP, ActionType.CALL, 2.0),
            _action(hid, Street.FLOP, ActionType.CALL, 5.0),
            _action(hid, Street.RIVER, ActionType.CALL, 20.0),
        ]
        assert self.calc.wtsd(actions) == 100.0


# -----------------------------------------------------------------------
# 3-Bet%
# -----------------------------------------------------------------------


class TestThreeBetPct:
    def setup_method(self) -> None:
        self.calc = StatCalculator()

    def test_empty(self) -> None:
        assert self.calc.three_bet_pct([]) == 0.0

    def test_no_opportunities(self) -> None:
        """Player opens but never faces a raise -- no 3-bet opportunity."""
        hid = _next_hand_id()
        actions = [_action(hid, Street.PREFLOP, ActionType.RAISE, 6.0)]
        assert self.calc.three_bet_pct(actions) == 0.0


# -----------------------------------------------------------------------
# Fold to 3-Bet
# -----------------------------------------------------------------------


class TestFoldToThreeBet:
    def setup_method(self) -> None:
        self.calc = StatCalculator()

    def test_empty(self) -> None:
        assert self.calc.fold_to_three_bet([]) == 0.0

    def test_raised_then_folded(self) -> None:
        """Player raised then folded -- implies they faced a 3-bet and folded."""
        hid = _next_hand_id()
        actions = [
            _action(hid, Street.PREFLOP, ActionType.RAISE, 6.0),
            _action(hid, Street.PREFLOP, ActionType.FOLD),
        ]
        assert self.calc.fold_to_three_bet(actions) == 100.0

    def test_raised_then_called(self) -> None:
        """Player raised then called -- faced 3-bet but didn't fold."""
        hid = _next_hand_id()
        actions = [
            _action(hid, Street.PREFLOP, ActionType.RAISE, 6.0),
            _action(hid, Street.PREFLOP, ActionType.CALL, 12.0),
        ]
        assert self.calc.fold_to_three_bet(actions) == 0.0

    def test_mixed(self) -> None:
        """Folded to 3-bet once, called once = 50%."""
        h1 = _next_hand_id()
        h2 = _next_hand_id()
        actions = [
            _action(h1, Street.PREFLOP, ActionType.RAISE, 6.0),
            _action(h1, Street.PREFLOP, ActionType.FOLD),
            _action(h2, Street.PREFLOP, ActionType.RAISE, 6.0),
            _action(h2, Street.PREFLOP, ActionType.CALL, 18.0),
        ]
        assert self.calc.fold_to_three_bet(actions) == 50.0


# -----------------------------------------------------------------------
# calculate_all
# -----------------------------------------------------------------------


class TestCalculateAll:
    def test_basic(self) -> None:
        calc = StatCalculator()
        hid = _next_hand_id()
        actions = [
            _action(hid, Street.PREFLOP, ActionType.RAISE, 6.0),
            _action(hid, Street.FLOP, ActionType.BET, 8.0),
            _action(hid, Street.TURN, ActionType.BET, 15.0),
            _action(hid, Street.RIVER, ActionType.CALL, 30.0),
            _action(hid, Street.SHOWDOWN, ActionType.CHECK),
        ]
        stats = calc.calculate_all(actions, total_profit=42.0)

        assert isinstance(stats, PlayerStats)
        assert stats.vpip == 100.0
        assert stats.pfr == 100.0
        assert stats.cbet_pct == 100.0
        assert stats.wtsd == 100.0
        assert stats.total_hands == 1
        assert stats.total_profit == 42.0
        assert stats.confidence == ConfidenceLevel.VERY_LOW
        # 3 aggressive (raise, bet, bet) / 1 passive (call) = 3.0
        assert stats.aggression_factor == 3.0

    def test_empty_returns_zeroed_stats(self) -> None:
        calc = StatCalculator()
        stats = calc.calculate_all([])
        assert stats.total_hands == 0
        assert stats.vpip == 0.0
        assert stats.pfr == 0.0
        assert stats.confidence == ConfidenceLevel.VERY_LOW


# -----------------------------------------------------------------------
# PlayerStats dataclass
# -----------------------------------------------------------------------


class TestPlayerStats:
    def test_defaults(self) -> None:
        stats = PlayerStats()
        assert stats.vpip == 0.0
        assert stats.total_hands == 0
        assert stats.confidence == ConfidenceLevel.VERY_LOW
