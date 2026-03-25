"""Player statistics calculations for poker HUD.

Computes standard poker statistics (VPIP, PFR, 3-Bet%, AF, WTSD, C-Bet%)
from raw hand action records.  Each calculation method is a pure function
that takes a list of ``HandRecord`` entries and returns a float percentage
(0.0-100.0) or ratio.

Usage:
    from src.stats.calculations import StatCalculator, PlayerStats

    calc = StatCalculator()
    stats = calc.calculate_all(hand_records)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Sequence

from src.engine.game_state import ActionType, Street

# ---------------------------------------------------------------------------
# Data structures for hand records fed into stat calculations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HandActionRecord:
    """A single action within a completed hand, used for stat calculation.

    This is a lightweight, immutable record extracted from ``GameState``
    after a hand completes.  It carries only the fields needed for
    statistics, decoupled from live game state.

    Attributes:
        hand_id: Unique identifier for the hand.
        player_name: Name of the player who acted.
        street: The street on which the action occurred.
        action_type: The type of action taken.
        amount: Chip amount (0 for folds/checks).
        is_blind: Whether this action is a forced blind post.
    """

    hand_id: str
    player_name: str
    street: Street
    action_type: ActionType
    amount: float = 0.0
    is_blind: bool = False


class ConfidenceLevel(Enum):
    """Confidence in the accuracy of computed statistics based on sample size."""

    VERY_LOW = auto()   # < 10 hands
    LOW = auto()        # 10-49 hands
    MEDIUM = auto()     # 50-199 hands
    HIGH = auto()       # 200-999 hands
    VERY_HIGH = auto()  # 1000+ hands


def confidence_from_sample_size(total_hands: int) -> ConfidenceLevel:
    """Determine confidence level from the number of observed hands.

    Args:
        total_hands: Number of hands in the sample.

    Returns:
        The corresponding confidence level.
    """
    if total_hands < 10:
        return ConfidenceLevel.VERY_LOW
    if total_hands < 50:
        return ConfidenceLevel.LOW
    if total_hands < 200:
        return ConfidenceLevel.MEDIUM
    if total_hands < 1000:
        return ConfidenceLevel.HIGH
    return ConfidenceLevel.VERY_HIGH


@dataclass
class PlayerStats:
    """Aggregated statistics for a single player.

    All percentage stats are expressed as 0.0-100.0.

    Attributes:
        vpip: Voluntarily Put $ In Pot percentage.
        pfr: Pre-Flop Raise percentage.
        three_bet_pct: 3-Bet percentage.
        fold_to_three_bet: Fold to 3-Bet percentage.
        cbet_pct: Continuation Bet percentage.
        aggression_factor: (bets + raises) / calls.
        wtsd: Went To Showdown percentage.
        total_hands: Total hands observed.
        total_profit: Cumulative profit/loss.
        confidence: Confidence level based on sample size.
    """

    vpip: float = 0.0
    pfr: float = 0.0
    three_bet_pct: float = 0.0
    fold_to_three_bet: float = 0.0
    cbet_pct: float = 0.0
    aggression_factor: float = 0.0
    wtsd: float = 0.0
    total_hands: int = 0
    total_profit: float = 0.0
    confidence: ConfidenceLevel = ConfidenceLevel.VERY_LOW


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or 0.0 when denominator is zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _unique_hand_ids(records: Sequence[HandActionRecord]) -> set[str]:
    """Return the set of distinct hand IDs present in *records*."""
    return {r.hand_id for r in records}


def _preflop_actions(
    records: Sequence[HandActionRecord],
) -> list[HandActionRecord]:
    """Filter to preflop actions only, excluding blind posts."""
    return [
        r for r in records
        if r.street == Street.PREFLOP and not r.is_blind
    ]


# ---------------------------------------------------------------------------
# StatCalculator
# ---------------------------------------------------------------------------


class StatCalculator:
    """Computes individual poker statistics from hand action records.

    All public methods accept a sequence of ``HandActionRecord`` for a
    single player and return a float.  Percentage stats are 0.0-100.0.
    """

    def vpip(self, actions: Sequence[HandActionRecord]) -> float:
        """Voluntarily Put $ In Pot: % of hands where player called or raised preflop.

        Excludes forced blind posts.  A hand counts as VPIP if the player
        made any voluntary call, raise, bet, or all-in action preflop.

        Args:
            actions: All action records for one player.

        Returns:
            VPIP percentage (0.0-100.0).
        """
        hand_ids = _unique_hand_ids(actions)
        if not hand_ids:
            return 0.0

        preflop = _preflop_actions(actions)
        vpip_hands: set[str] = set()
        for a in preflop:
            if a.action_type in (
                ActionType.CALL,
                ActionType.RAISE,
                ActionType.BET,
                ActionType.ALL_IN,
            ):
                vpip_hands.add(a.hand_id)

        return _safe_divide(len(vpip_hands), len(hand_ids)) * 100.0

    def pfr(self, actions: Sequence[HandActionRecord]) -> float:
        """Pre-Flop Raise: % of hands where player raised preflop.

        Args:
            actions: All action records for one player.

        Returns:
            PFR percentage (0.0-100.0).
        """
        hand_ids = _unique_hand_ids(actions)
        if not hand_ids:
            return 0.0

        preflop = _preflop_actions(actions)
        pfr_hands: set[str] = set()
        for a in preflop:
            if a.action_type in (ActionType.RAISE, ActionType.BET, ActionType.ALL_IN):
                pfr_hands.add(a.hand_id)

        return _safe_divide(len(pfr_hands), len(hand_ids)) * 100.0

    def three_bet_pct(self, actions: Sequence[HandActionRecord]) -> float:
        """3-Bet%: % of opportunities where player re-raised preflop.

        An "opportunity" is a hand where the player faced a raise preflop
        (i.e., there was at least one raise before their action).  The
        player 3-bet if they then raised.

        To detect 3-bet opportunities properly, *actions* should include
        all players' preflop actions for the hands in question, but only
        the target player's records are used for the numerator.

        For simplicity, this method works with per-player records and uses
        a heuristic: if the player raised preflop AND there was a prior
        raise in that hand, it counts as a 3-bet.  The ``faced_raise``
        and ``reraised`` flags should be pre-computed by the aggregator.

        Simplified approach: counts hands where player raised preflop out
        of hands where they had an opportunity (faced a raise).

        Args:
            actions: All action records for one player, plus context
                records from other players sharing the same hand IDs.

        Returns:
            3-Bet percentage (0.0-100.0).
        """
        hand_ids = _unique_hand_ids(actions)
        if not hand_ids:
            return 0.0

        # For each hand, determine if the target player faced a raise and
        # whether they re-raised.  We need to know the target player's name.
        player_names = {a.player_name for a in actions}
        if len(player_names) != 1:
            # If mixed players, cannot determine 3-bet without more context.
            # Fall back to 0.
            return 0.0

        preflop = _preflop_actions(actions)

        # Group by hand
        hands: dict[str, list[HandActionRecord]] = {}
        for a in preflop:
            hands.setdefault(a.hand_id, []).append(a)

        opportunities = 0
        three_bets = 0
        for hand_actions in hands.values():
            # Check if there was a raise before the player acted again
            raise_count = 0
            faced_raise = False
            for a in hand_actions:
                if a.action_type in (
                    ActionType.RAISE, ActionType.BET, ActionType.ALL_IN,
                ):
                    raise_count += 1
                    if raise_count >= 2:
                        three_bets += 1
                        break
                if raise_count >= 1 and a.action_type not in (
                    ActionType.RAISE, ActionType.BET, ActionType.ALL_IN,
                ):
                    # Player faced a raise but did not re-raise
                    faced_raise = True
                    break

            if faced_raise or raise_count >= 2:
                opportunities += 1

        return _safe_divide(three_bets, opportunities) * 100.0

    def fold_to_three_bet(self, actions: Sequence[HandActionRecord]) -> float:
        """Fold to 3-Bet: % of times player folded when facing a 3-bet.

        Requires per-player records.  Counts hands where the player raised
        preflop and then folded to a subsequent re-raise.

        This method uses per-player action sequence heuristics:
        - Player raised preflop (original raiser).
        - Player then folded preflop (implies they faced a re-raise).

        Args:
            actions: All action records for one player.

        Returns:
            Fold-to-3-Bet percentage (0.0-100.0).
        """
        preflop = _preflop_actions(actions)
        if not preflop:
            return 0.0

        # Group by hand
        hands: dict[str, list[HandActionRecord]] = {}
        for a in preflop:
            hands.setdefault(a.hand_id, []).append(a)

        opportunities = 0
        folds = 0
        for hand_actions in hands.values():
            raised = False
            for a in hand_actions:
                if not raised and a.action_type in (
                    ActionType.RAISE, ActionType.BET,
                ):
                    raised = True
                    continue
                if raised and a.action_type == ActionType.FOLD:
                    opportunities += 1
                    folds += 1
                    break
                if raised and a.action_type in (
                    ActionType.CALL, ActionType.RAISE, ActionType.ALL_IN,
                ):
                    # Faced 3-bet but didn't fold
                    opportunities += 1
                    break

        return _safe_divide(folds, opportunities) * 100.0

    def cbet_pct(self, actions: Sequence[HandActionRecord]) -> float:
        """Continuation Bet%: % of times preflop raiser bet on the flop.

        Counts hands where:
        - Player raised preflop (was the preflop aggressor).
        - Player had an opportunity to act on the flop.
        - Numerator: player bet on the flop.

        Args:
            actions: All action records for one player.

        Returns:
            C-Bet percentage (0.0-100.0).
        """
        if not actions:
            return 0.0

        # Group by hand
        hands: dict[str, list[HandActionRecord]] = {}
        for a in actions:
            if not a.is_blind:
                hands.setdefault(a.hand_id, []).append(a)

        opportunities = 0
        cbets = 0
        for hand_actions in hands.values():
            # Check if player raised preflop
            raised_preflop = any(
                a.action_type in (ActionType.RAISE, ActionType.BET, ActionType.ALL_IN)
                and a.street == Street.PREFLOP
                for a in hand_actions
            )
            if not raised_preflop:
                continue

            # Check flop actions
            flop_actions = [a for a in hand_actions if a.street == Street.FLOP]
            if not flop_actions:
                continue  # Didn't see the flop (hand ended preflop or player folded)

            opportunities += 1
            if any(
                a.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)
                for a in flop_actions
            ):
                cbets += 1

        return _safe_divide(cbets, opportunities) * 100.0

    def aggression_factor(self, actions: Sequence[HandActionRecord]) -> float:
        """Aggression Factor: (bets + raises) / calls.

        Computed across all streets, excluding blind posts.  If the player
        has zero calls, returns ``float('inf')`` when there are aggressive
        actions, or 0.0 if there are no actions at all.

        Args:
            actions: All action records for one player.

        Returns:
            Aggression factor as a float.
        """
        non_blind = [a for a in actions if not a.is_blind]
        if not non_blind:
            return 0.0

        aggressive = sum(
            1
            for a in non_blind
            if a.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)
        )
        passive = sum(
            1 for a in non_blind if a.action_type == ActionType.CALL
        )

        if passive == 0:
            return float("inf") if aggressive > 0 else 0.0

        return aggressive / passive

    def wtsd(self, actions: Sequence[HandActionRecord]) -> float:
        """Went To Showdown%: % of hands that saw the flop and reached showdown.

        Only considers hands where the player saw the flop (had actions on
        flop or later streets).  The hand "went to showdown" if the player
        had actions on the SHOWDOWN street, or was still active on the river
        (did not fold).

        Args:
            actions: All action records for one player.

        Returns:
            WTSD percentage (0.0-100.0).
        """
        if not actions:
            return 0.0

        # Group by hand
        hands: dict[str, list[HandActionRecord]] = {}
        for a in actions:
            hands.setdefault(a.hand_id, []).append(a)

        saw_flop = 0
        went_to_showdown = 0
        for hand_actions in hands.values():
            streets_seen = {a.street for a in hand_actions}
            post_preflop = streets_seen - {Street.PREFLOP}
            if not post_preflop:
                continue  # Never saw the flop

            saw_flop += 1

            # Did they reach showdown?
            if Street.SHOWDOWN in streets_seen:
                went_to_showdown += 1
                continue

            # If they acted on the river and didn't fold, count as showdown
            river_actions = [a for a in hand_actions if a.street == Street.RIVER]
            if river_actions:
                folded_river = any(
                    a.action_type == ActionType.FOLD for a in river_actions
                )
                if not folded_river:
                    went_to_showdown += 1

        return _safe_divide(went_to_showdown, saw_flop) * 100.0

    def calculate_all(
        self,
        actions: Sequence[HandActionRecord],
        total_profit: float = 0.0,
    ) -> PlayerStats:
        """Compute all statistics for a player from their action history.

        Args:
            actions: All action records for one player.
            total_profit: Cumulative profit/loss to include in the result.

        Returns:
            A fully populated ``PlayerStats`` instance.
        """
        total_hands = len(_unique_hand_ids(actions))
        return PlayerStats(
            vpip=self.vpip(actions),
            pfr=self.pfr(actions),
            three_bet_pct=self.three_bet_pct(actions),
            fold_to_three_bet=self.fold_to_three_bet(actions),
            cbet_pct=self.cbet_pct(actions),
            aggression_factor=self.aggression_factor(actions),
            wtsd=self.wtsd(actions),
            total_hands=total_hands,
            total_profit=total_profit,
            confidence=confidence_from_sample_size(total_hands),
        )
