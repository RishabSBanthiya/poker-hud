"""Opponent hand range estimation based on actions and player statistics.

Estimates an opponent's likely holdings as a weighted 13x13 matrix of
hand combos.  Rows and columns correspond to ranks (2..A), where
cells above the diagonal are suited combos and cells on or below are
offsuit / pairs.  Weights are floats in [0.0, 1.0] representing the
probability that the combo is in the opponent's range.

Usage:
    from src.solver.range_estimator import RangeEstimator, HandRange

    estimator = RangeEstimator()
    hand_range = estimator.estimate_range(player_stats, actions, board)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.common.logging import get_logger
from src.detection.card import Card, Rank
from src.engine.game_state import ActionType, PlayerAction, Street
from src.stats.calculations import PlayerStats

logger = get_logger("solver.range_estimator")

# Rank ordering from low (index 0) to high (index 12).
RANK_ORDER: list[Rank] = [
    Rank.TWO,
    Rank.THREE,
    Rank.FOUR,
    Rank.FIVE,
    Rank.SIX,
    Rank.SEVEN,
    Rank.EIGHT,
    Rank.NINE,
    Rank.TEN,
    Rank.JACK,
    Rank.QUEEN,
    Rank.KING,
    Rank.ACE,
]

RANK_TO_INDEX: dict[Rank, int] = {r: i for i, r in enumerate(RANK_ORDER)}

# Number of ranks in a standard deck.
_NUM_RANKS = 13

# ---------------------------------------------------------------------------
# Thresholds used to classify player types from stats
# ---------------------------------------------------------------------------

# VPIP thresholds (percentages 0-100)
_VPIP_TIGHT = 20.0
_VPIP_LOOSE = 35.0

# PFR thresholds
_PFR_PASSIVE = 10.0
_PFR_AGGRESSIVE = 25.0


# ---------------------------------------------------------------------------
# HandRange
# ---------------------------------------------------------------------------


@dataclass
class HandRange:
    """A 13x13 matrix of hand combo weights.

    The matrix is indexed by rank, with rows as the higher-rank card and
    columns as the lower-rank card.  Cells above the diagonal represent
    suited combos; cells on the diagonal are pocket pairs; cells below
    the diagonal are offsuit combos.

    Attributes:
        matrix: 13x13 numpy array of weights in [0.0, 1.0].
    """

    matrix: np.ndarray = field(
        default_factory=lambda: np.ones((_NUM_RANKS, _NUM_RANKS), dtype=np.float64)
    )

    def __post_init__(self) -> None:
        """Ensure the matrix is the correct shape and dtype."""
        if self.matrix.shape != (_NUM_RANKS, _NUM_RANKS):
            raise ValueError(
                f"HandRange matrix must be {_NUM_RANKS}x{_NUM_RANKS}, "
                f"got {self.matrix.shape}"
            )

    # -- Query helpers -------------------------------------------------------

    def get_weight(self, rank1: Rank, rank2: Rank, suited: bool) -> float:
        """Return the weight for a specific hand combo.

        Args:
            rank1: First card rank.
            rank2: Second card rank.
            suited: Whether the combo is suited.

        Returns:
            Weight in [0.0, 1.0].
        """
        row, col = self._indices(rank1, rank2, suited)
        return float(self.matrix[row, col])

    def set_weight(self, rank1: Rank, rank2: Rank, suited: bool, weight: float) -> None:
        """Set the weight for a specific hand combo.

        Args:
            rank1: First card rank.
            rank2: Second card rank.
            suited: Whether the combo is suited.
            weight: New weight in [0.0, 1.0].
        """
        weight = max(0.0, min(1.0, weight))
        row, col = self._indices(rank1, rank2, suited)
        self.matrix[row, col] = weight

    def total_combos(self) -> float:
        """Return the sum of all weights (effective number of combos).

        Returns:
            Sum of the weight matrix.
        """
        return float(np.sum(self.matrix))

    def range_pct(self) -> float:
        """Return what percentage of all possible hands are in this range.

        A full range is 169 unique hand types (13 pairs + 78 suited + 78 offsuit).

        Returns:
            Percentage 0.0-100.0.
        """
        max_combos = _NUM_RANKS * _NUM_RANKS  # 169
        return (self.total_combos() / max_combos) * 100.0

    def copy(self) -> HandRange:
        """Return a deep copy of this range."""
        return HandRange(matrix=self.matrix.copy())

    # -- Mutation helpers ----------------------------------------------------

    def scale(self, factor: float) -> None:
        """Multiply all weights by a factor, clamping to [0.0, 1.0].

        Args:
            factor: Multiplicative scaling factor.
        """
        self.matrix *= factor
        np.clip(self.matrix, 0.0, 1.0, out=self.matrix)

    def apply_mask(self, mask: np.ndarray) -> None:
        """Element-wise multiply by a mask array.

        Args:
            mask: 13x13 array of multipliers.
        """
        self.matrix *= mask
        np.clip(self.matrix, 0.0, 1.0, out=self.matrix)

    # -- Internal helpers ----------------------------------------------------

    @staticmethod
    def _indices(rank1: Rank, rank2: Rank, suited: bool) -> tuple[int, int]:
        """Convert (rank1, rank2, suited) to matrix (row, col).

        Convention: row = higher rank index, col = lower rank index.
        Suited combos are above the diagonal (row < col when we swap to
        put the higher rank in the row).  Actually, we use:
        - suited: (min_idx, max_idx) so row < col  (above diagonal)
        - offsuit/pair: (max_idx, min_idx) so row >= col  (on/below diagonal)
        """
        i1 = RANK_TO_INDEX[rank1]
        i2 = RANK_TO_INDEX[rank2]
        hi, lo = max(i1, i2), min(i1, i2)
        if hi == lo:
            # Pocket pair: on the diagonal
            return hi, lo
        if suited:
            # Above diagonal: row < col
            return lo, hi
        # Below diagonal: row > col
        return hi, lo

    @staticmethod
    def full_range() -> HandRange:
        """Create a range containing all hands at weight 1.0."""
        return HandRange(
            matrix=np.ones((_NUM_RANKS, _NUM_RANKS), dtype=np.float64)
        )

    @staticmethod
    def empty_range() -> HandRange:
        """Create an empty range with all weights at 0.0."""
        return HandRange(
            matrix=np.zeros((_NUM_RANKS, _NUM_RANKS), dtype=np.float64)
        )


# ---------------------------------------------------------------------------
# Pre-built range masks
# ---------------------------------------------------------------------------


def _build_top_range_mask(pct: float) -> np.ndarray:
    """Build a 13x13 mask that keeps roughly the top *pct*% of hands.

    Hands are ranked by a simple heuristic: high pairs first, then
    high suited broadways, then high offsuit broadways, etc.

    Args:
        pct: Target percentage of hands to include (0-100).

    Returns:
        13x13 numpy mask array.
    """
    # Build a priority matrix: higher value = stronger hand
    priority = np.zeros((_NUM_RANKS, _NUM_RANKS), dtype=np.float64)
    for r in range(_NUM_RANKS):
        for c in range(_NUM_RANKS):
            hi, lo = max(r, c), min(r, c)
            # Base value from ranks
            base = hi + lo * 0.1
            if r == c:
                # Pocket pair bonus
                base += 15.0
            elif r < c:
                # Suited bonus (above diagonal)
                base += 2.0
            priority[r, c] = base

    # Determine threshold to keep ~pct% of combos
    total_cells = _NUM_RANKS * _NUM_RANKS
    n_keep = max(1, int(total_cells * pct / 100.0))
    flat = priority.flatten()
    flat_sorted = np.sort(flat)[::-1]
    threshold = flat_sorted[min(n_keep - 1, len(flat_sorted) - 1)]

    mask = np.where(priority >= threshold, 1.0, 0.0)
    return mask


# Pre-computed masks for common range widths.
_TIGHT_MASK = _build_top_range_mask(12.0)   # ~12% of hands
_MEDIUM_MASK = _build_top_range_mask(25.0)  # ~25%
_WIDE_MASK = _build_top_range_mask(50.0)    # ~50%
_THREE_BET_MASK = _build_top_range_mask(6.0)  # ~6% for 3-bet


# ---------------------------------------------------------------------------
# RangeEstimator
# ---------------------------------------------------------------------------


class RangeEstimator:
    """Estimates an opponent's likely hand range from actions and stats.

    Uses VPIP, PFR, and 3-Bet% to calibrate how wide or narrow an
    opponent's range is, then narrows based on observed actions within
    the current hand.
    """

    def estimate_range(
        self,
        player_stats: PlayerStats,
        actions: Sequence[PlayerAction],
        board: list[Card],
    ) -> HandRange:
        """Estimate an opponent's hand range given their stats and actions.

        Starts with a position-based opening range calibrated to the
        player's VPIP, then narrows based on each observed action.

        Args:
            player_stats: Aggregated stats for this opponent.
            actions: The opponent's actions in the current hand, in order.
            board: Community cards currently on the board.

        Returns:
            Estimated HandRange with combo weights.
        """
        # Start with a range calibrated to their overall looseness
        current_range = self._initial_range(player_stats)

        # Narrow based on each observed action
        for action in actions:
            current_range = self.update_range(
                current_range,
                action.action_type,
                action.street,
                player_stats,
            )

        logger.debug(
            "range_estimated",
            range_pct=round(current_range.range_pct(), 1),
            total_combos=round(current_range.total_combos(), 1),
            num_actions=len(actions),
        )
        return current_range

    def update_range(
        self,
        current_range: HandRange,
        action: ActionType,
        street: Street,
        player_stats: PlayerStats,
    ) -> HandRange:
        """Narrow an existing range based on a single observed action.

        Args:
            current_range: The range before this action.
            action: The action type the opponent took.
            street: The street on which the action occurred.
            player_stats: The opponent's stats (for calibration).

        Returns:
            Updated (narrowed) HandRange.
        """
        updated = current_range.copy()

        if street == Street.PREFLOP:
            updated = self._update_preflop(updated, action, player_stats)
        else:
            updated = self._update_postflop(updated, action, street, player_stats)

        return updated

    # ------------------------------------------------------------------
    # Internal: initial range
    # ------------------------------------------------------------------

    def _initial_range(self, stats: PlayerStats) -> HandRange:
        """Build the starting range based on player's VPIP.

        Args:
            stats: The opponent's aggregated stats.

        Returns:
            A HandRange calibrated to their playing style.
        """
        if stats.total_hands == 0:
            # Unknown player: assume full range
            return HandRange.full_range()

        vpip = stats.vpip

        if vpip <= _VPIP_TIGHT:
            # Tight player: narrow starting range
            target_pct = max(8.0, vpip)
        elif vpip >= _VPIP_LOOSE:
            # Loose player: wide starting range
            target_pct = min(70.0, vpip)
        else:
            # Average player: moderate range
            target_pct = vpip

        mask = _build_top_range_mask(target_pct)
        return HandRange(matrix=mask.copy())

    # ------------------------------------------------------------------
    # Internal: preflop narrowing
    # ------------------------------------------------------------------

    def _update_preflop(
        self,
        hand_range: HandRange,
        action: ActionType,
        stats: PlayerStats,
    ) -> HandRange:
        """Narrow range based on a preflop action.

        Args:
            hand_range: Current estimated range.
            action: Preflop action taken.
            stats: Player stats for calibration.

        Returns:
            Narrowed range.
        """
        if action == ActionType.FOLD:
            return HandRange.empty_range()

        if action in (ActionType.RAISE, ActionType.BET):
            return self._preflop_raise_range(hand_range, stats)

        if action == ActionType.ALL_IN:
            # All-in preflop: very strong or desperate -- use tight mask
            hand_range.apply_mask(_TIGHT_MASK)
            return hand_range

        if action == ActionType.CALL:
            return self._preflop_call_range(hand_range, stats)

        if action == ActionType.CHECK:
            # Checking preflop (BB option): could be anything in their range
            # Slightly reduce top combos (would have raised)
            self._reduce_premium_hands(hand_range, factor=0.6)
            return hand_range

        # POST_BLIND or unknown: no change
        return hand_range

    def _preflop_raise_range(
        self, hand_range: HandRange, stats: PlayerStats
    ) -> HandRange:
        """Narrow range for a preflop raise.

        Tight players (low PFR) get a narrow raise range; loose
        players (high PFR) keep more hands.
        """
        pfr = stats.pfr if stats.total_hands > 0 else 20.0
        three_bet = stats.three_bet_pct

        if three_bet > 8.0:
            # Player is 3-betting: very narrow range
            hand_range.apply_mask(_THREE_BET_MASK)
            return hand_range

        if pfr <= _PFR_PASSIVE:
            # Very tight raiser
            mask = _build_top_range_mask(max(6.0, pfr))
            hand_range.apply_mask(mask)
        elif pfr >= _PFR_AGGRESSIVE:
            # Aggressive raiser: moderate narrowing
            mask = _build_top_range_mask(min(40.0, pfr))
            hand_range.apply_mask(mask)
        else:
            # Average raiser
            mask = _build_top_range_mask(pfr)
            hand_range.apply_mask(mask)

        return hand_range

    def _preflop_call_range(
        self, hand_range: HandRange, stats: PlayerStats
    ) -> HandRange:
        """Narrow range for a preflop call.

        Callers typically have medium-strength hands: suited connectors,
        small/medium pairs, suited aces.  Remove the very best hands
        (would have raised) and the worst hands (would have folded).
        """
        # Remove hands that would have raised (premium)
        self._reduce_premium_hands(hand_range, factor=0.3)

        # Remove very weak hands (would have folded)
        vpip = stats.vpip if stats.total_hands > 0 else 30.0
        weak_threshold = max(5.0, 100.0 - vpip)
        # Scale down the bottom portion
        bottom_mask = _build_top_range_mask(100.0 - weak_threshold)
        # Invert: keep only hands NOT in the bottom
        inverted = np.where(bottom_mask > 0.5, 0.3, 1.0)
        hand_range.apply_mask(inverted)

        return hand_range

    # ------------------------------------------------------------------
    # Internal: postflop narrowing
    # ------------------------------------------------------------------

    def _update_postflop(
        self,
        hand_range: HandRange,
        action: ActionType,
        street: Street,
        stats: PlayerStats,
    ) -> HandRange:
        """Narrow range based on a postflop action.

        Args:
            hand_range: Current estimated range.
            action: Postflop action taken.
            street: Current street (FLOP, TURN, RIVER).
            stats: Player stats for calibration.

        Returns:
            Narrowed range.
        """
        if action == ActionType.FOLD:
            return HandRange.empty_range()

        # Street-dependent narrowing factor: later streets narrow more
        street_factor = {
            Street.FLOP: 0.85,
            Street.TURN: 0.75,
            Street.RIVER: 0.65,
        }.get(street, 0.80)

        if action in (ActionType.BET, ActionType.RAISE):
            # Betting/raising postflop: likely connected or strong
            aggression = stats.aggression_factor if stats.total_hands > 0 else 1.5
            if aggression > 3.0:
                # Very aggressive: could be bluffing, wider range
                hand_range.scale(street_factor + 0.1)
            else:
                # Standard: narrow to stronger hands
                hand_range.scale(street_factor - 0.1)
                self._boost_premium_hands(hand_range, factor=1.2)
            return hand_range

        if action == ActionType.ALL_IN:
            # All-in postflop: polarised -- very strong or bluff
            hand_range.scale(street_factor - 0.15)
            return hand_range

        if action == ActionType.CALL:
            # Calling postflop: medium strength, drawing hands
            hand_range.scale(street_factor)
            # Reduce premium (would have raised)
            self._reduce_premium_hands(hand_range, factor=0.7)
            return hand_range

        if action == ActionType.CHECK:
            # Checking: usually weak or trapping
            cbet = stats.cbet_pct if stats.total_hands > 0 else 50.0
            if cbet > 70.0:
                # High c-bet player checking = very weak or trapping
                # Remove strong hands (would have bet)
                self._reduce_premium_hands(hand_range, factor=0.4)
                hand_range.scale(street_factor + 0.05)
            else:
                # Moderate: slight range retention
                hand_range.scale(street_factor + 0.10)
            return hand_range

        return hand_range

    # ------------------------------------------------------------------
    # Internal: premium hand adjustment helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reduce_premium_hands(hand_range: HandRange, factor: float) -> None:
        """Reduce weights of premium hands (top-left of matrix).

        Premium hands are high pairs (TT+) and big broadways (AK, AQ, etc.).
        Multiply their weights by *factor* (< 1.0 to reduce).

        Args:
            hand_range: Range to modify in place.
            factor: Multiplicative factor for premium cells.
        """
        # Premium region: ranks T(8) through A(12)
        premium_start = RANK_TO_INDEX[Rank.TEN]  # index 8
        for r in range(premium_start, _NUM_RANKS):
            for c in range(premium_start, _NUM_RANKS):
                hand_range.matrix[r, c] *= factor
        np.clip(hand_range.matrix, 0.0, 1.0, out=hand_range.matrix)

    @staticmethod
    def _boost_premium_hands(hand_range: HandRange, factor: float) -> None:
        """Boost weights of premium hands.

        Args:
            hand_range: Range to modify in place.
            factor: Multiplicative factor (> 1.0 to boost).
        """
        premium_start = RANK_TO_INDEX[Rank.TEN]
        for r in range(premium_start, _NUM_RANKS):
            for c in range(premium_start, _NUM_RANKS):
                hand_range.matrix[r, c] *= factor
        np.clip(hand_range.matrix, 0.0, 1.0, out=hand_range.matrix)
