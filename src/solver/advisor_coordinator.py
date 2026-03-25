"""Strategy advisor coordinator with async computation.

Orchestrates the solver pipeline: range estimation, equity calculation,
and action recommendation.  Results are cached per game state and equity
calculations run in a background thread so the main loop is never blocked.

Usage:
    from src.solver.advisor_coordinator import StrategyAdvisorCoordinator

    advisor = StrategyAdvisorCoordinator()
    advisor.on_advice_ready(lambda advice: overlay.update(advice))
    advisor.get_advice_async(game_state, player_stats)
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from src.common.logging import get_logger
from src.common.performance import LatencyTracker, PerfTimer
from src.detection.card import Card, Rank
from src.engine.game_state import ActionType, GameState, Player, Street
from src.solver.range_estimator import HandRange, RangeEstimator
from src.stats.calculations import PlayerStats

logger = get_logger("solver.advisor_coordinator")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ActionRecommendation(Enum):
    """Recommended action for the hero."""

    FOLD = auto()
    CALL = auto()
    RAISE = auto()
    CHECK = auto()
    ALL_IN = auto()


@dataclass
class StrategyAdvice:
    """Complete strategy recommendation from the solver.

    Attributes:
        preflop_range_position: Whether hero's hand falls within the
            recommended opening/continuing range for their position.
        equity: Estimated equity of hero's hand vs opponent ranges (0.0-1.0).
        recommendation: Suggested action.
        recommended_sizing: Suggested bet/raise size as fraction of pot
            (e.g. 0.75 = 75% pot).  None when action is fold/check.
        opponent_ranges: Estimated ranges per opponent name.
        reasoning: List of human-readable explanation strings.
    """

    preflop_range_position: bool = False
    equity: float = 0.0
    recommendation: ActionRecommendation = ActionRecommendation.FOLD
    recommended_sizing: Optional[float] = None
    opponent_ranges: dict[str, HandRange] = field(default_factory=dict)
    reasoning: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------

_MAX_CACHE_SIZE = 64


def _game_state_key(game_state: GameState) -> str:
    """Produce a hashable cache key for a game state snapshot.

    Includes hand_id, street, pot, community cards, and action count so
    that any state change invalidates the cache entry.

    Args:
        game_state: Current game state.

    Returns:
        A hex digest string.
    """
    parts = [
        game_state.hand_id,
        game_state.current_street.name,
        f"{game_state.pot_size:.2f}",
        str(len(game_state.action_history)),
        ":".join(str(c) for c in game_state.community_cards),
    ]
    raw = "|".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Simple equity estimator (no Monte Carlo yet -- heuristic-based)
# ---------------------------------------------------------------------------


def _rank_value(rank: Rank) -> int:
    """Return numeric value for a rank (2=2 .. A=14)."""
    values = {
        Rank.TWO: 2, Rank.THREE: 3, Rank.FOUR: 4, Rank.FIVE: 5,
        Rank.SIX: 6, Rank.SEVEN: 7, Rank.EIGHT: 8, Rank.NINE: 9,
        Rank.TEN: 10, Rank.JACK: 11, Rank.QUEEN: 12, Rank.KING: 13,
        Rank.ACE: 14,
    }
    return values[rank]


def _estimate_hand_equity(
    hole_cards: list[Card],
    board: list[Card],
    opponent_range: HandRange,
) -> float:
    """Heuristic equity estimate for hero's hand vs an opponent range.

    This is a fast approximation -- not a full Monte Carlo simulation.
    It considers hand strength relative to the opponent's range width.

    Args:
        hole_cards: Hero's two hole cards.
        board: Community cards.
        opponent_range: Estimated opponent range.

    Returns:
        Equity as a float in [0.0, 1.0].
    """
    if len(hole_cards) < 2:
        return 0.5  # No information

    card1, card2 = hole_cards[0], hole_cards[1]
    hi_val = max(_rank_value(card1.rank), _rank_value(card2.rank))
    lo_val = min(_rank_value(card1.rank), _rank_value(card2.rank))
    is_pair = card1.rank == card2.rank
    is_suited = card1.suit == card2.suit

    # Base equity from hand strength heuristic
    if is_pair:
        # Pairs: stronger pairs get higher equity
        base = 0.50 + (hi_val - 2) * 0.03  # 22=0.50, AA=0.86
    else:
        # Non-pairs: high card value + connectedness bonus
        base = 0.30 + (hi_val - 2) * 0.02 + (lo_val - 2) * 0.01
        gap = hi_val - lo_val
        if gap <= 2:
            base += 0.05  # Connected
        if is_suited:
            base += 0.04  # Suited bonus

    # Adjust based on opponent range width: narrower range = stronger opponent
    range_pct = opponent_range.range_pct()
    if range_pct < 10.0:
        # Very tight opponent: reduce our equity
        base *= 0.75
    elif range_pct < 20.0:
        base *= 0.85
    elif range_pct > 50.0:
        # Very wide opponent: increase our equity
        base *= 1.10

    # Board texture adjustment (simple)
    if board:
        board_ranks = {_rank_value(c.rank) for c in board}
        # If our high card hits the board, boost equity
        if hi_val in board_ranks:
            base += 0.10
        if lo_val in board_ranks:
            base += 0.07
        # Pair on board: slight equity reduction (opponent could have it too)
        if len(board_ranks) < len(board):
            base -= 0.03

    return max(0.0, min(1.0, base))


# ---------------------------------------------------------------------------
# StrategyAdvisorCoordinator
# ---------------------------------------------------------------------------


class StrategyAdvisorCoordinator:
    """Main entry point for solver recommendations.

    Orchestrates range estimation, equity calculation, and action
    recommendation.  Supports both synchronous and asynchronous modes.

    Args:
        range_estimator: RangeEstimator instance. Created if not provided.
        max_workers: Thread pool size for async equity calculations.
    """

    def __init__(
        self,
        range_estimator: RangeEstimator | None = None,
        max_workers: int = 1,
    ) -> None:
        self._range_estimator = range_estimator or RangeEstimator()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cache: OrderedDict[str, StrategyAdvice] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._callbacks: list[Callable[[StrategyAdvice], None]] = []

        # Performance tracking
        self._latency_tracker = LatencyTracker("advisor_full_pipeline")
        self._range_tracker = LatencyTracker("advisor_range_estimation")
        self._equity_tracker = LatencyTracker("advisor_equity_calculation")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_advice_ready(self, callback: Callable[[StrategyAdvice], None]) -> None:
        """Register a callback invoked when async advice is ready.

        Args:
            callback: Function receiving a ``StrategyAdvice`` instance.
        """
        self._callbacks.append(callback)

    def get_advice(
        self,
        game_state: GameState,
        player_stats: dict[str, PlayerStats],
    ) -> StrategyAdvice:
        """Synchronously compute strategy advice.

        Args:
            game_state: Current game state snapshot.
            player_stats: Stats keyed by player name.

        Returns:
            Complete strategy advice.
        """
        cache_key = _game_state_key(game_state)

        with self._cache_lock:
            if cache_key in self._cache:
                logger.debug("advice_cache_hit", key=cache_key)
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key]

        with PerfTimer("advisor_full", tracker=self._latency_tracker):
            advice = self._compute_advice(game_state, player_stats)

        with self._cache_lock:
            self._cache[cache_key] = advice
            # Evict oldest if over capacity
            while len(self._cache) > _MAX_CACHE_SIZE:
                self._cache.popitem(last=False)

        return advice

    def get_advice_async(
        self,
        game_state: GameState,
        player_stats: dict[str, PlayerStats],
    ) -> Future[StrategyAdvice]:
        """Submit advice computation to background thread.

        When complete, all registered ``on_advice_ready`` callbacks are
        invoked with the result.

        Args:
            game_state: Current game state snapshot.
            player_stats: Stats keyed by player name.

        Returns:
            A ``Future`` that resolves to ``StrategyAdvice``.
        """
        future = self._executor.submit(
            self._compute_and_notify, game_state, player_stats
        )
        return future

    def get_latency_summary(self) -> dict[str, float]:
        """Return average latencies for each pipeline stage.

        Returns:
            Dict mapping stage name to average latency in ms.
        """
        result: dict[str, float] = {}
        for tracker in (
            self._latency_tracker,
            self._range_tracker,
            self._equity_tracker,
        ):
            try:
                summary = tracker.summary()
                result[summary.name] = summary.avg_ms
            except ValueError:
                pass  # No measurements yet
        return result

    def clear_cache(self) -> None:
        """Clear the advice cache."""
        with self._cache_lock:
            self._cache.clear()

    def shutdown(self) -> None:
        """Shut down the background thread pool."""
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Internal: computation pipeline
    # ------------------------------------------------------------------

    def _compute_and_notify(
        self,
        game_state: GameState,
        player_stats: dict[str, PlayerStats],
    ) -> StrategyAdvice:
        """Compute advice and fire callbacks. Runs in worker thread."""
        advice = self.get_advice(game_state, player_stats)
        for cb in self._callbacks:
            try:
                cb(advice)
            except Exception:
                logger.warning("advice_callback_failed", exc_info=True)
        return advice

    def _compute_advice(
        self,
        game_state: GameState,
        player_stats: dict[str, PlayerStats],
    ) -> StrategyAdvice:
        """Full advice computation pipeline.

        Steps:
        1. Estimate opponent ranges
        2. Calculate hero equity
        3. Determine recommendation

        Args:
            game_state: Current game state.
            player_stats: Stats per player name.

        Returns:
            Computed StrategyAdvice.
        """
        reasoning: list[str] = []
        hero = game_state.get_hero()

        if hero is None:
            return StrategyAdvice(reasoning=["Hero not found in game state."])

        # Step 1: Estimate opponent ranges
        opponent_ranges: dict[str, HandRange] = {}
        with PerfTimer("range_estimation", tracker=self._range_tracker):
            for player in game_state.get_active_players():
                if player.seat_number == game_state.hero_seat:
                    continue
                stats = player_stats.get(player.name, PlayerStats())
                player_actions = [
                    a for a in player.actions
                    if a.action_type != ActionType.POST_BLIND
                ]
                opponent_range = self._range_estimator.estimate_range(
                    stats, player_actions, game_state.community_cards
                )
                opponent_ranges[player.name] = opponent_range
                pct = opponent_range.range_pct()
                reasoning.append(
                    f"{player.name}: estimated range ~{pct:.0f}% "
                    f"(VPIP={stats.vpip:.0f}, PFR={stats.pfr:.0f})"
                )

        # Step 2: Calculate equity
        equity = 0.5  # Default
        hole_cards = hero.hole_cards or []
        if hole_cards and opponent_ranges:
            with PerfTimer("equity_calc", tracker=self._equity_tracker):
                # Average equity across all opponents
                equities = []
                for opp_range in opponent_ranges.values():
                    eq = _estimate_hand_equity(
                        hole_cards, game_state.community_cards, opp_range
                    )
                    equities.append(eq)
                equity = sum(equities) / len(equities)
            reasoning.append(f"Estimated equity: {equity:.1%}")
        elif not hole_cards:
            reasoning.append("Hero hole cards unknown; equity defaulted to 50%.")

        # Step 3: Preflop range check
        preflop_in_range = self._check_preflop_range(
            hero, game_state.current_street
        )
        if game_state.current_street == Street.PREFLOP:
            if preflop_in_range:
                reasoning.append("Hero hand is within recommended preflop range.")
            else:
                reasoning.append("Hero hand is OUTSIDE recommended preflop range.")

        # Step 4: Action recommendation
        recommendation, sizing = self._recommend_action(
            equity, game_state, hero, preflop_in_range
        )
        reasoning.append(f"Recommendation: {recommendation.name}")
        if sizing is not None:
            reasoning.append(f"Suggested sizing: {sizing:.0%} of pot")

        return StrategyAdvice(
            preflop_range_position=preflop_in_range,
            equity=equity,
            recommendation=recommendation,
            recommended_sizing=sizing,
            opponent_ranges=opponent_ranges,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Internal: preflop range check
    # ------------------------------------------------------------------

    @staticmethod
    def _check_preflop_range(
        hero: Player,
        street: Street,
    ) -> bool:
        """Check whether hero's hand is in a standard preflop opening range.

        Uses a simple heuristic: top ~25% of hands are 'in range'.

        Args:
            hero: The hero player.
            street: Current street.

        Returns:
            True if the hand is in the recommended range.
        """
        if street != Street.PREFLOP:
            return True  # Only relevant preflop

        hole_cards = hero.hole_cards
        if not hole_cards or len(hole_cards) < 2:
            return False

        c1, c2 = hole_cards[0], hole_cards[1]
        v1 = _rank_value(c1.rank)
        v2 = _rank_value(c2.rank)
        hi, lo = max(v1, v2), min(v1, v2)
        is_pair = c1.rank == c2.rank
        is_suited = c1.suit == c2.suit

        # Simple threshold: pairs 55+, suited broadways, ATo+, KQo
        if is_pair and hi >= 5:
            return True
        if is_suited and hi >= 10 and lo >= 5:
            return True
        if not is_suited and hi >= 10 and lo >= 10:
            return True
        # A-x suited for A2s+
        if is_suited and hi == 14:
            return True

        return False

    # ------------------------------------------------------------------
    # Internal: action recommendation
    # ------------------------------------------------------------------

    @staticmethod
    def _recommend_action(
        equity: float,
        game_state: GameState,
        hero: Player,
        preflop_in_range: bool,
    ) -> tuple[ActionRecommendation, Optional[float]]:
        """Determine recommended action and sizing from equity.

        Uses pot-odds comparison: if equity exceeds the price of calling,
        continue.  Strong equity warrants raising.

        Args:
            equity: Hero's estimated equity (0.0-1.0).
            game_state: Current game state.
            hero: Hero player.
            preflop_in_range: Whether hero is in preflop range.

        Returns:
            Tuple of (recommendation, sizing_fraction_or_None).
        """
        pot = game_state.get_current_pot()
        street = game_state.current_street

        # Preflop: lean on range chart
        if street == Street.PREFLOP and not preflop_in_range:
            return ActionRecommendation.FOLD, None

        # Equity-based thresholds
        if equity >= 0.70:
            # Strong: raise / bet
            sizing = 0.75 if pot > 0 else None
            return ActionRecommendation.RAISE, sizing

        if equity >= 0.50:
            # Decent: call or small bet
            # If no bet to call, bet
            has_bet_to_call = hero.current_bet < max(
                (p.current_bet for p in game_state.players), default=0.0
            )
            if has_bet_to_call:
                return ActionRecommendation.CALL, None
            return ActionRecommendation.RAISE, 0.50

        if equity >= 0.35:
            # Marginal: check or call small bets
            has_bet_to_call = hero.current_bet < max(
                (p.current_bet for p in game_state.players), default=0.0
            )
            if has_bet_to_call and pot > 0:
                # Pot odds check
                amount_to_call = max(
                    (p.current_bet for p in game_state.players), default=0.0
                ) - hero.current_bet
                total = pot + amount_to_call
                pot_odds = (
                    amount_to_call / total if total > 0 else 1.0
                )
                if equity >= pot_odds:
                    return ActionRecommendation.CALL, None
                return ActionRecommendation.FOLD, None
            return ActionRecommendation.CHECK, None

        # Weak equity: fold (or check if free)
        has_bet_to_call = hero.current_bet < max(
            (p.current_bet for p in game_state.players), default=0.0
        )
        if not has_bet_to_call:
            return ActionRecommendation.CHECK, None
        return ActionRecommendation.FOLD, None
