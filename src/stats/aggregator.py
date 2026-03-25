"""Real-time statistics aggregation service for poker HUD.

Maintains running counters for each observed player so that statistics
can be updated incrementally as hands complete, without recalculating
from scratch.

Usage:
    from src.stats.aggregator import StatsAggregator

    aggregator = StatsAggregator()
    aggregator.process_completed_hand(game_state)
    stats = aggregator.get_player_stats("Villain1")
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from src.common.logging import get_logger
from src.engine.game_state import ActionType, GameState, Player, PlayerAction, Street
from src.stats.calculations import (
    PlayerStats,
    _safe_divide,
    confidence_from_sample_size,
)

logger = get_logger("stats.aggregator")

# ---------------------------------------------------------------------------
# Running counters dataclass
# ---------------------------------------------------------------------------


@dataclass
class RunningStatCounters:
    """Per-player running counters for incremental stat updates.

    Each statistic tracks a numerator (count of events) and a denominator
    (count of opportunities) separately, so the percentage can be derived
    at O(1) cost.

    Attributes:
        total_hands: Total completed hands observed.
        total_profit: Cumulative chip profit/loss.
        vpip_count: Hands where player voluntarily put $ in preflop.
        vpip_opportunities: Total hands dealt to player.
        pfr_count: Hands where player raised preflop.
        pfr_opportunities: Total hands dealt to player.
        three_bet_count: Hands where player 3-bet preflop.
        three_bet_opportunities: Hands where player faced a preflop raise.
        fold_to_3bet_count: Times player folded to a 3-bet.
        fold_to_3bet_opportunities: Times player faced a 3-bet after raising.
        cbet_count: Times preflop raiser bet on flop.
        cbet_opportunities: Times preflop raiser had a chance to bet flop.
        aggressive_actions: Total bets + raises (all streets, non-blind).
        passive_actions: Total calls (all streets, non-blind).
        saw_flop_count: Hands where player saw the flop.
        wtsd_count: Hands where player went to showdown (after seeing flop).
        processed_hand_ids: Track which hands have been processed to avoid
            double-counting.
    """

    total_hands: int = 0
    total_profit: float = 0.0

    # VPIP
    vpip_count: int = 0
    vpip_opportunities: int = 0

    # PFR
    pfr_count: int = 0
    pfr_opportunities: int = 0

    # 3-Bet
    three_bet_count: int = 0
    three_bet_opportunities: int = 0

    # Fold to 3-Bet
    fold_to_3bet_count: int = 0
    fold_to_3bet_opportunities: int = 0

    # C-Bet
    cbet_count: int = 0
    cbet_opportunities: int = 0

    # Aggression Factor
    aggressive_actions: int = 0
    passive_actions: int = 0

    # WTSD
    saw_flop_count: int = 0
    wtsd_count: int = 0

    # Deduplication
    processed_hand_ids: Set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_preflop_actions(player: Player) -> list[PlayerAction]:
    """Return non-blind preflop actions for a player."""
    return [
        a
        for a in player.actions
        if a.street == Street.PREFLOP and a.action_type != ActionType.POST_BLIND
    ]


def _player_saw_flop(player: Player) -> bool:
    """Check if player had any post-preflop actions."""
    post_preflop_streets = {Street.FLOP, Street.TURN, Street.RIVER, Street.SHOWDOWN}
    return any(a.street in post_preflop_streets for a in player.actions)


def _player_went_to_showdown(player: Player, game_state: GameState) -> bool:
    """Return True if the player reached showdown."""
    # Player has showdown actions
    if any(a.street == Street.SHOWDOWN for a in player.actions):
        return True
    # Player was active at end of hand and hand reached showdown
    if game_state.current_street == Street.SHOWDOWN and player.is_active:
        return True
    # Player was active on river and didn't fold
    river_actions = [a for a in player.actions if a.street == Street.RIVER]
    if river_actions and not any(
        a.action_type == ActionType.FOLD for a in river_actions
    ):
        # If the hand went to showdown (or river completed), player was there
        if game_state.current_street in (Street.RIVER, Street.SHOWDOWN):
            return player.is_active
    return False


def _count_raises_before_player(
    action_history: list[PlayerAction],
    player_name: str,
) -> tuple[int, bool, bool]:
    """Analyze preflop action history for 3-bet detection.

    Walks through the global preflop action history (excluding blinds) and
    determines:
    - How many raises occurred before the player's first voluntary action.
    - Whether the player raised after facing a raise (3-bet).
    - Whether the player folded after raising (fold to 3-bet scenario).

    Args:
        action_history: The global action_history from GameState.
        player_name: Name of the player to analyze.

    Returns:
        Tuple of (raises_before, player_3bet, unused):
            - raises_before: Number of raises seen before this player's first
              voluntary action.
            - player_3bet: True if the player re-raised after facing a raise.
            - (not used directly - see the main update method for fold-to-3bet logic)
    """
    preflop_actions = [
        a
        for a in action_history
        if a.street == Street.PREFLOP and a.action_type != ActionType.POST_BLIND
    ]

    raises_before = 0
    player_3bet = False

    for a in preflop_actions:
        if a.player_name == player_name:
            if raises_before >= 1 and a.action_type in (
                ActionType.RAISE,
                ActionType.ALL_IN,
            ):
                player_3bet = True
            break
        if a.action_type in (ActionType.RAISE, ActionType.BET, ActionType.ALL_IN):
            raises_before += 1

    return raises_before, player_3bet, False


# ---------------------------------------------------------------------------
# StatsAggregator
# ---------------------------------------------------------------------------


class StatsAggregator:
    """Maintains and incrementally updates player statistics.

    Thread-safe for concurrent reads via a ``threading.Lock``.  Stat
    lookups are O(1) dict access; hand processing is O(players * actions).

    Args:
        persistence_callback: Optional callable invoked after each hand
            with the full counters dict for external persistence.
            Signature: ``(counters: Dict[str, RunningStatCounters]) -> None``.
    """

    def __init__(
        self,
        persistence_callback: Optional[
            Callable[[Dict[str, RunningStatCounters]], None]
        ] = None,
    ) -> None:
        self._counters: Dict[str, RunningStatCounters] = {}
        self._lock = threading.Lock()
        self._persistence_callback = persistence_callback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_completed_hand(self, game_state: GameState) -> None:
        """Extract actions from a completed hand and update running stats.

        Idempotent: if the same ``hand_id`` is processed twice for a given
        player, the second call is a no-op for that player.

        Args:
            game_state: The completed hand's game state.
        """
        hand_id = game_state.hand_id

        with self._lock:
            for player in game_state.players:
                self._update_player(player, game_state, hand_id)

        if self._persistence_callback is not None:
            try:
                self._persistence_callback(self._counters.copy())
            except Exception:
                logger.warning(
                    "persistence_callback_failed",
                    hand_id=hand_id,
                    exc_info=True,
                )

    def get_player_stats(self, player_name: str) -> PlayerStats:
        """Compute current stats for a single player from running counters.

        Args:
            player_name: The player to look up.

        Returns:
            A ``PlayerStats`` instance.  Returns zeroed stats if the
            player has not been observed.
        """
        with self._lock:
            counters = self._counters.get(player_name)
            if counters is None:
                return PlayerStats()
            return self._counters_to_stats(counters)

    def get_all_stats(self) -> Dict[str, PlayerStats]:
        """Return computed stats for every observed player.

        Returns:
            Dict mapping player name to ``PlayerStats``.
        """
        with self._lock:
            return {
                name: self._counters_to_stats(c)
                for name, c in self._counters.items()
            }

    def get_table_stats(self, seat_names: List[str]) -> Dict[str, PlayerStats]:
        """Return stats for only the named players (current table occupants).

        Args:
            seat_names: List of player names at the current table.

        Returns:
            Dict mapping player name to ``PlayerStats`` for each name
            that has recorded data.
        """
        with self._lock:
            result: Dict[str, PlayerStats] = {}
            for name in seat_names:
                counters = self._counters.get(name)
                if counters is not None:
                    result[name] = self._counters_to_stats(counters)
                else:
                    result[name] = PlayerStats()
            return result

    def get_counters(self, player_name: str) -> Optional[RunningStatCounters]:
        """Return raw running counters for a player, or None.

        Useful for testing and debugging.

        Args:
            player_name: The player to look up.

        Returns:
            The ``RunningStatCounters`` instance, or None.
        """
        with self._lock:
            return self._counters.get(player_name)

    # ------------------------------------------------------------------
    # Internal update logic
    # ------------------------------------------------------------------

    def _ensure_counters(self, player_name: str) -> RunningStatCounters:
        """Get or create running counters for a player.  Caller holds lock."""
        if player_name not in self._counters:
            self._counters[player_name] = RunningStatCounters()
        return self._counters[player_name]

    def _update_player(
        self,
        player: Player,
        game_state: GameState,
        hand_id: str,
    ) -> None:
        """Update all running counters for one player.  Caller holds lock."""
        counters = self._ensure_counters(player.name)

        # Idempotency check
        if hand_id in counters.processed_hand_ids:
            return
        counters.processed_hand_ids.add(hand_id)
        counters.total_hands += 1

        preflop_actions = _extract_preflop_actions(player)
        all_non_blind = [
            a for a in player.actions if a.action_type != ActionType.POST_BLIND
        ]

        # --- VPIP ---
        counters.vpip_opportunities += 1
        if any(
            a.action_type
            in (ActionType.CALL, ActionType.RAISE, ActionType.BET, ActionType.ALL_IN)
            for a in preflop_actions
        ):
            counters.vpip_count += 1

        # --- PFR ---
        counters.pfr_opportunities += 1
        player_raised_preflop = any(
            a.action_type in (ActionType.RAISE, ActionType.BET, ActionType.ALL_IN)
            for a in preflop_actions
        )
        if player_raised_preflop:
            counters.pfr_count += 1

        # --- 3-Bet and Fold-to-3-Bet ---
        self._update_three_bet_stats(
            counters, player, game_state, player_raised_preflop
        )

        # --- C-Bet ---
        self._update_cbet_stats(counters, player, player_raised_preflop)

        # --- Aggression Factor ---
        for a in all_non_blind:
            if a.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
                counters.aggressive_actions += 1
            elif a.action_type == ActionType.CALL:
                counters.passive_actions += 1

        # --- WTSD ---
        saw_flop = _player_saw_flop(player)
        if saw_flop:
            counters.saw_flop_count += 1
            if _player_went_to_showdown(player, game_state):
                counters.wtsd_count += 1

    def _update_three_bet_stats(
        self,
        counters: RunningStatCounters,
        player: Player,
        game_state: GameState,
        player_raised_preflop: bool,
    ) -> None:
        """Update 3-bet and fold-to-3-bet counters.  Caller holds lock."""
        preflop_history = [
            a
            for a in game_state.action_history
            if a.street == Street.PREFLOP and a.action_type != ActionType.POST_BLIND
        ]
        if not preflop_history:
            return

        # Walk the preflop action sequence to find raise/re-raise patterns.
        raise_count = 0
        player_first_action_seen = False

        for a in preflop_history:
            if a.player_name == player.name and not player_first_action_seen:
                player_first_action_seen = True
                if raise_count >= 1:
                    # Player faced a raise -- 3-bet opportunity
                    counters.three_bet_opportunities += 1
                    if a.action_type in (
                        ActionType.RAISE,
                        ActionType.ALL_IN,
                    ):
                        counters.three_bet_count += 1
                break

            if a.action_type in (ActionType.RAISE, ActionType.BET, ActionType.ALL_IN):
                raise_count += 1

        # Fold to 3-bet: player raised, then someone re-raised, then player folded
        if not player_raised_preflop:
            return

        player_raise_seen = False
        opponent_reraise_seen = False
        for a in preflop_history:
            if a.player_name == player.name:
                if not player_raise_seen and a.action_type in (
                    ActionType.RAISE,
                    ActionType.BET,
                    ActionType.ALL_IN,
                ):
                    player_raise_seen = True
                    continue
                if player_raise_seen and opponent_reraise_seen:
                    # Player's response to the 3-bet
                    counters.fold_to_3bet_opportunities += 1
                    if a.action_type == ActionType.FOLD:
                        counters.fold_to_3bet_count += 1
                    break
            elif player_raise_seen and a.action_type in (
                ActionType.RAISE,
                ActionType.ALL_IN,
            ):
                opponent_reraise_seen = True

    def _update_cbet_stats(
        self,
        counters: RunningStatCounters,
        player: Player,
        player_raised_preflop: bool,
    ) -> None:
        """Update continuation bet counters.  Caller holds lock."""
        if not player_raised_preflop:
            return

        flop_actions = [a for a in player.actions if a.street == Street.FLOP]
        if not flop_actions:
            return  # Didn't see the flop or hand ended preflop

        counters.cbet_opportunities += 1
        if any(
            a.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)
            for a in flop_actions
        ):
            counters.cbet_count += 1

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _counters_to_stats(counters: RunningStatCounters) -> PlayerStats:
        """Convert running counters into a ``PlayerStats`` snapshot."""
        if counters.passive_actions == 0:
            af = (
                float("inf")
                if counters.aggressive_actions > 0
                else 0.0
            )
        else:
            af = counters.aggressive_actions / counters.passive_actions

        return PlayerStats(
            vpip=_safe_divide(counters.vpip_count, counters.vpip_opportunities) * 100.0,
            pfr=_safe_divide(counters.pfr_count, counters.pfr_opportunities) * 100.0,
            three_bet_pct=_safe_divide(
                counters.three_bet_count, counters.three_bet_opportunities
            )
            * 100.0,
            fold_to_three_bet=_safe_divide(
                counters.fold_to_3bet_count, counters.fold_to_3bet_opportunities
            )
            * 100.0,
            cbet_pct=_safe_divide(counters.cbet_count, counters.cbet_opportunities)
            * 100.0,
            aggression_factor=af,
            wtsd=_safe_divide(counters.wtsd_count, counters.saw_flop_count) * 100.0,
            total_hands=counters.total_hands,
            total_profit=counters.total_profit,
            confidence=confidence_from_sample_size(counters.total_hands),
        )
