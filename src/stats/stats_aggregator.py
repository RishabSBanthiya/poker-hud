"""Real-time stats aggregation service.

Processes completed hand records and updates per-player statistics.
Provides formatted stat strings for the HUD overlay.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Optional

from src.engine.hand_history import HandRecord
from src.stats.hand_repository import HandRepository
from src.stats.player_stats_repository import PlayerStats, PlayerStatsRepository

logger = logging.getLogger(__name__)


class StatsAggregator:
    """Aggregates and maintains player statistics from hand records.

    Processes completed hands, updates stats in the database, and
    provides formatted output for the overlay.

    Args:
        hand_repo: Repository for hand record storage.
        stats_repo: Repository for player stats storage.
        min_hands: Minimum hands before displaying stats.
    """

    def __init__(
        self,
        hand_repo: HandRepository,
        stats_repo: PlayerStatsRepository,
        min_hands: int = 5,
    ) -> None:
        self._hand_repo = hand_repo
        self._stats_repo = stats_repo
        self._min_hands = min_hands
        self._on_stats_update: Optional[
            Callable[[dict[str, PlayerStats]], None]
        ] = None

    def set_stats_update_callback(
        self, callback: Callable[[dict[str, PlayerStats]], None]
    ) -> None:
        """Set callback for stats updates.

        Args:
            callback: Function receiving a dict of player_name -> PlayerStats.
        """
        self._on_stats_update = callback

    def process_hand(self, record: HandRecord) -> None:
        """Process a completed hand record and update statistics.

        Args:
            record: The completed hand record.
        """
        self._hand_repo.save(record)

        actions = json.loads(record.actions_json)

        for player_name in record.players:
            stats = self._stats_repo.get(player_name) or PlayerStats(
                player_name=player_name
            )
            stats.total_hands += 1

            player_actions = [
                a for a in actions if a["player"] == player_name
            ]

            self._update_vpip(stats, player_actions)
            self._update_pfr(stats, player_actions)
            self._update_action_counts(stats, player_actions)
            self._update_showdown(stats, player_name, record, actions)

            self._stats_repo.save(stats)

        # Notify subscribers
        if self._on_stats_update is not None:
            all_stats = {
                s.player_name: s for s in self._stats_repo.get_all()
            }
            self._on_stats_update(all_stats)

        logger.debug(
            "Processed hand %s for %d players",
            record.hand_id,
            len(record.players),
        )

    def get_player_stats(self, player_name: str) -> Optional[PlayerStats]:
        """Retrieve current stats for a player.

        Args:
            player_name: Player name to look up.

        Returns:
            PlayerStats if available, None otherwise.
        """
        return self._stats_repo.get(player_name)

    def get_all_stats(self) -> dict[str, PlayerStats]:
        """Retrieve stats for all known players.

        Returns:
            Dictionary mapping player names to their stats.
        """
        return {s.player_name: s for s in self._stats_repo.get_all()}

    def format_player_hud(self, player_name: str) -> str:
        """Format a player's stats as a HUD display string.

        Args:
            player_name: Player name.

        Returns:
            Formatted string like "VPIP:24/PFR:18/3B:8/AF:2.1"
        """
        stats = self._stats_repo.get(player_name)
        if stats is None or stats.total_hands < self._min_hands:
            return f"{player_name}: ({stats.total_hands if stats else 0} hands)"

        return (
            f"{player_name}: "
            f"VPIP:{stats.vpip:.0f} "
            f"PFR:{stats.pfr:.0f} "
            f"3B:{stats.three_bet_pct:.0f} "
            f"AF:{stats.aggression_factor:.1f} "
            f"({stats.total_hands}h)"
        )

    @staticmethod
    def _update_vpip(
        stats: PlayerStats, player_actions: list[dict]
    ) -> None:
        """Update VPIP from preflop actions."""
        preflop_actions = [
            a for a in player_actions if a["street"] == "preflop"
        ]
        voluntary = any(
            a["action"] in ("call", "raise", "bet", "all_in")
            for a in preflop_actions
        )
        if voluntary:
            stats.vpip_hands += 1

    @staticmethod
    def _update_pfr(
        stats: PlayerStats, player_actions: list[dict]
    ) -> None:
        """Update PFR from preflop raise actions."""
        preflop_raises = [
            a
            for a in player_actions
            if a["street"] == "preflop" and a["action"] in ("raise", "bet")
        ]
        if preflop_raises:
            stats.pfr_hands += 1

    @staticmethod
    def _update_action_counts(
        stats: PlayerStats, player_actions: list[dict]
    ) -> None:
        """Update aggregate action counters."""
        for action in player_actions:
            act = action["action"]
            if act == "bet":
                stats.total_bets += 1
            elif act == "raise":
                stats.total_raises += 1
            elif act == "call":
                stats.total_calls += 1
            elif act == "fold":
                stats.total_folds += 1

    @staticmethod
    def _update_showdown(
        stats: PlayerStats,
        player_name: str,
        record: HandRecord,
        actions: list[dict],
    ) -> None:
        """Update showdown statistics."""
        player_actions = [a for a in actions if a["player"] == player_name]
        folded = any(a["action"] == "fold" for a in player_actions)

        # If player had postflop action, count as showdown opportunity
        postflop_actions = [
            a for a in player_actions if a["street"] != "preflop"
        ]
        if postflop_actions:
            stats.showdown_opportunities += 1
            if not folded:
                stats.went_to_showdown += 1
