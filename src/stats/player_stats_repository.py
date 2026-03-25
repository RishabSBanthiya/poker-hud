"""Repository for player statistics persistence.

Provides CRUD operations for per-player aggregated statistics
in SQLite, following the repository pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.stats.connection_manager import ConnectionManager

logger = logging.getLogger(__name__)


@dataclass
class PlayerStats:
    """Aggregated statistics for a single player.

    Attributes:
        player_name: Player identifier.
        total_hands: Total hands observed.
        vpip_hands: Hands where player voluntarily put money in pot.
        pfr_hands: Hands where player raised preflop.
        three_bet_opportunities: Opportunities to 3-bet.
        three_bet_hands: Times player actually 3-bet.
        cbet_opportunities: Opportunities to continuation bet.
        cbet_hands: Times player actually c-bet.
        total_bets: Total bet actions.
        total_raises: Total raise actions.
        total_calls: Total call actions.
        total_folds: Total fold actions.
        went_to_showdown: Times player went to showdown.
        showdown_opportunities: Opportunities to go to showdown.
    """

    player_name: str
    total_hands: int = 0
    vpip_hands: int = 0
    pfr_hands: int = 0
    three_bet_opportunities: int = 0
    three_bet_hands: int = 0
    cbet_opportunities: int = 0
    cbet_hands: int = 0
    total_bets: int = 0
    total_raises: int = 0
    total_calls: int = 0
    total_folds: int = 0
    went_to_showdown: int = 0
    showdown_opportunities: int = 0

    @property
    def vpip(self) -> float:
        """Voluntarily Put money In Pot percentage."""
        if self.total_hands == 0:
            return 0.0
        return (self.vpip_hands / self.total_hands) * 100

    @property
    def pfr(self) -> float:
        """Pre-Flop Raise percentage."""
        if self.total_hands == 0:
            return 0.0
        return (self.pfr_hands / self.total_hands) * 100

    @property
    def three_bet_pct(self) -> float:
        """3-Bet percentage."""
        if self.three_bet_opportunities == 0:
            return 0.0
        return (self.three_bet_hands / self.three_bet_opportunities) * 100

    @property
    def cbet_pct(self) -> float:
        """Continuation bet percentage."""
        if self.cbet_opportunities == 0:
            return 0.0
        return (self.cbet_hands / self.cbet_opportunities) * 100

    @property
    def aggression_factor(self) -> float:
        """Aggression Factor = (bets + raises) / calls."""
        if self.total_calls == 0:
            return float(self.total_bets + self.total_raises)
        return (self.total_bets + self.total_raises) / self.total_calls

    @property
    def wtsd(self) -> float:
        """Went To ShowDown percentage."""
        if self.showdown_opportunities == 0:
            return 0.0
        return (self.went_to_showdown / self.showdown_opportunities) * 100


class PlayerStatsRepository:
    """Repository for player statistics.

    Args:
        connection_manager: Database connection manager.
    """

    def __init__(self, connection_manager: ConnectionManager) -> None:
        self._conn_mgr = connection_manager

    def save(self, stats: PlayerStats) -> None:
        """Save or update player statistics.

        Args:
            stats: The player statistics to store.
        """
        conn = self._conn_mgr.get_connection()
        conn.execute(
            """
            INSERT OR REPLACE INTO player_stats
                (player_name, total_hands, vpip_hands, pfr_hands,
                 three_bet_opportunities, three_bet_hands,
                 cbet_opportunities, cbet_hands,
                 total_bets, total_raises, total_calls, total_folds,
                 went_to_showdown, showdown_opportunities)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stats.player_name,
                stats.total_hands,
                stats.vpip_hands,
                stats.pfr_hands,
                stats.three_bet_opportunities,
                stats.three_bet_hands,
                stats.cbet_opportunities,
                stats.cbet_hands,
                stats.total_bets,
                stats.total_raises,
                stats.total_calls,
                stats.total_folds,
                stats.went_to_showdown,
                stats.showdown_opportunities,
            ),
        )
        conn.commit()

    def get(self, player_name: str) -> Optional[PlayerStats]:
        """Retrieve statistics for a player.

        Args:
            player_name: Player name to look up.

        Returns:
            PlayerStats if found, None otherwise.
        """
        conn = self._conn_mgr.get_connection()
        row = conn.execute(
            "SELECT * FROM player_stats WHERE player_name = ?",
            (player_name,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_stats(row)

    def get_all(self) -> list[PlayerStats]:
        """Retrieve all player statistics.

        Returns:
            List of all PlayerStats.
        """
        conn = self._conn_mgr.get_connection()
        rows = conn.execute(
            "SELECT * FROM player_stats ORDER BY player_name"
        ).fetchall()
        return [self._row_to_stats(row) for row in rows]

    @staticmethod
    def _row_to_stats(row: object) -> PlayerStats:
        """Convert a database row to PlayerStats.

        Args:
            row: SQLite Row object.

        Returns:
            A PlayerStats.
        """
        return PlayerStats(
            player_name=row["player_name"],  # type: ignore[index]
            total_hands=row["total_hands"],  # type: ignore[index]
            vpip_hands=row["vpip_hands"],  # type: ignore[index]
            pfr_hands=row["pfr_hands"],  # type: ignore[index]
            three_bet_opportunities=row["three_bet_opportunities"],  # type: ignore[index]
            three_bet_hands=row["three_bet_hands"],  # type: ignore[index]
            cbet_opportunities=row["cbet_opportunities"],  # type: ignore[index]
            cbet_hands=row["cbet_hands"],  # type: ignore[index]
            total_bets=row["total_bets"],  # type: ignore[index]
            total_raises=row["total_raises"],  # type: ignore[index]
            total_calls=row["total_calls"],  # type: ignore[index]
            total_folds=row["total_folds"],  # type: ignore[index]
            went_to_showdown=row["went_to_showdown"],  # type: ignore[index]
            showdown_opportunities=row["showdown_opportunities"],  # type: ignore[index]
        )
