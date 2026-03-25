"""Repository for hand history persistence.

Provides CRUD operations for hand records in SQLite,
following the repository pattern.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from src.engine.hand_history import HandRecord
from src.stats.connection_manager import ConnectionManager

logger = logging.getLogger(__name__)


class HandRepository:
    """Repository for storing and retrieving hand records.

    Args:
        connection_manager: Database connection manager.
    """

    def __init__(self, connection_manager: ConnectionManager) -> None:
        self._conn_mgr = connection_manager

    def save(self, record: HandRecord) -> None:
        """Save a hand record to the database.

        Args:
            record: The hand record to store.
        """
        conn = self._conn_mgr.get_connection()
        conn.execute(
            """
            INSERT OR REPLACE INTO hands
                (hand_id, players_json, actions_json, community_cards,
                 pot, winner_name, big_blind, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.hand_id,
                json.dumps(record.players),
                record.actions_json,
                record.community_cards_str,
                record.pot,
                record.winner_name,
                record.big_blind,
                record.timestamp,
            ),
        )
        conn.commit()
        logger.debug("Saved hand %s", record.hand_id)

    def get_by_id(self, hand_id: str) -> Optional[HandRecord]:
        """Retrieve a hand record by ID.

        Args:
            hand_id: The hand ID to look up.

        Returns:
            The HandRecord, or None if not found.
        """
        conn = self._conn_mgr.get_connection()
        row = conn.execute(
            "SELECT * FROM hands WHERE hand_id = ?", (hand_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_all(self) -> list[HandRecord]:
        """Retrieve all hand records.

        Returns:
            List of all stored HandRecord objects.
        """
        conn = self._conn_mgr.get_connection()
        rows = conn.execute(
            "SELECT * FROM hands ORDER BY timestamp"
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_by_player(self, player_name: str) -> list[HandRecord]:
        """Retrieve all hands involving a specific player.

        Args:
            player_name: Player name to filter by.

        Returns:
            List of HandRecord objects involving the player.
        """
        conn = self._conn_mgr.get_connection()
        # Use JSON search to find hands containing this player
        rows = conn.execute(
            "SELECT * FROM hands WHERE players_json LIKE ? ORDER BY timestamp",
            (f'%"{player_name}"%',),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def count(self) -> int:
        """Return the total number of stored hands.

        Returns:
            Number of hand records in the database.
        """
        conn = self._conn_mgr.get_connection()
        row = conn.execute("SELECT COUNT(*) FROM hands").fetchone()
        return row[0] if row else 0

    def delete(self, hand_id: str) -> bool:
        """Delete a hand record.

        Args:
            hand_id: The hand ID to delete.

        Returns:
            True if a record was deleted, False if not found.
        """
        conn = self._conn_mgr.get_connection()
        cursor = conn.execute(
            "DELETE FROM hands WHERE hand_id = ?", (hand_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_record(row: object) -> HandRecord:
        """Convert a database row to a HandRecord.

        Args:
            row: SQLite Row object.

        Returns:
            A HandRecord.
        """
        return HandRecord(
            hand_id=row["hand_id"],  # type: ignore[index]
            players=json.loads(row["players_json"]),  # type: ignore[index]
            actions_json=row["actions_json"],  # type: ignore[index]
            community_cards_str=row["community_cards"],  # type: ignore[index]
            pot=row["pot"],  # type: ignore[index]
            winner_name=row["winner_name"],  # type: ignore[index]
            big_blind=row["big_blind"],  # type: ignore[index]
            timestamp=row["timestamp"],  # type: ignore[index]
        )
