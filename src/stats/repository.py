"""Database access layer (repository pattern) for the stats subsystem.

Provides ``PlayerRepository``, ``HandRepository``, and ``ActionRepository``
for persisting and querying poker hand data.  All repositories accept a
``sqlite3.Connection`` and use parameterized queries exclusively.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Optional

from src.common.logging import get_logger
from src.engine.game_state import ActionType, GameState, Street

logger = get_logger("stats.repository")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cards_to_json(cards: list) -> str:
    """Serialize a list of Card objects to a JSON string."""
    return json.dumps([str(c) for c in cards])


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    """Convert a sqlite3.Row to a plain dict, or return None."""
    if row is None:
        return None
    return dict(row)


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    """Convert a list of sqlite3.Row objects to a list of dicts."""
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# PlayerRepository
# ---------------------------------------------------------------------------


class PlayerRepository:
    """CRUD operations for the ``players`` table.

    Args:
        connection: An open ``sqlite3.Connection`` with foreign keys enabled.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection

    def get_or_create(self, name: str) -> dict:
        """Retrieve an existing player by name, or create a new one.

        Args:
            name: The player's screen name.

        Returns:
            A dict representing the player row.
        """
        existing = self.get_by_name(name)
        if existing is not None:
            return existing

        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO players (name) VALUES (?)",
            (name,),
        )
        self._conn.commit()
        logger.debug("player_created", player_name=name)
        return self.get_by_name(name)  # type: ignore[return-value]

    def get_by_name(self, name: str) -> Optional[dict]:
        """Fetch a player by screen name.

        Args:
            name: The player's screen name.

        Returns:
            A dict representing the player row, or None if not found.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM players WHERE name = ?", (name,))
        return _row_to_dict(cur.fetchone())

    def update(self, player: dict) -> None:
        """Update a player record.

        The dict must contain an ``id`` key.  All mutable fields
        (``last_seen``, ``total_hands``, ``notes``) are written back.

        Args:
            player: A dict with at least ``id``, ``last_seen``,
                ``total_hands``, and ``notes``.
        """
        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE players
               SET last_seen   = ?,
                   total_hands = ?,
                   notes       = ?
             WHERE id = ?
            """,
            (
                player.get("last_seen", time.strftime("%Y-%m-%d %H:%M:%S")),
                player.get("total_hands", 0),
                player.get("notes", ""),
                player["id"],
            ),
        )
        self._conn.commit()

    def get_all(self) -> list[dict]:
        """Return all player records.

        Returns:
            A list of dicts, one per player.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM players ORDER BY name")
        return _rows_to_dicts(cur.fetchall())


# ---------------------------------------------------------------------------
# HandRepository
# ---------------------------------------------------------------------------


class HandRepository:
    """Operations for persisting and querying complete poker hands.

    Args:
        connection: An open ``sqlite3.Connection`` with foreign keys enabled.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection

    def save_hand(self, game_state: GameState) -> int:
        """Persist a complete hand from a ``GameState`` snapshot.

        This inserts rows into ``hands``, ``hand_players``, ``actions``, and
        ``community_cards`` within a single transaction.

        Args:
            game_state: The completed hand state to persist.

        Returns:
            The database id of the newly inserted hand row.
        """
        cur = self._conn.cursor()
        player_repo = PlayerRepository(self._conn)

        try:
            # 1. Insert the hand record.
            cur.execute(
                """
                INSERT INTO hands
                    (hand_number, table_name, small_blind, big_blind, ante,
                     timestamp, hero_seat)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_state.hand_number,
                    game_state.table_name,
                    game_state.small_blind,
                    game_state.big_blind,
                    game_state.ante,
                    game_state.timestamp,
                    game_state.hero_seat,
                ),
            )
            hand_id = cur.lastrowid
            assert hand_id is not None

            # 2. Upsert players and insert hand_players join rows.
            for player in game_state.players:
                player_row = player_repo.get_or_create(player.name)
                player_id = player_row["id"]

                hole_cards_str = (
                    _cards_to_json(player.hole_cards)
                    if player.hole_cards
                    else ""
                )

                cur.execute(
                    """
                    INSERT INTO hand_players
                        (hand_id, player_id, seat, position, stack_start,
                         stack_end, hole_cards, is_winner, net_profit)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        hand_id,
                        player_id,
                        player.seat_number,
                        player.position.value if player.position else "",
                        player.stack_size + player.current_bet,
                        player.stack_size,
                        hole_cards_str,
                        0,  # is_winner -- to be updated by stats aggregation
                        0.0,  # net_profit -- to be updated by stats aggregation
                    ),
                )

                # Update player stats.
                cur.execute(
                    """
                    UPDATE players
                       SET last_seen   = datetime('now'),
                           total_hands = total_hands + 1
                     WHERE id = ?
                    """,
                    (player_id,),
                )

            # 3. Insert actions.
            for seq, action in enumerate(game_state.action_history):
                player_row = player_repo.get_by_name(action.player_name)
                if player_row is None:
                    continue
                cur.execute(
                    """
                    INSERT INTO actions
                        (hand_id, player_id, street, action_type, amount,
                         is_all_in, sequence_number)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        hand_id,
                        player_row["id"],
                        action.street.name,
                        action.action_type.value,
                        action.amount,
                        1 if action.action_type == ActionType.ALL_IN else 0,
                        seq,
                    ),
                )

            # 4. Insert community cards per street.
            cards_by_street = self._group_community_cards(game_state)
            for street_name, cards_json in cards_by_street.items():
                cur.execute(
                    """
                    INSERT INTO community_cards (hand_id, street, cards)
                    VALUES (?, ?, ?)
                    """,
                    (hand_id, street_name, cards_json),
                )

            self._conn.commit()
            logger.info(
                "hand_saved",
                hand_id=hand_id,
                hand_number=game_state.hand_number,
                num_players=len(game_state.players),
            )
            return hand_id

        except Exception:
            self._conn.rollback()
            logger.exception("hand_save_failed")
            raise

    def get_hand(self, hand_id: int) -> Optional[dict]:
        """Fetch a hand by its database id.

        Args:
            hand_id: The primary key of the hand row.

        Returns:
            A dict representing the hand, or None if not found.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM hands WHERE id = ?", (hand_id,))
        return _row_to_dict(cur.fetchone())

    def get_hands_for_player(
        self, player_name: str, limit: int = 100
    ) -> list[dict]:
        """Fetch hands in which a given player participated.

        Args:
            player_name: The player's screen name.
            limit: Maximum number of hands to return.

        Returns:
            A list of hand dicts ordered by timestamp descending.
        """
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT h.*
              FROM hands h
              JOIN hand_players hp ON h.id = hp.hand_id
              JOIN players p       ON hp.player_id = p.id
             WHERE p.name = ?
             ORDER BY h.timestamp DESC
             LIMIT ?
            """,
            (player_name, limit),
        )
        return _rows_to_dicts(cur.fetchall())

    def get_recent_hands(self, limit: int = 50) -> list[dict]:
        """Fetch the most recent hands.

        Args:
            limit: Maximum number of hands to return.

        Returns:
            A list of hand dicts ordered by timestamp descending.
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM hands ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return _rows_to_dicts(cur.fetchall())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_community_cards(game_state: GameState) -> dict[str, str]:
        """Group community cards by the street they were dealt on.

        Returns a dict mapping street names to JSON-serialized card lists.
        """
        cards = game_state.community_cards
        result: dict[str, str] = {}

        if len(cards) >= 3:
            result[Street.FLOP.name] = _cards_to_json(cards[:3])
        if len(cards) >= 4:
            result[Street.TURN.name] = _cards_to_json(cards[:4])
        if len(cards) >= 5:
            result[Street.RIVER.name] = _cards_to_json(cards[:5])

        return result


# ---------------------------------------------------------------------------
# ActionRepository
# ---------------------------------------------------------------------------


class ActionRepository:
    """Query operations for the ``actions`` table.

    Args:
        connection: An open ``sqlite3.Connection`` with foreign keys enabled.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection

    def get_actions_for_hand(self, hand_id: int) -> list[dict]:
        """Fetch all actions for a given hand, ordered by sequence number.

        Args:
            hand_id: The database id of the hand.

        Returns:
            A list of action dicts.
        """
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT a.*, p.name AS player_name
              FROM actions a
              JOIN players p ON a.player_id = p.id
             WHERE a.hand_id = ?
             ORDER BY a.sequence_number
            """,
            (hand_id,),
        )
        return _rows_to_dicts(cur.fetchall())

    def get_actions_for_player(
        self,
        player_name: str,
        street: Optional[str] = None,
    ) -> list[dict]:
        """Fetch all actions for a given player, optionally filtered by street.

        Args:
            player_name: The player's screen name.
            street: Optional street name filter (e.g. ``"PREFLOP"``).

        Returns:
            A list of action dicts ordered by hand and sequence number.
        """
        cur = self._conn.cursor()

        if street is not None:
            cur.execute(
                """
                SELECT a.*, p.name AS player_name
                  FROM actions a
                  JOIN players p ON a.player_id = p.id
                 WHERE p.name = ? AND a.street = ?
                 ORDER BY a.hand_id, a.sequence_number
                """,
                (player_name, street),
            )
        else:
            cur.execute(
                """
                SELECT a.*, p.name AS player_name
                  FROM actions a
                  JOIN players p ON a.player_id = p.id
                 WHERE p.name = ?
                 ORDER BY a.hand_id, a.sequence_number
                """,
                (player_name,),
            )

        return _rows_to_dicts(cur.fetchall())
