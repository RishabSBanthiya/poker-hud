"""SQLite connection management with thread safety.

Provides a connection manager that handles database creation,
schema initialization, and connection pooling for SQLite.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS hands (
    hand_id TEXT PRIMARY KEY,
    players_json TEXT NOT NULL,
    actions_json TEXT NOT NULL,
    community_cards TEXT NOT NULL DEFAULT '',
    pot REAL NOT NULL DEFAULT 0.0,
    winner_name TEXT NOT NULL DEFAULT '',
    big_blind REAL NOT NULL DEFAULT 1.0,
    timestamp REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS player_stats (
    player_name TEXT PRIMARY KEY,
    total_hands INTEGER NOT NULL DEFAULT 0,
    vpip_hands INTEGER NOT NULL DEFAULT 0,
    pfr_hands INTEGER NOT NULL DEFAULT 0,
    three_bet_opportunities INTEGER NOT NULL DEFAULT 0,
    three_bet_hands INTEGER NOT NULL DEFAULT 0,
    cbet_opportunities INTEGER NOT NULL DEFAULT 0,
    cbet_hands INTEGER NOT NULL DEFAULT 0,
    total_bets INTEGER NOT NULL DEFAULT 0,
    total_raises INTEGER NOT NULL DEFAULT 0,
    total_calls INTEGER NOT NULL DEFAULT 0,
    total_folds INTEGER NOT NULL DEFAULT 0,
    went_to_showdown INTEGER NOT NULL DEFAULT 0,
    showdown_opportunities INTEGER NOT NULL DEFAULT 0
);
"""


class ConnectionManager:
    """Manages SQLite database connections with thread-local storage.

    Args:
        db_path: Path to the SQLite database file. Use ':memory:'
            for an in-memory database.
    """

    def __init__(self, db_path: str = "data/poker_hud.db") -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._initialized = False

    @property
    def db_path(self) -> str:
        """The database file path."""
        return self._db_path

    @property
    def is_initialized(self) -> bool:
        """Whether the schema has been initialized."""
        return self._initialized

    def initialize(self) -> None:
        """Create the database file and initialize the schema.

        Creates parent directories if needed.
        """
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = self.get_connection()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        self._initialized = True
        logger.info("Database initialized at %s", self._db_path)

    def get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection.

        Returns:
            An SQLite connection for the current thread.
        """
        conn: Optional[sqlite3.Connection] = getattr(
            self._local, "connection", None
        )
        if conn is None:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.connection = conn
        return conn

    def close(self) -> None:
        """Close the connection for the current thread."""
        conn: Optional[sqlite3.Connection] = getattr(
            self._local, "connection", None
        )
        if conn is not None:
            conn.close()
            self._local.connection = None
        logger.info("Database connection closed")

    def close_all(self) -> None:
        """Close all connections. Call during shutdown."""
        self.close()
        self._initialized = False
