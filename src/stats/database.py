"""SQLite database schema and initialization for the stats subsystem.

Defines the schema for tracking poker hands, players, actions, and community
cards.  The ``DatabaseManager`` class handles table creation, index setup, and
schema versioning so the database can be migrated forward as the application
evolves.
"""

from __future__ import annotations

import sqlite3
from typing import Callable

from src.common.logging import get_logger

logger = get_logger("stats.database")

# Current schema version -- bump when adding migrations.
SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# DDL statements
# ---------------------------------------------------------------------------

_CREATE_SCHEMA_VERSION = """
CREATE TABLE IF NOT EXISTS schema_version (
    version   INTEGER NOT NULL,
    applied_at TEXT   NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_PLAYERS = """
CREATE TABLE IF NOT EXISTS players (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    first_seen  TEXT    NOT NULL DEFAULT (datetime('now')),
    last_seen   TEXT    NOT NULL DEFAULT (datetime('now')),
    total_hands INTEGER NOT NULL DEFAULT 0,
    notes       TEXT    NOT NULL DEFAULT ''
);
"""

_CREATE_HANDS = """
CREATE TABLE IF NOT EXISTS hands (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_number INTEGER NOT NULL,
    table_name  TEXT    NOT NULL DEFAULT '',
    small_blind REAL    NOT NULL DEFAULT 0.0,
    big_blind   REAL    NOT NULL DEFAULT 0.0,
    ante        REAL    NOT NULL DEFAULT 0.0,
    timestamp   REAL    NOT NULL,
    hero_seat   INTEGER NOT NULL DEFAULT 0
);
"""

_CREATE_HAND_PLAYERS = """
CREATE TABLE IF NOT EXISTS hand_players (
    hand_id     INTEGER NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    player_id   INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    seat        INTEGER NOT NULL,
    position    TEXT    NOT NULL DEFAULT '',
    stack_start REAL    NOT NULL DEFAULT 0.0,
    stack_end   REAL    NOT NULL DEFAULT 0.0,
    hole_cards  TEXT    NOT NULL DEFAULT '',
    is_winner   INTEGER NOT NULL DEFAULT 0,
    net_profit  REAL    NOT NULL DEFAULT 0.0,
    PRIMARY KEY (hand_id, player_id)
);
"""

_CREATE_ACTIONS = """
CREATE TABLE IF NOT EXISTS actions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id         INTEGER NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    player_id       INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    street          TEXT    NOT NULL,
    action_type     TEXT    NOT NULL,
    amount          REAL    NOT NULL DEFAULT 0.0,
    is_all_in       INTEGER NOT NULL DEFAULT 0,
    sequence_number INTEGER NOT NULL DEFAULT 0
);
"""

_CREATE_COMMUNITY_CARDS = """
CREATE TABLE IF NOT EXISTS community_cards (
    hand_id INTEGER NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    street  TEXT    NOT NULL,
    cards   TEXT    NOT NULL DEFAULT '[]',
    PRIMARY KEY (hand_id, street)
);
"""

# ---------------------------------------------------------------------------
# Index statements
# ---------------------------------------------------------------------------

_INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_players_name ON players(name);",
    "CREATE INDEX IF NOT EXISTS idx_hands_timestamp ON hands(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_hands_table_name ON hands(table_name);",
    "CREATE INDEX IF NOT EXISTS idx_hand_players_player_id ON hand_players(player_id);",
    "CREATE INDEX IF NOT EXISTS idx_hand_players_hand_id ON hand_players(hand_id);",
    "CREATE INDEX IF NOT EXISTS idx_actions_hand_id ON actions(hand_id);",
    "CREATE INDEX IF NOT EXISTS idx_actions_player_id ON actions(player_id);",
    "CREATE INDEX IF NOT EXISTS idx_actions_street ON actions(street);",
    "CREATE INDEX IF NOT EXISTS idx_community_cards_hand_id "
    "ON community_cards(hand_id);",
]


class DatabaseManager:
    """Manages SQLite schema creation, indexing, and versioning.

    Args:
        connection: An open ``sqlite3.Connection`` (foreign keys should
            already be enabled by the caller / connection manager).
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create all tables and indexes if they do not already exist.

        Inserts the current ``SCHEMA_VERSION`` into the ``schema_version``
        table when creating a fresh database.  For existing databases whose
        version is behind, ``_apply_migrations`` is called to bring the
        schema up to date.
        """
        logger.info("database_initialize_start", schema_version=SCHEMA_VERSION)

        cur = self._conn.cursor()
        cur.execute(_CREATE_SCHEMA_VERSION)

        current_version = self._get_current_version()

        if current_version == 0:
            # Fresh database -- create everything.
            self._create_tables()
            self._create_indexes()
            cur.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            self._conn.commit()
            logger.info(
                "database_initialized_fresh", schema_version=SCHEMA_VERSION
            )
        elif current_version < SCHEMA_VERSION:
            self._apply_migrations(current_version, SCHEMA_VERSION)
            self._conn.commit()
            logger.info(
                "database_migrated",
                from_version=current_version,
                to_version=SCHEMA_VERSION,
            )
        else:
            logger.info(
                "database_already_current", schema_version=current_version
            )

    def get_schema_version(self) -> int:
        """Return the current schema version recorded in the database.

        Returns:
            The version integer, or 0 if the version table does not exist or
            is empty.
        """
        return self._get_current_version()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_current_version(self) -> int:
        """Read the latest schema version from the database."""
        cur = self._conn.cursor()
        try:
            cur.execute("SELECT MAX(version) FROM schema_version")
            row = cur.fetchone()
            return row[0] if row and row[0] is not None else 0
        except sqlite3.OperationalError:
            # Table doesn't exist yet.
            return 0

    def _create_tables(self) -> None:
        """Execute all CREATE TABLE statements."""
        cur = self._conn.cursor()
        for ddl in (
            _CREATE_PLAYERS,
            _CREATE_HANDS,
            _CREATE_HAND_PLAYERS,
            _CREATE_ACTIONS,
            _CREATE_COMMUNITY_CARDS,
        ):
            cur.execute(ddl)

    def _create_indexes(self) -> None:
        """Execute all CREATE INDEX statements."""
        cur = self._conn.cursor()
        for stmt in _INDEXES:
            cur.execute(stmt)

    def _apply_migrations(self, from_version: int, to_version: int) -> None:
        """Apply incremental migrations from *from_version* to *to_version*.

        Each migration is a function named ``_migrate_vN_to_vM`` where N and M
        are consecutive version numbers.  Add new migration functions here as
        the schema evolves.

        Args:
            from_version: The version the database is currently at.
            to_version: The target version to migrate to.
        """
        migrations: dict[tuple[int, int], Callable[[], None]] = {
            # Example: (1, 2): self._migrate_v1_to_v2,
        }

        for v in range(from_version, to_version):
            key = (v, v + 1)
            migrate_fn = migrations.get(key)
            if migrate_fn is None:
                raise RuntimeError(
                    f"No migration path from schema v{v} to v{v + 1}"
                )
            logger.info("applying_migration", from_v=v, to_v=v + 1)
            migrate_fn()

        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (to_version,),
        )
