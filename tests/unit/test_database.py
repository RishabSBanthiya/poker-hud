"""Tests for src.stats.database -- schema creation and versioning."""

from __future__ import annotations

import sqlite3

import pytest
from src.stats.database import SCHEMA_VERSION, DatabaseManager


@pytest.fixture()
def conn() -> sqlite3.Connection:
    """Return an in-memory SQLite connection with foreign keys enabled."""
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys = ON;")
    c.row_factory = sqlite3.Row
    return c


# ------------------------------------------------------------------
# Table creation
# ------------------------------------------------------------------


class TestDatabaseManagerInitialize:
    """Tests for DatabaseManager.initialize()."""

    def test_creates_all_tables(self, conn: sqlite3.Connection) -> None:
        dm = DatabaseManager(conn)
        dm.initialize()

        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row["name"] for row in cur.fetchall()}

        expected = {
            "schema_version",
            "players",
            "hands",
            "hand_players",
            "actions",
            "community_cards",
        }
        assert expected.issubset(tables)

    def test_creates_indexes(self, conn: sqlite3.Connection) -> None:
        dm = DatabaseManager(conn)
        dm.initialize()

        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name LIKE 'idx_%'"
        )
        indexes = {row["name"] for row in cur.fetchall()}
        assert "idx_players_name" in indexes
        assert "idx_hands_timestamp" in indexes
        assert "idx_actions_hand_id" in indexes

    def test_sets_schema_version(self, conn: sqlite3.Connection) -> None:
        dm = DatabaseManager(conn)
        dm.initialize()
        assert dm.get_schema_version() == SCHEMA_VERSION

    def test_idempotent(self, conn: sqlite3.Connection) -> None:
        dm = DatabaseManager(conn)
        dm.initialize()
        dm.initialize()  # Should not raise.
        assert dm.get_schema_version() == SCHEMA_VERSION


# ------------------------------------------------------------------
# Schema version
# ------------------------------------------------------------------


class TestSchemaVersion:
    """Tests for schema versioning."""

    def test_version_zero_before_init(self, conn: sqlite3.Connection) -> None:
        dm = DatabaseManager(conn)
        assert dm.get_schema_version() == 0

    def test_version_after_init(self, conn: sqlite3.Connection) -> None:
        dm = DatabaseManager(conn)
        dm.initialize()
        assert dm.get_schema_version() == SCHEMA_VERSION


# ------------------------------------------------------------------
# Foreign keys
# ------------------------------------------------------------------


class TestForeignKeys:
    """Verify foreign key constraints are honoured."""

    def test_hand_players_fk_rejects_invalid_hand(
        self, conn: sqlite3.Connection
    ) -> None:
        dm = DatabaseManager(conn)
        dm.initialize()

        # Insert a player so we only violate the hand FK.
        conn.execute("INSERT INTO players (name) VALUES ('alice')")
        conn.commit()

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO hand_players
                    (hand_id, player_id, seat)
                VALUES (9999, 1, 0)
                """
            )

    def test_actions_fk_rejects_invalid_hand(
        self, conn: sqlite3.Connection
    ) -> None:
        dm = DatabaseManager(conn)
        dm.initialize()

        conn.execute("INSERT INTO players (name) VALUES ('alice')")
        conn.commit()

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO actions
                    (hand_id, player_id, street, action_type)
                VALUES (9999, 1, 'PREFLOP', 'fold')
                """
            )


# ------------------------------------------------------------------
# Column defaults
# ------------------------------------------------------------------


class TestColumnDefaults:
    """Verify sensible defaults on insert."""

    def test_player_defaults(self, conn: sqlite3.Connection) -> None:
        dm = DatabaseManager(conn)
        dm.initialize()

        conn.execute("INSERT INTO players (name) VALUES ('bob')")
        conn.commit()

        cur = conn.execute("SELECT * FROM players WHERE name = 'bob'")
        row = dict(cur.fetchone())
        assert row["total_hands"] == 0
        assert row["notes"] == ""
        assert row["first_seen"] is not None
        assert row["last_seen"] is not None


# ------------------------------------------------------------------
# All expected indexes
# ------------------------------------------------------------------


class TestAllIndexes:
    """Verify every defined index is created."""

    def test_all_indexes_created(self, conn: sqlite3.Connection) -> None:
        dm = DatabaseManager(conn)
        dm.initialize()

        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name LIKE 'idx_%'"
        )
        indexes = {row["name"] for row in cur.fetchall()}

        expected_indexes = {
            "idx_players_name",
            "idx_hands_timestamp",
            "idx_hands_table_name",
            "idx_hand_players_player_id",
            "idx_hand_players_hand_id",
            "idx_actions_hand_id",
            "idx_actions_player_id",
            "idx_actions_street",
            "idx_community_cards_hand_id",
        }
        assert expected_indexes.issubset(indexes)


# ------------------------------------------------------------------
# Cascade deletes
# ------------------------------------------------------------------


class TestCascadeDelete:
    """Verify ON DELETE CASCADE behaviour."""

    def test_deleting_hand_cascades_to_actions(
        self, conn: sqlite3.Connection
    ) -> None:
        dm = DatabaseManager(conn)
        dm.initialize()

        conn.execute("INSERT INTO players (name) VALUES ('alice')")
        conn.execute(
            "INSERT INTO hands (hand_number, timestamp) VALUES (1, 1000.0)"
        )
        conn.execute(
            """
            INSERT INTO actions
                (hand_id, player_id, street, action_type, sequence_number)
            VALUES (1, 1, 'PREFLOP', 'fold', 0)
            """
        )
        conn.commit()

        # Verify action exists
        cur = conn.execute("SELECT COUNT(*) FROM actions WHERE hand_id = 1")
        assert cur.fetchone()[0] == 1

        # Delete the hand
        conn.execute("DELETE FROM hands WHERE id = 1")
        conn.commit()

        # Action should be gone
        cur = conn.execute("SELECT COUNT(*) FROM actions WHERE hand_id = 1")
        assert cur.fetchone()[0] == 0

    def test_deleting_hand_cascades_to_community_cards(
        self, conn: sqlite3.Connection
    ) -> None:
        dm = DatabaseManager(conn)
        dm.initialize()

        conn.execute(
            "INSERT INTO hands (hand_number, timestamp) VALUES (1, 1000.0)"
        )
        conn.execute(
            """
            INSERT INTO community_cards (hand_id, street, cards)
            VALUES (1, 'FLOP', '["Ah", "Kd", "Qs"]')
            """
        )
        conn.commit()

        conn.execute("DELETE FROM hands WHERE id = 1")
        conn.commit()

        cur = conn.execute(
            "SELECT COUNT(*) FROM community_cards WHERE hand_id = 1"
        )
        assert cur.fetchone()[0] == 0
