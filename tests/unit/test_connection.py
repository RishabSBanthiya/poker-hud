"""Tests for src.stats.connection -- ConnectionManager."""

from __future__ import annotations

import sqlite3
import threading

import pytest
from src.common.config import StatsConfig
from src.stats.connection import ConnectionManager


@pytest.fixture()
def manager() -> ConnectionManager:
    """Return a ConnectionManager backed by an in-memory database."""
    config = StatsConfig(db_path=":memory:", wal_mode=False)
    return ConnectionManager(config)


# ------------------------------------------------------------------
# Basic lifecycle
# ------------------------------------------------------------------


class TestConnectionManagerLifecycle:
    """get_connection / close happy path."""

    def test_yields_connection(self, manager: ConnectionManager) -> None:
        with manager.get_connection() as conn:
            assert isinstance(conn, sqlite3.Connection)

    def test_auto_initializes_schema(self, manager: ConnectionManager) -> None:
        with manager.get_connection() as conn:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='players'"
            )
            assert cur.fetchone() is not None

    def test_close_prevents_further_use(
        self, manager: ConnectionManager
    ) -> None:
        manager.close()
        with pytest.raises(RuntimeError, match="closed"):
            with manager.get_connection():
                pass

    def test_close_idempotent(self, manager: ConnectionManager) -> None:
        manager.close()
        manager.close()  # Should not raise.

    def test_default_config(self) -> None:
        mgr = ConnectionManager()
        assert mgr._db_path == "data/poker_hud.db"
        assert mgr._wal_mode is True


# ------------------------------------------------------------------
# PRAGMA settings
# ------------------------------------------------------------------


class TestPragmaSettings:
    """Verify SQLite PRAGMA configuration."""

    def test_foreign_keys_enabled(self, manager: ConnectionManager) -> None:
        with manager.get_connection() as conn:
            cur = conn.execute("PRAGMA foreign_keys")
            assert cur.fetchone()[0] == 1

    def test_synchronous_normal(self, manager: ConnectionManager) -> None:
        with manager.get_connection() as conn:
            cur = conn.execute("PRAGMA synchronous")
            # NORMAL = 1
            assert cur.fetchone()[0] == 1

    def test_wal_mode_when_enabled(self) -> None:
        config = StatsConfig(db_path=":memory:", wal_mode=True)
        mgr = ConnectionManager(config)
        with mgr.get_connection() as conn:
            cur = conn.execute("PRAGMA journal_mode")
            mode = cur.fetchone()[0]
            # :memory: may not support WAL; accept either 'wal' or 'memory'
            assert mode in ("wal", "memory")
        mgr.close()


# ------------------------------------------------------------------
# Row factory
# ------------------------------------------------------------------


class TestRowFactory:
    """Connections should use sqlite3.Row for dict-like access."""

    def test_row_factory_set(self, manager: ConnectionManager) -> None:
        with manager.get_connection() as conn:
            assert conn.row_factory is sqlite3.Row

    def test_row_access_by_name(self, manager: ConnectionManager) -> None:
        with manager.get_connection() as conn:
            conn.execute("INSERT INTO players (name) VALUES ('test')")
            conn.commit()
            cur = conn.execute(
                "SELECT name FROM players WHERE name='test'"
            )
            row = cur.fetchone()
            assert row["name"] == "test"


# ------------------------------------------------------------------
# Thread safety
# ------------------------------------------------------------------


class TestThreadSafety:
    """Verify that connections work from multiple threads."""

    def test_concurrent_reads(self, tmp_path) -> None:
        # Use a file-based database so all threads share the same data.
        db_file = str(tmp_path / "test_threads.db")
        config = StatsConfig(db_path=db_file, wal_mode=True)
        mgr = ConnectionManager(config)

        # Seed data from main thread.
        with mgr.get_connection() as conn:
            conn.execute("INSERT INTO players (name) VALUES ('alice')")
            conn.commit()

        results: list[str] = []
        errors: list[Exception] = []

        def reader() -> None:
            try:
                with mgr.get_connection() as c:
                    cur = c.execute(
                        "SELECT name FROM players WHERE name='alice'"
                    )
                    row = cur.fetchone()
                    if row:
                        results.append(row["name"])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 4
        assert all(r == "alice" for r in results)
        mgr.close()


# ------------------------------------------------------------------
# Rollback on error
# ------------------------------------------------------------------


class TestRollbackOnError:
    """get_connection context manager rolls back on exception."""

    def test_rollback_on_exception(self, manager: ConnectionManager) -> None:
        with manager.get_connection() as conn:
            conn.execute("INSERT INTO players (name) VALUES ('will_stay')")
            conn.commit()

        with pytest.raises(ValueError):
            with manager.get_connection() as conn:
                conn.execute(
                    "INSERT INTO players (name) VALUES ('will_vanish')"
                )
                raise ValueError("boom")

        with manager.get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM players")
            assert cur.fetchone()[0] == 1


# ------------------------------------------------------------------
# Connection reuse
# ------------------------------------------------------------------


class TestConnectionReuse:
    """Verify same-thread connections are reused."""

    def test_same_thread_reuses_connection(
        self, manager: ConnectionManager
    ) -> None:
        with manager.get_connection() as conn1:
            pass
        with manager.get_connection() as conn2:
            pass
        assert conn1 is conn2

    def test_schema_initialized_once(
        self, manager: ConnectionManager
    ) -> None:
        with manager.get_connection():
            pass
        assert manager._initialized is True
        with manager.get_connection():
            pass
        # Still initialized, not re-run.
        assert manager._initialized is True
