"""SQLite connection management for the stats subsystem.

Provides ``ConnectionManager``, which configures WAL mode, connection pooling
with thread safety, automatic database initialization, and graceful shutdown.

Usage::

    from src.common.config import StatsConfig
    from src.stats.connection import ConnectionManager

    manager = ConnectionManager(StatsConfig())
    with manager.get_connection() as conn:
        conn.execute("SELECT 1")
    manager.close()
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from src.common.config import StatsConfig
from src.common.logging import get_logger
from src.stats.database import DatabaseManager

logger = get_logger("stats.connection")


class ConnectionManager:
    """Thread-safe SQLite connection manager with WAL mode.

    Maintains one connection per thread (thread-local storage) so that
    concurrent readers never block each other under WAL journaling.

    Args:
        config: Stats subsystem configuration containing ``db_path`` and
            ``wal_mode`` settings.
    """

    def __init__(self, config: StatsConfig | None = None) -> None:
        self._config = config or StatsConfig()
        self._db_path = self._config.db_path
        self._wal_mode = self._config.wal_mode
        self._local = threading.local()
        self._lock = threading.Lock()
        self._initialized = False
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a configured ``sqlite3.Connection``.

        The connection is stored per-thread and reused across calls.  On first
        use the database schema is auto-initialized.

        Yields:
            A ready-to-use ``sqlite3.Connection`` with foreign keys and
            WAL mode enabled.

        Raises:
            RuntimeError: If the manager has already been closed.
        """
        if self._closed:
            raise RuntimeError("ConnectionManager has been closed")

        conn = self._get_thread_connection()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise

    def close(self) -> None:
        """Close all open connections and mark the manager as shut down.

        Safe to call multiple times.
        """
        if self._closed:
            return

        self._closed = True
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.ProgrammingError:
                pass
            self._local.connection = None

        logger.info("connection_manager_closed", db_path=self._db_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_thread_connection(self) -> sqlite3.Connection:
        """Return (or create) the connection for the current thread."""
        conn: sqlite3.Connection | None = getattr(
            self._local, "connection", None
        )
        if conn is not None:
            return conn

        conn = self._create_connection()
        self._local.connection = conn

        # Auto-initialize on first connection across all threads.
        with self._lock:
            if not self._initialized:
                self._initialize_db(conn)
                self._initialized = True

        return conn

    def _create_connection(self) -> sqlite3.Connection:
        """Open a new SQLite connection and apply PRAGMA settings."""
        # Ensure parent directory exists for file-based databases.
        if self._db_path != ":memory:":
            db_dir = Path(self._db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row

        # Apply PRAGMA settings.
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA synchronous = NORMAL;")

        if self._wal_mode:
            conn.execute("PRAGMA journal_mode = WAL;")
            logger.debug("wal_mode_enabled", db_path=self._db_path)

        return conn

    def _initialize_db(self, conn: sqlite3.Connection) -> None:
        """Run schema initialization on the provided connection."""
        logger.info("auto_initializing_database", db_path=self._db_path)
        manager = DatabaseManager(conn)
        manager.initialize()
