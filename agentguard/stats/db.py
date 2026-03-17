"""SQLite database for AgentGuard usage statistics.

Stores per-tool-call metrics, compaction events, and session summaries
in ``~/.agentguard/stats.db`` with WAL mode for concurrent safety.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_DB_DIR = os.path.join(Path.home(), ".agentguard")
_DB_PATH = os.path.join(_DB_DIR, "stats.db")

_TOOL_EVENTS_DDL = """\
CREATE TABLE IF NOT EXISTS tool_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    session_id      TEXT    NOT NULL,
    project_path    TEXT,
    project_name    TEXT,
    tool_name       TEXT    NOT NULL,
    archetype       TEXT,
    duration_ms     INTEGER NOT NULL DEFAULT 0,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    context_size_chars INTEGER NOT NULL DEFAULT 0,
    parameters_summary TEXT,
    status          TEXT    NOT NULL DEFAULT 'success',
    error_message   TEXT
);
"""

_COMPACTION_EVENTS_DDL = """\
CREATE TABLE IF NOT EXISTS compaction_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT    NOT NULL,
    session_id          TEXT    NOT NULL,
    project_path        TEXT,
    context_before_chars INTEGER NOT NULL DEFAULT 0,
    context_after_chars  INTEGER NOT NULL DEFAULT 0,
    reduction_pct       REAL   NOT NULL DEFAULT 0.0
);
"""

_SESSION_SUMMARIES_DDL = """\
CREATE TABLE IF NOT EXISTS session_summaries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    UNIQUE NOT NULL,
    project_path    TEXT,
    started_at      TEXT,
    ended_at        TEXT,
    total_tool_calls INTEGER NOT NULL DEFAULT 0,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    total_duration_ms INTEGER NOT NULL DEFAULT 0,
    tools_used      TEXT,
    archetypes_used TEXT,
    compaction_count INTEGER NOT NULL DEFAULT 0
);
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_tool_events_session ON tool_events(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_tool_events_tool ON tool_events(tool_name);",
    "CREATE INDEX IF NOT EXISTS idx_tool_events_ts ON tool_events(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_compaction_session ON compaction_events(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_session_summaries_sid ON session_summaries(session_id);",
]


def _ensure_dir() -> None:
    """Create the database directory if it does not exist."""
    os.makedirs(_DB_DIR, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Return a sqlite3 Connection with WAL mode, creating tables if needed.

    The database lives at ``~/.agentguard/stats.db``.  Tables and indexes
    are created on the first call (idempotent ``CREATE IF NOT EXISTS``).
    """
    _ensure_dir()
    try:
        conn = sqlite3.connect(_DB_PATH, timeout=5)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=3000;")
        conn.row_factory = sqlite3.Row
        # Create tables idempotently
        conn.execute(_TOOL_EVENTS_DDL)
        conn.execute(_COMPACTION_EVENTS_DDL)
        conn.execute(_SESSION_SUMMARIES_DDL)
        for idx_sql in _INDEXES:
            conn.execute(idx_sql)
        conn.commit()
        return conn
    except sqlite3.Error:
        logger.exception("Failed to open or initialize stats database at %s", _DB_PATH)
        raise


def init_db() -> None:
    """Explicitly initialize the database (creates tables if needed).

    Safe to call multiple times.
    """
    conn = get_connection()
    conn.close()
