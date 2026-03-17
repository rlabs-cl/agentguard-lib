"""StatsCollector — singleton that records per-tool-call metrics.

Thread-safe via SQLite WAL mode.  All writes go through a single
``get_collector()`` instance per process.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import UTC, datetime
from uuid import uuid4

from agentguard.stats.db import get_connection

logger = logging.getLogger(__name__)


def estimate_tokens(text: str | None) -> int:
    """Rough token estimate: len(text) // 4, minimum 1."""
    if not text:
        return 1
    return max(1, len(str(text)) // 4)


class StatsCollector:
    """Collects tool-call and compaction metrics for the current session."""

    def __init__(self) -> None:
        self.session_id: str = str(uuid4())
        self.project_path: str = os.getcwd()
        self.project_name: str = os.path.basename(self.project_path)
        self._started_at: str = datetime.now(UTC).isoformat()
        self._tool_calls: int = 0
        self._total_tokens: int = 0
        self._total_duration_ms: int = 0
        self._tools_used: set[str] = set()
        self._archetypes_used: set[str] = set()
        self._compaction_count: int = 0

    # ── Recording ──────────────────────────────────────────────────

    def record_tool_call(
        self,
        tool_name: str,
        duration_ms: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        archetype: str | None = None,
        status: str = "success",
        error_message: str | None = None,
        parameters_summary: str | None = None,
        context_size_chars: int = 0,
    ) -> None:
        """Insert a row into ``tool_events`` and update running totals."""
        total_tokens = input_tokens + output_tokens
        now = datetime.now(UTC).isoformat()

        try:
            conn = get_connection()
            try:
                conn.execute(
                    """INSERT INTO tool_events
                       (timestamp, session_id, project_path, project_name,
                        tool_name, archetype, duration_ms,
                        input_tokens, output_tokens, total_tokens,
                        context_size_chars, parameters_summary, status, error_message)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        now,
                        self.session_id,
                        self.project_path,
                        self.project_name,
                        tool_name,
                        archetype,
                        duration_ms,
                        input_tokens,
                        output_tokens,
                        total_tokens,
                        context_size_chars,
                        parameters_summary,
                        status,
                        error_message,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            logger.debug("Failed to record tool call for %s", tool_name, exc_info=True)
            return

        # Update in-memory totals
        self._tool_calls += 1
        self._total_tokens += total_tokens
        self._total_duration_ms += duration_ms
        self._tools_used.add(tool_name)
        if archetype:
            self._archetypes_used.add(archetype)

    def record_compaction(
        self,
        context_before_chars: int,
        context_after_chars: int,
    ) -> None:
        """Insert a row into ``compaction_events``."""
        now = datetime.now(UTC).isoformat()
        reduction_pct = 0.0
        if context_before_chars > 0:
            reduction_pct = round(
                (1 - context_after_chars / context_before_chars) * 100, 2,
            )

        try:
            conn = get_connection()
            try:
                conn.execute(
                    """INSERT INTO compaction_events
                       (timestamp, session_id, project_path,
                        context_before_chars, context_after_chars, reduction_pct)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        now,
                        self.session_id,
                        self.project_path,
                        context_before_chars,
                        context_after_chars,
                        reduction_pct,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            logger.debug("Failed to record compaction event", exc_info=True)
            return

        self._compaction_count += 1

    def finalize_session(self) -> None:
        """Write (or update) the session summary row."""
        now = datetime.now(UTC).isoformat()
        try:
            conn = get_connection()
            try:
                conn.execute(
                    """INSERT INTO session_summaries
                       (session_id, project_path, started_at, ended_at,
                        total_tool_calls, total_tokens, total_duration_ms,
                        tools_used, archetypes_used, compaction_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(session_id) DO UPDATE SET
                        ended_at = excluded.ended_at,
                        total_tool_calls = excluded.total_tool_calls,
                        total_tokens = excluded.total_tokens,
                        total_duration_ms = excluded.total_duration_ms,
                        tools_used = excluded.tools_used,
                        archetypes_used = excluded.archetypes_used,
                        compaction_count = excluded.compaction_count""",
                    (
                        self.session_id,
                        self.project_path,
                        self._started_at,
                        now,
                        self._tool_calls,
                        self._total_tokens,
                        self._total_duration_ms,
                        json.dumps(sorted(self._tools_used)),
                        json.dumps(sorted(self._archetypes_used)),
                        self._compaction_count,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            logger.debug("Failed to finalize session %s", self.session_id, exc_info=True)


# ── Module-level singleton ─────────────────────────────────────────

_collector: StatsCollector | None = None
_lock = threading.Lock()


def get_collector() -> StatsCollector:
    """Return the process-wide StatsCollector singleton."""
    global _collector
    if _collector is None:
        with _lock:
            if _collector is None:
                _collector = StatsCollector()
    return _collector
