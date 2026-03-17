"""Aggregation queries for the ``get_usage_stats`` MCP tool.

All SQL uses parameterized placeholders (``?``) — no f-strings in queries.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from agentguard.stats.db import get_connection

logger = logging.getLogger(__name__)


def _period_start(period: str) -> str | None:
    """Return an ISO-8601 timestamp for the start of *period*, or None for 'all'."""
    now = datetime.now(UTC)
    if period == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        start = now - timedelta(days=7)
    elif period == "month":
        start = now - timedelta(days=30)
    elif period == "all":
        return None
    else:
        # Default to week for unknown periods
        start = now - timedelta(days=7)
    return start.isoformat()


def get_usage_summary(
    period: str = "week",
    project: str | None = None,
    group_by: str = "tool",
) -> dict[str, Any]:
    """Return aggregated stats for the given period.

    Parameters
    ----------
    period : str
        ``'today'``, ``'week'``, ``'month'``, or ``'all'``.
    project : str or None
        Filter by project name (substring match).
    group_by : str
        ``'tool'``, ``'project'``, ``'day'``, or ``'session'``.

    Returns
    -------
    dict
        Keys: ``total_calls``, ``total_tokens``, ``total_duration_ms``,
        ``top_tools``, ``avg_context_size``, ``compaction_count``, ``breakdown``.
    """
    period_start = _period_start(period)
    conditions: list[str] = []
    params: list[Any] = []

    if period_start is not None:
        conditions.append("timestamp >= ?")
        params.append(period_start)
    if project:
        conditions.append("project_name LIKE ?")
        params.append(f"%{project}%")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    try:
        conn = get_connection()
        try:
            # Totals
            row = conn.execute(
                f"SELECT COUNT(*) AS cnt, COALESCE(SUM(total_tokens),0) AS tokens, "
                f"COALESCE(SUM(duration_ms),0) AS dur, "
                f"COALESCE(AVG(context_size_chars),0) AS avg_ctx "
                f"FROM tool_events {where}",
                params,
            ).fetchone()
            total_calls = row["cnt"]
            total_tokens = row["tokens"]
            total_duration_ms = row["dur"]
            avg_context_size = int(row["avg_ctx"])

            # Top tools
            top_tools_rows = conn.execute(
                f"SELECT tool_name, COUNT(*) AS cnt, SUM(total_tokens) AS tokens "
                f"FROM tool_events {where} "
                f"GROUP BY tool_name ORDER BY cnt DESC LIMIT 10",
                params,
            ).fetchall()
            top_tools = [
                {"tool": r["tool_name"], "calls": r["cnt"], "tokens": r["tokens"]}
                for r in top_tools_rows
            ]

            # Compaction count
            comp_conditions: list[str] = []
            comp_params: list[Any] = []
            if period_start is not None:
                comp_conditions.append("timestamp >= ?")
                comp_params.append(period_start)
            if project:
                comp_conditions.append("project_path LIKE ?")
                comp_params.append(f"%{project}%")
            comp_where = ("WHERE " + " AND ".join(comp_conditions)) if comp_conditions else ""
            comp_row = conn.execute(
                f"SELECT COUNT(*) AS cnt FROM compaction_events {comp_where}",
                comp_params,
            ).fetchone()
            compaction_count = comp_row["cnt"]

            # Breakdown
            breakdown = _build_breakdown(conn, group_by, where, params)

            return {
                "period": period,
                "total_calls": total_calls,
                "total_tokens": total_tokens,
                "total_duration_ms": total_duration_ms,
                "avg_context_size_chars": avg_context_size,
                "top_tools": top_tools,
                "compaction_count": compaction_count,
                "breakdown": breakdown,
            }
        finally:
            conn.close()
    except Exception:
        logger.exception("Failed to query usage summary")
        return {
            "period": period,
            "total_calls": 0,
            "total_tokens": 0,
            "total_duration_ms": 0,
            "avg_context_size_chars": 0,
            "top_tools": [],
            "compaction_count": 0,
            "breakdown": [],
            "error": "Failed to query stats database",
        }


def _build_breakdown(
    conn: Any,
    group_by: str,
    where: str,
    params: list[Any],
) -> list[dict[str, Any]]:
    """Build the breakdown list according to *group_by*."""
    if group_by == "tool":
        group_col = "tool_name"
    elif group_by == "project":
        group_col = "project_name"
    elif group_by == "day":
        group_col = "DATE(timestamp)"
    elif group_by == "session":
        group_col = "session_id"
    else:
        group_col = "tool_name"

    rows = conn.execute(
        f"SELECT {group_col} AS grp, COUNT(*) AS cnt, "
        f"COALESCE(SUM(total_tokens),0) AS tokens, "
        f"COALESCE(SUM(duration_ms),0) AS dur "
        f"FROM tool_events {where} "
        f"GROUP BY grp ORDER BY cnt DESC",
        params,
    ).fetchall()
    return [
        {"group": r["grp"], "calls": r["cnt"], "tokens": r["tokens"], "duration_ms": r["dur"]}
        for r in rows
    ]


def get_session_history(limit: int = 10) -> list[dict[str, Any]]:
    """Return recent sessions with summaries."""
    try:
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM session_summaries ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
    except Exception:
        logger.exception("Failed to query session history")
        return []


def clear_stats(
    before: str | None = None,
    project: str | None = None,
) -> int:
    """Delete stats. Returns count of deleted records.

    Parameters
    ----------
    before : str or None
        ISO-8601 timestamp — delete records older than this.
    project : str or None
        Delete only records matching this project name (substring).
    """
    total_deleted = 0
    try:
        conn = get_connection()
        try:
            for table, ts_col, proj_col in [
                ("tool_events", "timestamp", "project_name"),
                ("compaction_events", "timestamp", "project_path"),
                ("session_summaries", "started_at", "project_path"),
            ]:
                conditions: list[str] = []
                params: list[Any] = []
                if before:
                    conditions.append(f"{ts_col} < ?")
                    params.append(before)
                if project:
                    conditions.append(f"{proj_col} LIKE ?")
                    params.append(f"%{project}%")

                where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
                cursor = conn.execute(f"DELETE FROM {table} {where}", params)
                total_deleted += cursor.rowcount

            conn.commit()
        finally:
            conn.close()
    except Exception:
        logger.exception("Failed to clear stats")
    return total_deleted


def get_raw_events(
    limit: int = 100,
    tool_name: str | None = None,
    project: str | None = None,
) -> list[dict[str, Any]]:
    """Return raw tool events for debugging."""
    conditions: list[str] = []
    params: list[Any] = []

    if tool_name:
        conditions.append("tool_name = ?")
        params.append(tool_name)
    if project:
        conditions.append("project_name LIKE ?")
        params.append(f"%{project}%")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    try:
        conn = get_connection()
        try:
            rows = conn.execute(
                f"SELECT * FROM tool_events {where} ORDER BY timestamp DESC LIMIT ?",
                [*params, limit],
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
    except Exception:
        logger.exception("Failed to query raw events")
        return []
