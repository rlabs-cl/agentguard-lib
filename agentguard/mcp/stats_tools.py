"""MCP tool implementations for usage statistics."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


async def agentguard_get_usage_stats(
    period: str = "week",
    project: str | None = None,
    group_by: str = "tool",
) -> str:
    """Return aggregated usage statistics as a JSON string."""
    from agentguard.stats.queries import get_usage_summary

    try:
        result = get_usage_summary(period=period, project=project, group_by=group_by)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)[:200]})


async def agentguard_get_session_history(limit: int = 10) -> str:
    """Return recent session history as a JSON string."""
    from agentguard.stats.queries import get_session_history

    try:
        result = get_session_history(limit=limit)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)[:200]})


async def agentguard_clear_usage_stats(
    before: str | None = None,
    project: str | None = None,
    confirm: bool = False,
) -> str:
    """Clear usage statistics. Requires confirm=True to execute."""
    if not confirm:
        return json.dumps({
            "status": "dry_run",
            "message": "Set confirm=True to actually delete records. "
            "This will permanently remove usage statistics.",
        })

    from agentguard.stats.queries import clear_stats

    try:
        deleted = clear_stats(before=before, project=project)
        return json.dumps({
            "status": "success",
            "records_deleted": deleted,
            "filters": {"before": before, "project": project},
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)[:200]})


async def agentguard_export_usage_stats(
    target: str = "file",
    period: str = "week",
    webhook_url: str | None = None,
    file_path: str | None = None,
    format: str = "json",
) -> str:
    """Export usage stats to platform, webhook, or file."""
    from agentguard.stats.exporter import export_stats

    try:
        result = await export_stats(
            target=target,
            period=period,
            webhook_url=webhook_url,
            file_path=file_path,
            format=format,
        )
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)[:200]})


async def agentguard_report_compaction(
    context_before_chars: int = 0,
    context_after_chars: int = 0,
) -> str:
    """Record a context compaction event."""
    from agentguard.stats.collector import get_collector

    try:
        collector = get_collector()
        collector.record_compaction(
            context_before_chars=context_before_chars,
            context_after_chars=context_after_chars,
        )
        reduction_pct = 0.0
        if context_before_chars > 0:
            reduction_pct = round(
                (1 - context_after_chars / context_before_chars) * 100, 2,
            )
        return json.dumps({
            "status": "success",
            "session_id": collector.session_id,
            "context_before_chars": context_before_chars,
            "context_after_chars": context_after_chars,
            "reduction_pct": reduction_pct,
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)[:200]})
