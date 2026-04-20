"""Export usage stats to platform, webhook, or file."""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

from agentguard.stats.queries import get_raw_events, get_usage_summary

logger = logging.getLogger(__name__)


async def export_stats(
    target: str,
    period: str = "week",
    webhook_url: str | None = None,
    file_path: str | None = None,
    format: str = "json",
) -> dict[str, Any]:
    """Export usage stats to platform, webhook, or local file.

    Parameters
    ----------
    target : str
        ``'platform'``, ``'webhook'``, or ``'file'``.
    period : str
        Time period filter (``'today'``, ``'week'``, ``'month'``, ``'all'``).
    webhook_url : str or None
        Required when *target* is ``'webhook'``.
    file_path : str or None
        Destination path when *target* is ``'file'``.  Defaults to
        ``~/.agentguard/usage_export_{timestamp}.{format}``.
    format : str
        ``'json'`` or ``'csv'`` (only for file target).

    Returns
    -------
    dict
        ``{ records_exported, destination, status }``
    """
    summary = get_usage_summary(period=period)
    events = get_raw_events(limit=10000)

    payload = {
        "exported_at": datetime.now(UTC).isoformat(),
        "period": period,
        "summary": summary,
        "events": events,
    }

    if target == "platform":
        return await _export_platform(payload, events)
    elif target == "webhook":
        return await _export_webhook(payload, events, webhook_url)
    elif target == "webhooks":
        return await _export_configured_webhooks(payload, events)
    elif target == "all":
        results = []
        results.append(await _export_platform(payload, events))
        results.append(await _export_configured_webhooks(payload, events))
        total = sum(r.get("records_exported", 0) for r in results)
        errors = [r["error"] for r in results if r.get("error")]
        return {
            "records_exported": total,
            "destinations": [r.get("destination", "") for r in results],
            "status": "success" if not errors else "partial",
            **({"errors": errors} if errors else {}),
        }
    elif target == "file":
        return _export_file(payload, events, file_path, format)
    else:
        return {
            "records_exported": 0,
            "destination": target,
            "status": "error",
            "error": f"Unknown target: {target}. Use 'platform', 'webhook', 'webhooks', 'all', or 'file'.",
        }


async def _export_platform(
    payload: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """POST usage report to the AgentGuard platform."""
    import urllib.error
    import urllib.request

    from agentguard._http import make_request

    api_url = os.environ.get("AGENTGUARD_API_URL", "https://api.agentguard.dev")
    api_key = os.environ.get("AGENTGUARD_API_KEY", "")

    if not api_key:
        return {
            "records_exported": 0,
            "destination": "platform",
            "status": "error",
            "error": "AGENTGUARD_API_KEY environment variable not set.",
        }

    url = f"{api_url}/api/usage/report"
    data = json.dumps(payload, default=str).encode()
    req = make_request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status_code = resp.status
            return {
                "records_exported": len(events),
                "destination": url,
                "status": "success" if status_code < 300 else f"http_{status_code}",
            }
    except urllib.error.HTTPError as e:
        return {
            "records_exported": 0,
            "destination": url,
            "status": "error",
            "error": f"HTTP {e.code}: {e.reason}",
        }
    except Exception as e:
        return {
            "records_exported": 0,
            "destination": url,
            "status": "error",
            "error": str(e)[:200],
        }


async def _export_configured_webhooks(
    payload: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """POST usage stats to all URLs in AGENTGUARD_STATS_WEBHOOKS env var."""
    urls_str = os.environ.get("AGENTGUARD_STATS_WEBHOOKS", "")
    if not urls_str:
        return {
            "records_exported": 0,
            "destination": "webhooks",
            "status": "error",
            "error": "AGENTGUARD_STATS_WEBHOOKS not configured. Set comma-separated URLs in MCP env config.",
        }

    urls = [u.strip() for u in urls_str.split(",") if u.strip()]
    results = []
    for url in urls:
        r = await _export_webhook(payload, events, url)
        results.append(r)

    total = sum(r.get("records_exported", 0) for r in results)
    errors = [f"{r['destination']}: {r['error']}" for r in results if r.get("error")]
    return {
        "records_exported": total,
        "destination": f"{len(urls)} webhooks",
        "status": "success" if not errors else "partial",
        **({"errors": errors} if errors else {}),
    }


async def _export_webhook(
    payload: dict[str, Any],
    events: list[dict[str, Any]],
    webhook_url: str | None,
) -> dict[str, Any]:
    """POST usage stats to a user-provided webhook URL."""
    import urllib.error
    import urllib.request

    from agentguard._http import make_request

    if not webhook_url:
        return {
            "records_exported": 0,
            "destination": "webhook",
            "status": "error",
            "error": "webhook_url is required for webhook target.",
        }

    data = json.dumps(payload, default=str).encode()
    headers: dict[str, str] = {"Content-Type": "application/json"}

    # Optional HMAC signature
    signing_secret = os.environ.get("AGENTGUARD_WEBHOOK_SECRET", "")
    if signing_secret:
        sig = hmac.new(signing_secret.encode(), data, hashlib.sha256).hexdigest()
        headers["X-AgentGuard-Signature"] = f"sha256={sig}"

    req = make_request(
        webhook_url,
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status_code = resp.status
            return {
                "records_exported": len(events),
                "destination": webhook_url,
                "status": "success" if status_code < 300 else f"http_{status_code}",
            }
    except urllib.error.HTTPError as e:
        return {
            "records_exported": 0,
            "destination": webhook_url,
            "status": "error",
            "error": f"HTTP {e.code}: {e.reason}",
        }
    except Exception as e:
        return {
            "records_exported": 0,
            "destination": webhook_url,
            "status": "error",
            "error": str(e)[:200],
        }


def _export_file(
    payload: dict[str, Any],
    events: list[dict[str, Any]],
    file_path: str | None,
    format: str,
) -> dict[str, Any]:
    """Write usage stats to a local file (JSON or CSV)."""
    if not file_path:
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        ext = "csv" if format == "csv" else "json"
        file_path = os.path.join(
            os.path.expanduser("~"), ".agentguard", f"usage_export_{ts}.{ext}",
        )

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        if format == "csv":
            _write_csv(events, file_path)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)

        return {
            "records_exported": len(events),
            "destination": file_path,
            "status": "success",
        }
    except Exception as e:
        return {
            "records_exported": 0,
            "destination": file_path,
            "status": "error",
            "error": str(e)[:200],
        }


def _write_csv(events: list[dict[str, Any]], file_path: str) -> None:
    """Write events to a CSV file."""
    if not events:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(events[0].keys())
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)
