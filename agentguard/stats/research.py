"""Research-cohort telemetry — export controlled-study data.

Opt-in. A session only participates if ``AGENTGUARD_RESEARCH_COHORT``
is set in the environment before the pipeline runs; otherwise events
carry ``research_cohort_id=NULL`` and are invisible to this module.

Module surface:

- ``get_cohort_events(cohort_id, limit=10000)`` — read cohort-tagged
  rows from the local stats DB.
- ``anonymise_events(events)`` — hash paths, drop free-text fields
  that might carry specification content.
- ``upload_cohort(cohort_id, endpoint=None)`` — fetch, anonymise, and
  POST to the research endpoint. Returns a summary dict.

The module does not run on its own. It is triggered by the user via
``python -m agentguard research upload`` (see ``agentguard.__main__``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

from agentguard._version import __version__
from agentguard.stats.db import get_connection

logger = logging.getLogger(__name__)

_DEFAULT_RESEARCH_ENDPOINT = "https://api.agentguard.rlabs.cl/v1/research/events"
_ANON_SALT = "agentguard-research"


# ── Read ────────────────────────────────────────────────────────────


def get_cohort_events(cohort_id: str, limit: int = 10_000) -> list[dict[str, Any]]:
    """Return tool_events rows tagged with *cohort_id*, newest first."""
    if not cohort_id:
        return []
    try:
        conn = get_connection()
        try:
            rows = conn.execute(
                """SELECT * FROM tool_events
                   WHERE research_cohort_id = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (cohort_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
    except Exception:
        logger.exception("Failed to query cohort events for %s", cohort_id)
        return []


# ── Anonymise ───────────────────────────────────────────────────────


def _hash12(value: str | None) -> str | None:
    """Return a 12-char salted SHA256 digest of *value*, or ``None``.

    Same input always hashes to the same output for the same lib
    install, so researchers can distinguish projects without learning
    which project.
    """
    if value is None:
        return None
    digest = hashlib.sha256((_ANON_SALT + str(value)).encode()).hexdigest()
    return digest[:12]


def anonymise_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip or hash identifying fields from cohort events.

    Rules applied per event:

    - ``project_path`` and ``project_name`` are replaced with a
      12-char hash; two events on the same project stay linkable but
      the project itself is not recoverable.
    - ``parameters_summary`` is dropped entirely. It can embed parts
      of the user's specification text, which may be proprietary.
    - ``error_message`` is kept but truncated to 200 characters; if
      the text happens to embed a path it remains but is already
      limited in scope.
    - All other fields (timings, token counts, archetype name,
      tool name, status) are preserved verbatim — they are the
      research signal.
    """
    out: list[dict[str, Any]] = []
    for e in events:
        row = dict(e)
        row["project_path"] = _hash12(row.get("project_path"))
        row["project_name"] = _hash12(row.get("project_name"))
        row.pop("parameters_summary", None)
        if row.get("error_message"):
            row["error_message"] = str(row["error_message"])[:200]
        out.append(row)
    return out


# ── Upload ──────────────────────────────────────────────────────────


def _research_endpoint() -> str:
    return os.environ.get(
        "AGENTGUARD_RESEARCH_ENDPOINT", _DEFAULT_RESEARCH_ENDPOINT,
    )


def upload_cohort(
    cohort_id: str,
    endpoint: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Fetch cohort events, anonymise, and POST to the research endpoint.

    Parameters
    ----------
    cohort_id : str
        Research cohort identifier. Required.
    endpoint : str or None
        Override the endpoint URL. Defaults to env
        ``AGENTGUARD_RESEARCH_ENDPOINT`` or the rlabs production URL.
    dry_run : bool
        If ``True``, return the anonymised payload without sending.
        Useful for the user to inspect exactly what will leave their
        machine before allowing the upload.

    Returns
    -------
    dict
        ``{records_exported, destination, status, cohort_id[, error, payload]}``.
        ``payload`` is included only when ``dry_run=True``.
    """
    if not cohort_id:
        return {
            "records_exported": 0,
            "destination": endpoint or _research_endpoint(),
            "status": "error",
            "error": "cohort_id is required",
            "cohort_id": cohort_id,
        }

    events = get_cohort_events(cohort_id)
    anon = anonymise_events(events)
    payload = {
        "cohort_id": cohort_id,
        "client_version": __version__,
        "uploaded_at": datetime.now(UTC).isoformat(),
        "event_count": len(anon),
        "events": anon,
    }

    url = endpoint or _research_endpoint()
    if dry_run:
        return {
            "records_exported": len(anon),
            "destination": url,
            "status": "dry_run",
            "cohort_id": cohort_id,
            "payload": payload,
        }

    return _post_payload(url, payload, len(anon), cohort_id)


def _post_payload(
    url: str,
    payload: dict[str, Any],
    event_count: int,
    cohort_id: str,
) -> dict[str, Any]:
    """POST ``payload`` as JSON to ``url`` and map the result."""
    import urllib.error
    import urllib.request

    from agentguard._http import make_request

    data = json.dumps(payload, default=str).encode()
    req = make_request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": f"agentguard/{__version__} research-upload",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status_code = resp.status
            return {
                "records_exported": event_count,
                "destination": url,
                "status": "success" if status_code < 300 else f"http_{status_code}",
                "cohort_id": cohort_id,
            }
    except urllib.error.HTTPError as e:
        return {
            "records_exported": 0,
            "destination": url,
            "status": "error",
            "error": f"HTTP {e.code}: {e.reason}",
            "cohort_id": cohort_id,
        }
    except Exception as e:
        return {
            "records_exported": 0,
            "destination": url,
            "status": "error",
            "error": str(e)[:200],
            "cohort_id": cohort_id,
        }
