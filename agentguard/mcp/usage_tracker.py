"""MCPUsageTracker — lightweight singleton that records MCP tool-call events.

Sends one ``mcp_tool`` analytics event per tool call to the AgentGuard platform.
Transport is fire-and-forget in a daemon thread, so it never blocks tool responses.

Flush triggers (whichever fires first):

- **Inactivity**: no new event for ≥ 30 s   → flush
- **Periodic**:   every 5 min               → flush
- **Terminal**:   ``benchmark_evaluate``    → force flush immediately
- **Process exit**: ``atexit``              → best-effort sync flush
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_INACTIVITY_SECONDS: float = 30.0
_PERIODIC_SECONDS: float = 300.0  # 5 minutes


@dataclass
class _ToolEvent:
    tool: str
    archetype_slug: str | None
    success: bool
    duration_ms: int
    timestamp: float = field(default_factory=time.time)


class MCPUsageTracker:
    """Thread-safe buffer that batches MCP tool-call events and flushes them
    to ``POST /api/analytics/events/batch``.

    Uses only stdlib (``urllib``) — no extra dependencies required.

    Credentials are read from environment variables:

    - ``AGENTGUARD_API_KEY``       — required; reporting is no-op without it
    - ``AGENTGUARD_PLATFORM_URL``  — defaults to ``https://api.agentguard.dev``
    """

    def __init__(self) -> None:
        self._api_key = os.environ.get("AGENTGUARD_API_KEY", "")
        self._api_url = os.environ.get(
            "AGENTGUARD_PLATFORM_URL",
            os.environ.get("AGENTGUARD_API_URL", "https://api.agentguard.dev"),
        ).rstrip("/")
        self._buffer: list[_ToolEvent] = []
        self._lock = threading.Lock()
        self._inactivity_timer: threading.Timer | None = None
        self._periodic_timer: threading.Timer | None = None

        if self._api_key:
            self._start_periodic()
            atexit.register(self._flush_sync)

    @property
    def is_configured(self) -> bool:
        """True when an API key is present."""
        return bool(self._api_key)

    # ── Public API ──────────────────────────────────────────────────────────

    def track(
        self,
        tool: str,
        archetype_slug: str | None,
        success: bool,
        duration_ms: int,
    ) -> None:
        """Buffer one event and reset the inactivity flush timer."""
        if not self.is_configured:
            return
        event = _ToolEvent(
            tool=tool,
            archetype_slug=archetype_slug,
            success=success,
            duration_ms=duration_ms,
        )
        with self._lock:
            self._buffer.append(event)
        self._reset_inactivity_timer()

    def force_flush(self) -> None:
        """Flush immediately — call after terminal events like ``benchmark_evaluate``."""
        self._cancel_inactivity_timer()
        self._flush_sync()

    # ── Timer management ────────────────────────────────────────────────────

    def _reset_inactivity_timer(self) -> None:
        self._cancel_inactivity_timer()
        t = threading.Timer(_INACTIVITY_SECONDS, self._flush_sync)
        t.daemon = True
        t.start()
        self._inactivity_timer = t

    def _cancel_inactivity_timer(self) -> None:
        if self._inactivity_timer is not None:
            self._inactivity_timer.cancel()
            self._inactivity_timer = None

    def _start_periodic(self) -> None:
        def _tick() -> None:
            self._flush_sync()
            self._start_periodic()  # reschedule

        t = threading.Timer(_PERIODIC_SECONDS, _tick)
        t.daemon = True
        t.start()
        self._periodic_timer = t

    # ── HTTP flush ──────────────────────────────────────────────────────────

    def _flush_sync(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            events = self._buffer[:]
            self._buffer.clear()

        payload: list[dict[str, Any]] = [
            {
                "event_type": "mcp_tool",
                "archetype_slug": e.archetype_slug,
                "duration_ms": e.duration_ms,
                "success": e.success,
                "metadata_json": {"tool": e.tool},
            }
            for e in events
        ]

        try:
            body = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self._api_url}/api/analytics/events/batch",
                data=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                logger.debug(
                    "MCPUsageTracker: reported %d events (HTTP %d)",
                    len(events),
                    resp.status,
                )
        except Exception:
            logger.debug(
                "MCPUsageTracker: failed to report %d events",
                len(events),
                exc_info=True,
            )


# ── Singleton ────────────────────────────────────────────────────────────────

_tracker: MCPUsageTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker() -> MCPUsageTracker:
    """Return the process-scoped :class:`MCPUsageTracker` singleton."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = MCPUsageTracker()
    return _tracker
