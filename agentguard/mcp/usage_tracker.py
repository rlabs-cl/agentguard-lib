"""MCPUsageTracker — lightweight singleton that records MCP tool-call events.

Sends one ``mcp_tool`` analytics event per tool call to the AgentGuard platform.
Transport is fire-and-forget in a daemon thread, so it never blocks tool responses.

**Store-and-forward**: events that cannot be delivered (no network, platform
unreachable) are persisted to ``~/.agentguard/pending_events.jsonl`` and
retried on the next flush.  The combined in-memory + on-disk queue is capped
at ``_MAX_STORED_EVENTS`` (1 000) total; oldest events are dropped first when
the cap is reached so new tool calls are never blocked.

Flush triggers (whichever fires first):

- **Inactivity**: no new event for ≥ 30 s   → flush
- **Periodic**:   every 5 min               → flush
- **Terminal**:   ``benchmark_evaluate``    → force flush immediately
- **Process exit**: ``atexit``              → best-effort sync flush
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_INACTIVITY_SECONDS: float = 30.0
_PERIODIC_SECONDS: float = 300.0   # 5 minutes
_MAX_STORED_EVENTS: int = 1_000    # cap across in-memory buffer + disk store
_STORE_PATH = Path.home() / ".agentguard" / "pending_events.jsonl"


@dataclass
class _ToolEvent:
    tool: str
    archetype_slug: str | None
    success: bool
    duration_ms: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> _ToolEvent:
        return _ToolEvent(
            tool=d["tool"],
            archetype_slug=d.get("archetype_slug"),
            success=bool(d.get("success", True)),
            duration_ms=int(d.get("duration_ms", 0)),
            timestamp=float(d.get("timestamp", time.time())),
        )


def _event_to_payload(e: _ToolEvent) -> dict[str, Any]:
    return {
        "event_type": "mcp_tool",
        "archetype_slug": e.archetype_slug,
        "duration_ms": e.duration_ms,
        "success": e.success,
        "metadata_json": {"tool": e.tool, "timestamp": e.timestamp},
    }


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
        """Buffer one event and reset the inactivity flush timer.

        Silently drops the oldest event when the combined in-memory + on-disk
        queue would exceed ``_MAX_STORED_EVENTS``.
        """
        if not self.is_configured:
            return
        event = _ToolEvent(
            tool=tool,
            archetype_slug=archetype_slug,
            success=success,
            duration_ms=duration_ms,
        )
        with self._lock:
            disk_count = self._disk_event_count()
            total = len(self._buffer) + disk_count
            if total >= _MAX_STORED_EVENTS:
                # Drop oldest: prefer trimming from disk first (oldest by design)
                if disk_count > 0:
                    self._drop_oldest_disk_event()
                elif self._buffer:
                    self._buffer.pop(0)
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

    # ── Disk store helpers ──────────────────────────────────────────────────

    def _disk_event_count(self) -> int:
        """Return number of events persisted on disk (no lock assumed by caller)."""
        try:
            if not _STORE_PATH.exists():
                return 0
            return sum(1 for ln in _STORE_PATH.read_text(encoding="utf-8").splitlines() if ln.strip())
        except Exception:
            return 0

    def _drop_oldest_disk_event(self) -> None:
        """Remove the first (oldest) line from the disk store."""
        try:
            lines = [ln for ln in _STORE_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                _STORE_PATH.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
        except Exception:
            pass

    def _load_pending(self) -> list[_ToolEvent]:
        """Read all persisted events from disk and clear the file."""
        try:
            if not _STORE_PATH.exists():
                return []
            lines = [ln.strip() for ln in _STORE_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
            events = []
            for ln in lines:
                with contextlib.suppress(Exception):
                    events.append(_ToolEvent.from_dict(json.loads(ln)))
            _STORE_PATH.write_text("", encoding="utf-8")
            return events
        except Exception:
            return []

    def _persist(self, events: list[_ToolEvent]) -> None:
        """Append events to the disk store (called on send failure)."""
        try:
            _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _STORE_PATH.open("a", encoding="utf-8") as fh:
                for e in events:
                    fh.write(json.dumps(e.to_dict()) + "\n")
        except Exception:
            logger.debug("MCPUsageTracker: could not persist %d events to disk", len(events))

    # ── HTTP flush ──────────────────────────────────────────────────────────

    def _flush_sync(self) -> None:
        with self._lock:
            # Collect in-memory buffer + any previously persisted events
            pending_disk = self._load_pending()
            all_events = pending_disk + self._buffer[:]
            self._buffer.clear()

        if not all_events:
            return

        payload = [_event_to_payload(e) for e in all_events]

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
                    len(all_events),
                    resp.status,
                )
            # Success — disk store already cleared in _load_pending()
        except Exception:
            logger.debug(
                "MCPUsageTracker: offline — persisted %d events to %s",
                len(all_events),
                _STORE_PATH,
            )
            # Network unavailable or platform unreachable — save for next flush
            self._persist(all_events)


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
