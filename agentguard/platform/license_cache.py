"""License cache — local 24-hour cache for marketplace license checks.

Avoids hitting the platform API on every archetype load.  Entries are
stored as JSON in ``~/.agentguard/cache/licenses.json``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default cache location
CACHE_DIR = Path.home() / ".agentguard" / "cache"
CACHE_FILE = CACHE_DIR / "licenses.json"

# Default TTL: 24 hours
DEFAULT_TTL_SECONDS = 86_400


@dataclass
class LicenseEntry:
    """A single cached license result."""

    slug: str
    licensed: bool
    reason: str  # "free", "purchased", "author", "not_purchased"
    checked_at: float  # epoch seconds

    def is_expired(self, ttl: float = DEFAULT_TTL_SECONDS) -> bool:
        """True if the entry is older than *ttl* seconds."""
        return (time.time() - self.checked_at) > ttl

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "licensed": self.licensed,
            "reason": self.reason,
            "checked_at": self.checked_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LicenseEntry:
        return cls(
            slug=data["slug"],
            licensed=data["licensed"],
            reason=data.get("reason", "unknown"),
            checked_at=data.get("checked_at", 0.0),
        )


class LicenseCache:
    """Persistent file-backed license cache with configurable TTL.

    Usage::

        cache = LicenseCache()
        entry = cache.get("my-archetype")
        if entry is None:
            # Call platform API …
            cache.set("my-archetype", licensed=True, reason="purchased")

    Args:
        cache_file: Path to the JSON cache file.
        ttl: Time-to-live in seconds (default 24h).
    """

    def __init__(
        self,
        cache_file: Path | None = None,
        ttl: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._file = cache_file or CACHE_FILE
        self._ttl = ttl
        self._entries: dict[str, LicenseEntry] = {}
        self._loaded = False

    def get(self, slug: str) -> LicenseEntry | None:
        """Return a valid (non-expired) cache entry, or ``None``."""
        self._ensure_loaded()
        entry = self._entries.get(slug)
        if entry is None or entry.is_expired(self._ttl):
            return None
        return entry

    def set(self, slug: str, *, licensed: bool, reason: str = "") -> None:
        """Store a license check result."""
        self._ensure_loaded()
        self._entries[slug] = LicenseEntry(
            slug=slug,
            licensed=licensed,
            reason=reason,
            checked_at=time.time(),
        )
        self._persist()

    def remove(self, slug: str) -> None:
        """Remove a single entry from the cache."""
        self._ensure_loaded()
        self._entries.pop(slug, None)
        self._persist()

    def clear(self) -> None:
        """Remove all cached entries."""
        self._entries.clear()
        self._persist()

    def list_entries(self) -> list[LicenseEntry]:
        """Return all non-expired entries."""
        self._ensure_loaded()
        return [e for e in self._entries.values() if not e.is_expired(self._ttl)]

    # ── Persistence ───────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._file.exists():
            return
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            for item in data.get("licenses", []):
                entry = LicenseEntry.from_dict(item)
                self._entries[entry.slug] = entry
        except Exception:
            logger.debug("Failed to load license cache from %s", self._file, exc_info=True)

    def _persist(self) -> None:
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "licenses": [e.to_dict() for e in self._entries.values()],
            }
            self._file.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("Failed to persist license cache to %s", self._file, exc_info=True)
