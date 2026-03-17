"""Auto-update module — checks PyPI for newer versions of rlabs-agentguard.

On import, a non-blocking background check is scheduled.  Results are cached
for 24 hours in ``~/.agentguard/update_check.json`` to avoid repeated network
calls.

Public API:
    check_for_update()   — async; returns latest version string or None.
    get_update_notice()   — sync; returns a human-readable notice or None.
    do_update()           — async; runs ``pip install --upgrade``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PYPI_URL = "https://pypi.org/pypi/rlabs-agentguard/json"
CACHE_DIR = Path.home() / ".agentguard"
CACHE_FILE = CACHE_DIR / "update_check.json"
CACHE_TTL = 86400  # 24 hours

_PACKAGE_NAME = "rlabs-agentguard"


# ── Helpers ──────────────────────────────────────────────────────────


def _get_installed_version() -> str:
    """Return the currently installed package version."""
    try:
        from importlib.metadata import version as pkg_version

        return pkg_version(_PACKAGE_NAME)
    except Exception:
        # Fallback: read from internal _version module
        try:
            from agentguard._version import __version__

            return __version__
        except Exception:
            return "0.0.0"


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse a PEP-440 version string into a tuple of ints for comparison.

    Handles common formats like "1.2.3", "1.2.3rc1", "1.2.3.dev4".
    Non-numeric suffixes are stripped so "1.2.3rc1" compares as (1, 2, 3).
    """
    parts: list[int] = []
    for segment in v.split("."):
        # Strip non-numeric suffixes (rc, dev, post, a, b)
        numeric = ""
        for ch in segment:
            if ch.isdigit():
                numeric += ch
            else:
                break
        if numeric:
            parts.append(int(numeric))
    return tuple(parts) if parts else (0,)


def _is_newer(latest: str, current: str) -> bool:
    """Return True if *latest* is strictly newer than *current*."""
    return _parse_version(latest) > _parse_version(current)


def _read_cache() -> dict[str, Any] | None:
    """Read the cached update-check result, or None if stale/missing."""
    try:
        if not CACHE_FILE.exists():
            return None
        data = json.loads(CACHE_FILE.read_text())
        if time.time() - data.get("checked_at", 0) > CACHE_TTL:
            return None
        return data
    except Exception:
        return None


def _write_cache(latest_version: str) -> None:
    """Persist the latest-version result to disk."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(
            json.dumps(
                {
                    "latest_version": latest_version,
                    "checked_at": time.time(),
                },
                indent=2,
            )
        )
    except Exception as exc:
        logger.debug("Failed to write update cache: %s", exc)


# ── Core async check ────────────────────────────────────────────────


async def check_for_update() -> str | None:
    """Check PyPI for a newer version of rlabs-agentguard.

    Returns the latest version string if an update is available, or
    ``None`` if the installed version is current (or if the check fails).

    Results are cached for 24 hours.
    """
    # Check cache first
    cached = _read_cache()
    if cached is not None:
        latest = cached.get("latest_version", "")
        current = _get_installed_version()
        if latest and _is_newer(latest, current):
            return latest
        return None

    # Fetch from PyPI
    try:
        import urllib.request

        loop = asyncio.get_running_loop()
        response_text = await loop.run_in_executor(
            None,
            lambda: urllib.request.urlopen(PYPI_URL, timeout=5).read().decode(),  # noqa: S310
        )
        data = json.loads(response_text)
        latest = data.get("info", {}).get("version", "")

        if not latest:
            return None

        _write_cache(latest)

        current = _get_installed_version()
        if _is_newer(latest, current):
            return latest
        return None

    except Exception as exc:
        logger.debug("Update check failed (non-fatal): %s", exc)
        return None


# ── Sync notice (for startup banners, etc.) ─────────────────────────


def get_update_notice() -> str | None:
    """Return a human-readable update notice, or ``None`` if up to date.

    This is a **synchronous** function that reads only from the local
    cache.  Call ``check_for_update()`` first (async) to populate the
    cache — for example during server startup.
    """
    cached = _read_cache()
    if cached is None:
        return None

    latest = cached.get("latest_version", "")
    current = _get_installed_version()

    if latest and _is_newer(latest, current):
        return (
            f"A newer version of AgentGuard is available: "
            f"{current} -> {latest}.  "
            f"Run `agentguard_update` or "
            f"`pip install --upgrade {_PACKAGE_NAME}` to update."
        )
    return None


# ── Async update action ─────────────────────────────────────────────


async def do_update() -> str:
    """Run ``pip install --upgrade rlabs-agentguard``.

    Returns a human-readable result message.
    """
    loop = asyncio.get_running_loop()

    def _run_pip() -> str:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", _PACKAGE_NAME],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                # Invalidate cache so next check re-evaluates
                with contextlib.suppress(Exception):
                    CACHE_FILE.unlink(missing_ok=True)
                return (
                    f"Successfully upgraded {_PACKAGE_NAME}. "
                    f"Restart the MCP server to use the new version.\n"
                    f"{result.stdout.strip()}"
                )
            else:
                return (
                    f"Upgrade failed (exit code {result.returncode}).\n"
                    f"stdout: {result.stdout.strip()}\n"
                    f"stderr: {result.stderr.strip()}"
                )
        except subprocess.TimeoutExpired:
            return "Upgrade timed out after 120 seconds."
        except Exception as exc:
            return f"Upgrade failed: {exc}"

    return await loop.run_in_executor(None, _run_pip)


# ── Background check on import ──────────────────────────────────────


def _schedule_background_check() -> None:
    """Schedule a non-blocking update check if there's a running event loop."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(check_for_update())
    except RuntimeError:
        # No running event loop — skip the background check.
        # It will run when check_for_update() is explicitly called.
        pass


_schedule_background_check()
