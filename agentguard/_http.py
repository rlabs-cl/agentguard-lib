"""Shared HTTP helpers for urllib-based calls.

Centralised here so every outbound request carries a recognisable
User-Agent. urllib's default UA (``Python-urllib/3.x``) is indistinguishable
from bot traffic and gets blocked by Cloudflare with a 403 code 1010 —
which the MCP previously mis-classified as "API key invalid or expired".
"""

from __future__ import annotations

import urllib.request
from typing import Any

from agentguard._version import __version__

DEFAULT_USER_AGENT = f"agentguard-lib/{__version__} (+https://agentguard.rlabs.cl)"


def make_request(
    url: str,
    *,
    data: bytes | None = None,
    method: str | None = None,
    headers: dict[str, str] | None = None,
) -> urllib.request.Request:
    """Build an urllib Request with a User-Agent header.

    Any caller-supplied ``User-Agent`` wins over the default so test fixtures
    and custom integrations can override it explicitly.
    """
    merged: dict[str, Any] = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        merged.update(headers)
    return urllib.request.Request(url, data=data, method=method, headers=merged)
