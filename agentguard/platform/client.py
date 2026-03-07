"""PlatformClient — async HTTP client for the AgentGuard platform.

Handles usage telemetry reporting and marketplace / license operations.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentguard.platform.config import PlatformConfig
    from agentguard.platform.license_cache import LicenseCache

logger = logging.getLogger(__name__)


def _save_config_if_possible(config: Any) -> None:
    """Persist config to disk silently — never raises."""
    try:
        from agentguard.platform.config import save_config
        save_config(config)
    except Exception:
        logger.debug("Could not persist config after claim update", exc_info=True)



@dataclass
class UsageEventPayload:
    """A single usage event to send to the platform API."""

    event_type: str
    archetype_slug: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0
    duration_ms: int = 0
    success: bool = True
    metadata_json: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON POST body."""
        d = asdict(self)
        # Remove None values so the API uses its defaults
        return {k: v for k, v in d.items() if v is not None}


class PlatformClient:
    """Async client that reports usage telemetry to the AgentGuard platform.

    Features:
        - Batches events and flushes periodically or when batch is full.
        - Fire-and-forget: never blocks the pipeline on telemetry failures.
        - Gracefully degrades when no API key is configured or platform is unreachable.

    Usage::

        from agentguard.platform import PlatformClient, load_config

        config = load_config()
        client = PlatformClient(config)

        # Within an async context:
        await client.track(UsageEventPayload(event_type="generation", ...))
        await client.flush()  # Call at pipeline end
        await client.close()  # Clean up HTTP resources
    """

    def __init__(self, config: PlatformConfig) -> None:
        self._config = config
        self._buffer: list[UsageEventPayload] = []
        self._http: Any = None  # Lazy httpx.AsyncClient
        self._flush_task: asyncio.Task[None] | None = None
        self._closed = False
        self._license_cache: LicenseCache | None = None

    @property
    def is_configured(self) -> bool:
        """True if the client has an API key and is enabled."""
        return self._config.is_configured

    # ── Public API ────────────────────────────────────────────────

    async def track(self, event: UsageEventPayload) -> None:
        """Buffer a usage event for later transmission.

        If the buffer exceeds ``batch_size``, it is flushed immediately.
        """
        if not self.is_configured:
            return

        self._buffer.append(event)

        if len(self._buffer) >= self._config.batch_size:
            await self.flush()

    async def track_many(self, events: list[UsageEventPayload]) -> None:
        """Buffer multiple events at once."""
        if not self.is_configured:
            return
        self._buffer.extend(events)
        if len(self._buffer) >= self._config.batch_size:
            await self.flush()

    async def flush(self) -> None:
        """Send all buffered events to the platform API.

        Silent on failure — logs a warning but never raises.
        """
        if not self._buffer or not self.is_configured:
            return

        events = self._buffer[:]
        self._buffer.clear()

        try:
            client = await self._get_http()
            url = f"{self._config.platform_url.rstrip('/')}/api/analytics/events/batch"

            payload = [e.to_dict() for e in events]
            response = await client.post(url, json=payload)

            if response.status_code in (200, 201):
                logger.debug(
                    "Platform: reported %d events (%d bytes)",
                    len(events),
                    len(response.content),
                )
            else:
                logger.warning(
                    "Platform: event reporting failed (HTTP %d): %s",
                    response.status_code,
                    response.text[:200],
                )
        except Exception:
            logger.debug(
                "Platform: failed to report %d events (platform unreachable)",
                len(events),
                exc_info=True,
            )

    async def close(self) -> None:
        """Flush remaining events and close the HTTP client."""
        if self._closed:
            return
        self._closed = True

        # Cancel periodic flush task
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flush_task

        # Final flush
        await self.flush()

        # Close httpx client
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def start_background_flush(self) -> None:
        """Start a background task that periodically flushes the buffer.

        Call this at pipeline start; the task is cancelled on ``close()``.
        """
        if not self.is_configured:
            return
        if self._flush_task and not self._flush_task.done():
            return

        async def _periodic_flush() -> None:
            while True:
                await asyncio.sleep(self._config.flush_interval_seconds)
                await self.flush()

        try:
            loop = asyncio.get_running_loop()
            self._flush_task = loop.create_task(_periodic_flush())
        except RuntimeError:
            # No running event loop — caller must flush manually
            pass

    # ── Convenience: build events from trace data ──────────────

    def build_generation_event(
        self,
        archetype: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        duration_ms: int,
        success: bool = True,
        level: str | None = None,
        files_count: int | None = None,
    ) -> UsageEventPayload:
        """Build a generation usage event from pipeline data."""
        provider, _, model_name = model.partition("/")
        metadata: dict[str, Any] = {}
        if level:
            metadata["level"] = level
        if files_count is not None:
            metadata["files_count"] = files_count

        return UsageEventPayload(
            event_type="generation",
            archetype_slug=archetype,
            llm_provider=provider or None,
            llm_model=model_name or model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=cost,
            duration_ms=duration_ms,
            success=success,
            metadata_json=metadata or None,
        )

    def build_validation_event(
        self,
        archetype: str,
        duration_ms: int,
        passed: bool,
        fixes: int = 0,
        errors: int = 0,
    ) -> UsageEventPayload:
        """Build a validation usage event."""
        return UsageEventPayload(
            event_type="validation",
            archetype_slug=archetype,
            duration_ms=duration_ms,
            success=passed,
            metadata_json={"fixes": fixes, "errors": errors} if (fixes or errors) else None,
        )

    def build_challenge_event(
        self,
        archetype: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        duration_ms: int,
        passed: bool,
        rework_attempts: int = 0,
    ) -> UsageEventPayload:
        """Build a self-challenge usage event."""
        provider, _, model_name = model.partition("/")
        return UsageEventPayload(
            event_type="challenge",
            archetype_slug=archetype,
            llm_provider=provider or None,
            llm_model=model_name or model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=cost,
            duration_ms=duration_ms,
            success=passed,
            metadata_json={"rework_attempts": rework_attempts} if rework_attempts else None,
        )

    # ── Marketplace ─────────────────────────────────────────────

    async def search_marketplace(
        self,
        *,
        query: str | None = None,
        category: str | None = None,
        sort: str = "popular",
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """Search / browse published marketplace archetypes.

        Returns the JSON response dict with ``items``, ``total``, ``page``,
        ``page_size`` keys.  No authentication required.
        """
        client = await self._get_http()
        url = f"{self._config.platform_url.rstrip('/')}/api/marketplace/archetypes"
        params: dict[str, Any] = {"sort": sort, "page": page, "page_size": page_size}
        if query:
            params["q"] = query
        if category:
            params["category"] = category

        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    async def get_archetype_detail(self, slug: str) -> dict[str, Any]:
        """Fetch full detail of a marketplace archetype (including YAML if authorized)."""
        client = await self._get_http()
        url = f"{self._config.platform_url.rstrip('/')}/api/marketplace/archetypes/{slug}"
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    async def download_archetype(self, slug: str) -> dict[str, Any]:
        """Download the YAML content of a marketplace archetype.

        Two-step secure flow:

        1. ``POST /engine/archetypes/{slug}/download-token``
           — verifies licence and returns a short-lived (5-min) single-use JWT.
        2. ``GET /engine/archetypes/{slug}/content?token=<token>``
           — consumes the token and returns the watermarked YAML.

        Returns a dict with ``yaml_content``, ``content_hash``, ``trust_level``,
        ``slug``, ``name``, ``version``.
        """
        client = await self._get_http()
        base = self._config.platform_url.rstrip("/")

        # Step 1: obtain a short-lived download token
        token_url = f"{base}/api/engine/archetypes/{slug}/download-token"
        token_resp = await client.post(token_url)
        token_resp.raise_for_status()
        download_token: str = token_resp.json()["download_token"]

        # Step 2: consume the token and receive watermarked YAML
        content_url = f"{base}/api/engine/archetypes/{slug}/content"
        content_resp = await client.get(content_url, params={"token": download_token})
        content_resp.raise_for_status()
        return content_resp.json()  # type: ignore[no-any-return]

    # ── License ──────────────────────────────────────────────────

    @property
    def license_cache(self) -> LicenseCache:
        """Lazy-init the local license cache."""
        if self._license_cache is None:
            from agentguard.platform.license_cache import LicenseCache as _LC

            self._license_cache = _LC()
        return self._license_cache

    async def check_license(self, slug: str, *, use_cache: bool = True) -> dict[str, Any]:
        """Check if the current user is licensed to use an archetype.

        Uses a local 24-hour cache by default to avoid excessive API calls.

        Returns ``{"slug": ..., "licensed": bool, "reason": ...}``.
        """
        # Check local cache first
        if use_cache:
            cached = self.license_cache.get(slug)
            if cached is not None:
                return {"slug": cached.slug, "licensed": cached.licensed, "reason": cached.reason}

        # Call platform API
        client = await self._get_http()
        url = f"{self._config.platform_url.rstrip('/')}/api/engine/license/{slug}"
        resp = await client.get(url)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()

        # Cache the result
        self.license_cache.set(
            slug,
            licensed=data.get("licensed", False),
            reason=data.get("reason", ""),
        )

        return data

    # ── API key validation ───────────────────────────────────────

    async def validate_api_key(self) -> dict[str, Any]:
        """Validate the configured API key against the platform.

        Returns user info dict on success (``valid``, ``user_id``,
        ``email``, ``tier``, ``name``).

        Raises ``httpx.HTTPStatusError`` on 401/403.
        """
        client = await self._get_http()
        url = f"{self._config.platform_url.rstrip('/')}/api/engine/validate"
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    # ── Claim session management ──────────────────────────────────

    async def claim_session(
        self,
        label: str = "unknown session",
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """Start a session claim on the current API key.

        Sends ``POST /engine/claim`` and stores the returned ``claim_token``
        in the local config file so every subsequent request includes it
        transparently via ``X-Claim-Token``.

        Args:
            label:  Human-readable name for this session
                    (e.g. ``socket.gethostname()``).
            force:  If ``True``, revoke any existing claim and start fresh.
                    The server returns 409 without this flag when another
                    session is active.

        Returns:
            The raw JSON response: ``{claim_token, label, expires_at}``.
        """
        # Use a temporary client without X-Claim-Token for this call
        client = await self._get_http()
        url = f"{self._config.platform_url.rstrip('/')}/api/engine/claim"
        resp = await client.post(url, json={"label": label, "force": force})
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()

        # Persist the claim token to disk
        self._config.claim_token = data["claim_token"]
        self._config.claim_expires_at = data["expires_at"]
        _save_config_if_possible(self._config)

        # Rebuild the HTTP client so it picks up the new claim token header
        if self._http is not None:
            await self._http.aclose()
            self._http = None

        return data

    async def release_session(self) -> None:
        """Release the current session claim on the API key.

        Sends ``DELETE /engine/claim`` to free the key, then clears the
        claim token from the local config file.  Idempotent if no claim exists.
        """
        if not self._config.claim_token:
            return  # Nothing to release

        try:
            client = await self._get_http()
            url = f"{self._config.platform_url.rstrip('/')}/api/engine/claim"
            resp = await client.request("DELETE", url)
            # 204 = released, 403 = token mismatch (already released elsewhere)
            if resp.status_code not in (204, 403):
                resp.raise_for_status()
        except Exception:
            logger.debug("release_session: server call failed (ignored)", exc_info=True)
        finally:
            self._config.claim_token = None
            self._config.claim_expires_at = None
            _save_config_if_possible(self._config)
            if self._http is not None:
                await self._http.aclose()
                self._http = None

    # ── Timing helper ─────────────────────────────────────────

    @staticmethod
    def timer() -> _Timer:
        """Create a simple timing context manager."""
        return _Timer()

    # ── Internal ──────────────────────────────────────────────

    async def _get_http(self) -> Any:
        """Lazy-initialize the httpx async client.

        Headers are rebuilt when the claim token changes (e.g. after
        ``claim_session`` or ``release_session``).
        """
        if self._http is None:
            try:
                import httpx
            except ImportError:
                raise ImportError(
                    "httpx is required for platform integration.\n"
                    'Install it with: pip install "rlabs-agentguard[platform]"'
                ) from None

            headers: dict[str, str] = {
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"agentguard-engine/{_get_version()}",
            }
            # Inject claim token transparently when present and locally valid
            if self._config.claim_token and self._config.has_live_claim:
                headers["X-Claim-Token"] = self._config.claim_token

            # Renew local claim_expires_at on every successful response so the
            # config file stays in sync with the server's rolling TTL.
            cfg = self._config

            async def _renew_claim_ttl(response: Any) -> None:
                if (
                    cfg.claim_token
                    and response.status_code < 300
                ):
                    from datetime import UTC, datetime, timedelta
                    cfg.claim_expires_at = (
                        datetime.now(UTC) + timedelta(hours=24)
                    ).isoformat()
                    _save_config_if_possible(cfg)

            self._http = httpx.AsyncClient(
                timeout=self._config.timeout_seconds,
                headers=headers,
                event_hooks={"response": [_renew_claim_ttl]},
            )
        return self._http

    def __del__(self) -> None:
        """Warn if not properly closed."""
        if self._http is not None and not self._closed:
            logger.warning("PlatformClient was not properly closed. Call await client.close().")


class _Timer:
    """Simple timing context manager."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> _Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> int:
        return int((self._end - self._start) * 1000)


def _get_version() -> str:
    """Get the engine version for the User-Agent header."""
    try:
        from agentguard._version import __version__
        return __version__
    except ImportError:
        return "0.1.0"
