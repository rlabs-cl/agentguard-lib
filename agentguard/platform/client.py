"""PlatformClient — async HTTP client for reporting usage to the AgentGuard platform."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentguard.platform.config import PlatformConfig

logger = logging.getLogger(__name__)


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

    # ── Timing helper ─────────────────────────────────────────

    @staticmethod
    def timer() -> _Timer:
        """Create a simple timing context manager."""
        return _Timer()

    # ── Internal ──────────────────────────────────────────────

    async def _get_http(self) -> Any:
        """Lazy-initialize the httpx async client."""
        if self._http is None:
            try:
                import httpx
            except ImportError:
                raise ImportError(
                    "httpx is required for platform integration.\n"
                    'Install it with: pip install "rlabs-agentguard[platform]"'
                ) from None

            self._http = httpx.AsyncClient(
                timeout=self._config.timeout_seconds,
                headers={
                    "Authorization": f"Bearer {self._config.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": f"agentguard-engine/{_get_version()}",
                },
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
