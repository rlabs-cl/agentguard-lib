"""Platform configuration — persistent settings for API key and platform URL."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default platform URL (staging for now, production later)
DEFAULT_PLATFORM_URL = "https://agentguard-api-staging-zsyap4psrq-tl.a.run.app"

# Config directory and file
CONFIG_DIR = Path.home() / ".agentguard"
CONFIG_FILE = CONFIG_DIR / "config.yaml"


@dataclass
class PlatformConfig:
    """Configuration for platform connectivity."""

    api_key: str | None = None
    platform_url: str = DEFAULT_PLATFORM_URL
    enabled: bool = True
    batch_size: int = 50
    flush_interval_seconds: float = 30.0
    timeout_seconds: float = 10.0
    # ── Session claim ────────────────────────────────────────────
    # Ephemeral token issued by POST /engine/claim.  Sent as X-Claim-Token.
    claim_token: str | None = None
    # UTC ISO-8601 string — used to detect stale claims without an HTTP round-trip.
    claim_expires_at: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_configured(self) -> bool:
        """True if an API key is set and reporting is enabled."""
        return bool(self.api_key) and self.enabled

    @property
    def has_live_claim(self) -> bool:
        """True if the local claim token appears still valid (expiry in the future).

        This is a client-side optimistic check.  The server is authoritative.
        """
        if not self.claim_token or not self.claim_expires_at:
            return False
        try:
            from datetime import UTC

            expires = datetime.fromisoformat(self.claim_expires_at)
            if expires.tzinfo is None:
                from datetime import timezone
                expires = expires.replace(tzinfo=timezone.utc)
            return expires > datetime.now(UTC)
        except ValueError:
            return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for YAML storage."""
        d: dict[str, Any] = {}
        if self.api_key:
            d["api_key"] = self.api_key
        if self.platform_url != DEFAULT_PLATFORM_URL:
            d["platform_url"] = self.platform_url
        if not self.enabled:
            d["enabled"] = False
        if self.batch_size != 50:
            d["batch_size"] = self.batch_size
        if self.flush_interval_seconds != 30.0:
            d["flush_interval_seconds"] = self.flush_interval_seconds
        if self.timeout_seconds != 10.0:
            d["timeout_seconds"] = self.timeout_seconds
        if self.claim_token:
            d["claim_token"] = self.claim_token
        if self.claim_expires_at:
            d["claim_expires_at"] = self.claim_expires_at
        if self.extra:
            d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlatformConfig:
        """Create from a dict (from YAML)."""
        known_keys = {
            "api_key", "platform_url", "enabled",
            "batch_size", "flush_interval_seconds", "timeout_seconds",
            "claim_token", "claim_expires_at",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            api_key=data.get("api_key"),
            platform_url=data.get("platform_url", DEFAULT_PLATFORM_URL),
            enabled=data.get("enabled", True),
            batch_size=data.get("batch_size", 50),
            flush_interval_seconds=data.get("flush_interval_seconds", 30.0),
            timeout_seconds=data.get("timeout_seconds", 10.0),
            claim_token=data.get("claim_token"),
            claim_expires_at=data.get("claim_expires_at"),
            extra=extra,
        )


def load_config(config_path: Path | None = None) -> PlatformConfig:
    """Load platform config from disk.

    Resolution order:
    1. Explicit ``config_path`` argument.
    2. ``AGENTGUARD_CONFIG`` env var.
    3. ``~/.agentguard/config.yaml``.

    Returns defaults if no config file exists.
    """
    import os

    if config_path is None:
        env_path = os.environ.get("AGENTGUARD_CONFIG")
        config_path = Path(env_path) if env_path else CONFIG_FILE

    if not config_path.exists():
        # Also check for env-var overrides even without a config file
        api_key = os.environ.get("AGENTGUARD_PLATFORM_KEY") or os.environ.get("AGENTGUARD_API_KEY")
        url = os.environ.get("AGENTGUARD_PLATFORM_URL")
        cfg = PlatformConfig(api_key=api_key)
        if url:
            cfg.platform_url = url
        return cfg

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        cfg = PlatformConfig.from_dict(data)
    except Exception:
        logger.warning("Failed to load config from %s, using defaults", config_path)
        cfg = PlatformConfig()

    # Env vars override file values
    import os as _os

    env_key = _os.environ.get("AGENTGUARD_PLATFORM_KEY") or _os.environ.get("AGENTGUARD_API_KEY")
    if env_key:
        cfg.api_key = env_key
    env_url = _os.environ.get("AGENTGUARD_PLATFORM_URL")
    if env_url:
        cfg.platform_url = env_url

    return cfg


def save_config(config: PlatformConfig, config_path: Path | None = None) -> Path:
    """Persist platform config to disk.

    Args:
        config: The configuration to save.
        config_path: Explicit path; defaults to ``~/.agentguard/config.yaml``.

    Returns:
        The path the config was written to.
    """
    if config_path is None:
        config_path = CONFIG_FILE

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.dump(config.to_dict(), default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    logger.info("Platform config saved to %s", config_path)
    return config_path
