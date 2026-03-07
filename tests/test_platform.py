"""Tests for the platform integration module (Phase 5)."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from agentguard.platform.client import PlatformClient, UsageEventPayload, _Timer
from agentguard.platform.config import (
    DEFAULT_PLATFORM_URL,
    PlatformConfig,
    load_config,
    save_config,
)

if TYPE_CHECKING:
    from pathlib import Path


# ─── PlatformConfig Tests ────────────────────────────────────────


class TestPlatformConfig:
    """Tests for PlatformConfig dataclass and serialization."""

    def test_default_config(self) -> None:
        cfg = PlatformConfig()
        assert cfg.api_key is None
        assert cfg.platform_url == DEFAULT_PLATFORM_URL
        assert cfg.enabled is True
        assert cfg.batch_size == 50
        assert cfg.is_configured is False

    def test_configured_with_key(self) -> None:
        cfg = PlatformConfig(api_key="ag_test_key_123")
        assert cfg.is_configured is True

    def test_disabled_not_configured(self) -> None:
        cfg = PlatformConfig(api_key="ag_test_key_123", enabled=False)
        assert cfg.is_configured is False

    def test_to_dict_minimal(self) -> None:
        cfg = PlatformConfig()
        d = cfg.to_dict()
        # Default values should be omitted
        assert "platform_url" not in d
        assert "enabled" not in d
        assert "batch_size" not in d

    def test_to_dict_with_key(self) -> None:
        cfg = PlatformConfig(api_key="ag_xxx")
        d = cfg.to_dict()
        assert d["api_key"] == "ag_xxx"

    def test_to_dict_custom_url(self) -> None:
        cfg = PlatformConfig(platform_url="http://localhost:8000")
        d = cfg.to_dict()
        assert d["platform_url"] == "http://localhost:8000"

    def test_from_dict(self) -> None:
        data = {
            "api_key": "ag_roundtrip",
            "platform_url": "https://custom.example.com",
            "batch_size": 100,
        }
        cfg = PlatformConfig.from_dict(data)
        assert cfg.api_key == "ag_roundtrip"
        assert cfg.platform_url == "https://custom.example.com"
        assert cfg.batch_size == 100
        assert cfg.enabled is True  # default

    def test_from_dict_extra_keys(self) -> None:
        data = {"api_key": "ag_x", "custom_field": "hello"}
        cfg = PlatformConfig.from_dict(data)
        assert cfg.extra == {"custom_field": "hello"}

    def test_roundtrip_yaml(self) -> None:
        original = PlatformConfig(
            api_key="ag_round",
            platform_url="https://test.example.com",
            batch_size=25,
        )
        yaml_str = yaml.dump(original.to_dict())
        data = yaml.safe_load(yaml_str)
        restored = PlatformConfig.from_dict(data)
        assert restored.api_key == original.api_key
        assert restored.platform_url == original.platform_url
        assert restored.batch_size == original.batch_size


# ─── Config persistence ──────────────────────────────────────────


class TestConfigPersistence:
    """Tests for load_config and save_config functions."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        cfg = PlatformConfig(api_key="ag_persist_test", batch_size=10)
        save_config(cfg, config_path=config_file)

        loaded = load_config(config_path=config_file)
        assert loaded.api_key == "ag_persist_test"
        assert loaded.batch_size == 10

    def test_load_nonexistent_returns_defaults(self, tmp_path: Path) -> None:
        cfg = load_config(config_path=tmp_path / "nope.yaml")
        assert cfg.api_key is None
        assert cfg.platform_url == DEFAULT_PLATFORM_URL

    def test_env_var_overrides_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        cfg = PlatformConfig(api_key="ag_file_key")
        save_config(cfg, config_path=config_file)

        with patch.dict("os.environ", {"AGENTGUARD_PLATFORM_KEY": "ag_env_key"}):
            loaded = load_config(config_path=config_file)
            assert loaded.api_key == "ag_env_key"

    def test_env_var_url_override(self, tmp_path: Path) -> None:
        with patch.dict("os.environ", {"AGENTGUARD_PLATFORM_URL": "http://local:9000"}):
            loaded = load_config(config_path=tmp_path / "nope.yaml")
            assert loaded.platform_url == "http://local:9000"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        config_file = tmp_path / "deep" / "nested" / "config.yaml"
        cfg = PlatformConfig(api_key="ag_deep")
        save_config(cfg, config_path=config_file)
        assert config_file.exists()

    def test_corrupted_config_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(":::invalid yaml{{{", encoding="utf-8")
        cfg = load_config(config_path=config_file)
        assert cfg.api_key is None  # Falls back to defaults


# ─── UsageEventPayload Tests ─────────────────────────────────────


class TestUsageEventPayload:
    """Tests for the event payload dataclass."""

    def test_basic_event(self) -> None:
        event = UsageEventPayload(event_type="generation")
        d = event.to_dict()
        assert d["event_type"] == "generation"
        assert d["success"] is True
        assert d["input_tokens"] == 0

    def test_full_event(self) -> None:
        event = UsageEventPayload(
            event_type="generation",
            archetype_slug="api_backend",
            llm_provider="anthropic",
            llm_model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
            estimated_cost=0.025,
            duration_ms=3000,
            success=True,
            metadata_json={"level": "L1"},
        )
        d = event.to_dict()
        assert d["llm_provider"] == "anthropic"
        assert d["input_tokens"] == 1000
        assert d["metadata_json"]["level"] == "L1"

    def test_none_values_stripped(self) -> None:
        event = UsageEventPayload(event_type="test")
        d = event.to_dict()
        assert "archetype_slug" not in d
        assert "llm_provider" not in d
        assert "metadata_json" not in d


# ─── PlatformClient Tests ────────────────────────────────────────


class TestPlatformClient:
    """Tests for the PlatformClient."""

    def _make_client(self, api_key: str = "ag_test_key", **kwargs) -> PlatformClient:
        cfg = PlatformConfig(api_key=api_key, **kwargs)
        return PlatformClient(cfg)

    def _make_unconfigured_client(self) -> PlatformClient:
        cfg = PlatformConfig()  # No API key
        return PlatformClient(cfg)

    def test_is_configured(self) -> None:
        client = self._make_client()
        assert client.is_configured is True

    def test_not_configured(self) -> None:
        client = self._make_unconfigured_client()
        assert client.is_configured is False

    @pytest.mark.asyncio
    async def test_track_buffers_event(self) -> None:
        client = self._make_client()
        event = UsageEventPayload(event_type="test")
        await client.track(event)
        assert len(client._buffer) == 1

    @pytest.mark.asyncio
    async def test_track_noop_when_unconfigured(self) -> None:
        client = self._make_unconfigured_client()
        event = UsageEventPayload(event_type="test")
        await client.track(event)
        assert len(client._buffer) == 0

    @pytest.mark.asyncio
    async def test_flush_noop_when_empty(self) -> None:
        client = self._make_client()
        await client.flush()  # Should not raise

    @pytest.mark.asyncio
    async def test_flush_clears_buffer(self) -> None:
        client = self._make_client()
        client._buffer = [UsageEventPayload(event_type="test")]

        # Mock httpx
        mock_http = AsyncMock()
        mock_http.post.return_value = MagicMock(status_code=201, content=b"[]")
        client._http = mock_http

        await client.flush()
        assert len(client._buffer) == 0
        mock_http.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_sends_batch(self) -> None:
        client = self._make_client()
        events = [UsageEventPayload(event_type=f"test_{i}") for i in range(3)]
        client._buffer = events

        mock_http = AsyncMock()
        mock_http.post.return_value = MagicMock(status_code=201, content=b"[]")
        client._http = mock_http

        await client.flush()

        call_args = mock_http.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert len(payload) == 3
        assert payload[0]["event_type"] == "test_0"

    @pytest.mark.asyncio
    async def test_flush_handles_http_error(self) -> None:
        client = self._make_client()
        client._buffer = [UsageEventPayload(event_type="test")]

        mock_http = AsyncMock()
        mock_http.post.return_value = MagicMock(
            status_code=500, text="Internal Server Error", content=b""
        )
        client._http = mock_http

        # Should not raise
        await client.flush()
        assert len(client._buffer) == 0  # Buffer still cleared

    @pytest.mark.asyncio
    async def test_flush_handles_connection_error(self) -> None:
        client = self._make_client()
        client._buffer = [UsageEventPayload(event_type="test")]

        mock_http = AsyncMock()
        mock_http.post.side_effect = ConnectionError("unreachable")
        client._http = mock_http

        # Should not raise — fire-and-forget
        await client.flush()
        assert len(client._buffer) == 0

    @pytest.mark.asyncio
    async def test_auto_flush_on_batch_size(self) -> None:
        client = self._make_client(batch_size=3)

        mock_http = AsyncMock()
        mock_http.post.return_value = MagicMock(status_code=201, content=b"[]")
        client._http = mock_http

        # Track 3 events — should auto-flush
        for i in range(3):
            await client.track(UsageEventPayload(event_type=f"test_{i}"))

        mock_http.post.assert_called_once()
        assert len(client._buffer) == 0

    @pytest.mark.asyncio
    async def test_close_flushes_and_cleans_up(self) -> None:
        client = self._make_client()
        client._buffer = [UsageEventPayload(event_type="final")]

        mock_http = AsyncMock()
        mock_http.post.return_value = MagicMock(status_code=201, content=b"[]")
        mock_http.aclose = AsyncMock()
        client._http = mock_http

        await client.close()
        mock_http.post.assert_called_once()
        mock_http.aclose.assert_called_once()
        assert client._closed is True

    @pytest.mark.asyncio
    async def test_close_idempotent(self) -> None:
        client = self._make_client()
        mock_http = AsyncMock()
        mock_http.aclose = AsyncMock()
        client._http = mock_http

        await client.close()
        await client.close()  # Second close should be a no-op
        mock_http.aclose.assert_called_once()


# ─── Event Builder Tests ─────────────────────────────────────────


class TestEventBuilders:
    """Tests for convenience event builder methods."""

    def _make_client(self) -> PlatformClient:
        return PlatformClient(PlatformConfig(api_key="ag_test"))

    def test_build_generation_event(self) -> None:
        client = self._make_client()
        event = client.build_generation_event(
            archetype="api_backend",
            model="anthropic/claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
            cost=0.025,
            duration_ms=5000,
            files_count=12,
        )
        assert event.event_type == "generation"
        assert event.archetype_slug == "api_backend"
        assert event.llm_provider == "anthropic"
        assert event.llm_model == "claude-sonnet-4-20250514"
        assert event.input_tokens == 1000
        assert event.output_tokens == 500
        assert event.estimated_cost == 0.025
        assert event.duration_ms == 5000
        assert event.metadata_json is not None
        assert event.metadata_json["files_count"] == 12

    def test_build_generation_event_with_level(self) -> None:
        client = self._make_client()
        event = client.build_generation_event(
            archetype="web_app",
            model="openai/gpt-4o",
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            duration_ms=100,
            level="L1",
        )
        assert event.metadata_json is not None
        assert event.metadata_json["level"] == "L1"

    def test_build_validation_event(self) -> None:
        client = self._make_client()
        event = client.build_validation_event(
            archetype="api_backend",
            duration_ms=200,
            passed=True,
            fixes=3,
        )
        assert event.event_type == "validation"
        assert event.success is True
        assert event.metadata_json is not None
        assert event.metadata_json["fixes"] == 3

    def test_build_challenge_event(self) -> None:
        client = self._make_client()
        event = client.build_challenge_event(
            archetype="api_backend",
            model="anthropic/claude-sonnet-4-20250514",
            input_tokens=500,
            output_tokens=200,
            cost=0.01,
            duration_ms=3000,
            passed=False,
            rework_attempts=2,
        )
        assert event.event_type == "challenge"
        assert event.success is False
        assert event.metadata_json is not None
        assert event.metadata_json["rework_attempts"] == 2

    def test_build_validation_event_no_metadata(self) -> None:
        client = self._make_client()
        event = client.build_validation_event(
            archetype="script",
            duration_ms=50,
            passed=True,
        )
        # No fixes or errors → metadata should be None
        assert event.metadata_json is None


# ─── Timer Tests ─────────────────────────────────────────────────


class TestTimer:
    """Tests for the _Timer context manager."""

    def test_timer_measures_elapsed(self) -> None:
        import time

        timer = _Timer()
        with timer:
            time.sleep(0.05)
        assert timer.elapsed_ms >= 40  # Allow some margin


# ─── Pipeline integration (mock-based) ───────────────────────────


class TestPipelinePlatformIntegration:
    """Verify that Pipeline initializes platform client based on config."""

    def test_pipeline_no_platform_by_default(self) -> None:
        """Pipeline should have no platform client when nothing is configured."""
        from agentguard.pipeline import Pipeline

        with patch("agentguard.pipeline.create_llm_provider") as mock_llm:
            mock_llm.return_value = MagicMock(provider_name="mock")
            pipe = Pipeline(
                archetype="api_backend",
                llm="mock/model",
                report_usage=False,
            )
            assert pipe.platform is None

    def test_pipeline_with_platform_config(self) -> None:
        """Pipeline should create a platform client when config is provided."""
        from agentguard.pipeline import Pipeline

        cfg = PlatformConfig(api_key="ag_test_pipeline")

        with patch("agentguard.pipeline.create_llm_provider") as mock_llm:
            mock_llm.return_value = MagicMock(provider_name="mock")
            pipe = Pipeline(
                archetype="api_backend",
                llm="mock/model",
                platform_config=cfg,
                report_usage=True,
            )
            assert pipe.platform is not None
            assert pipe.platform.is_configured is True

    def test_pipeline_report_false_disables(self) -> None:
        """report_usage=False should always disable platform."""
        from agentguard.pipeline import Pipeline

        cfg = PlatformConfig(api_key="ag_key")

        with patch("agentguard.pipeline.create_llm_provider") as mock_llm:
            mock_llm.return_value = MagicMock(provider_name="mock")
            pipe = Pipeline(
                archetype="api_backend",
                llm="mock/model",
                platform_config=cfg,
                report_usage=False,
            )
            assert pipe.platform is None


class TestPlatformConfigClaim:
    """Test the client-side has_live_claim property."""

    def test_has_live_claim_true(self) -> None:
        from datetime import UTC, timedelta
        cfg = PlatformConfig(
            api_key="ag_test",
            claim_token="some-token",
            claim_expires_at=(datetime.now(UTC) + timedelta(hours=12)).isoformat(),
        )
        assert cfg.has_live_claim is True

    def test_has_live_claim_expired(self) -> None:
        from datetime import UTC, timedelta
        cfg = PlatformConfig(
            api_key="ag_test",
            claim_token="some-token",
            claim_expires_at=(datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
        )
        assert cfg.has_live_claim is False

    def test_has_live_claim_no_token(self) -> None:
        cfg = PlatformConfig(api_key="ag_test")
        assert cfg.has_live_claim is False

    def test_has_live_claim_no_expires(self) -> None:
        cfg = PlatformConfig(api_key="ag_test", claim_token="tok")
        assert cfg.has_live_claim is False

    def test_config_roundtrip_preserves_claim(self) -> None:
        from datetime import UTC, timedelta
        exp = (datetime.now(UTC) + timedelta(hours=12)).isoformat()
        cfg = PlatformConfig(
            api_key="ag_test",
            claim_token="my-claim-token",
            claim_expires_at=exp,
        )
        d = cfg.to_dict()
        assert d["claim_token"] == "my-claim-token"
        assert d["claim_expires_at"] == exp

        restored = PlatformConfig.from_dict(d)
        assert restored.claim_token == "my-claim-token"
        assert restored.claim_expires_at == exp
        assert restored.has_live_claim is True
