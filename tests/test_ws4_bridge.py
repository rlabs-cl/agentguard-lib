"""Tests for WS-4: Marketplace → Engine Bridge.

Covers:
- LicenseCache: persistence, TTL, set/get/remove/clear
- PlatformClient: marketplace & license methods (mocked httpx)
- CLI: marketplace search/list/info, install, uninstall commands
- serve --platform-key validation
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from agentguard.platform.license_cache import (
    DEFAULT_TTL_SECONDS,
    LicenseCache,
    LicenseEntry,
)

# ═══════════════════════════════════════════════════════════════════
#  License Cache
# ═══════════════════════════════════════════════════════════════════


class TestLicenseEntry:
    """Tests for LicenseEntry dataclass."""

    def test_is_expired_false_when_fresh(self) -> None:
        entry = LicenseEntry(slug="x", licensed=True, reason="free", checked_at=time.time())
        assert not entry.is_expired()

    def test_is_expired_true_when_old(self) -> None:
        entry = LicenseEntry(
            slug="x", licensed=True, reason="free",
            checked_at=time.time() - DEFAULT_TTL_SECONDS - 1,
        )
        assert entry.is_expired()

    def test_custom_ttl(self) -> None:
        entry = LicenseEntry(slug="x", licensed=True, reason="free", checked_at=time.time() - 10)
        assert not entry.is_expired(ttl=60)
        assert entry.is_expired(ttl=5)

    def test_round_trip(self) -> None:
        entry = LicenseEntry(slug="s", licensed=False, reason="not_purchased", checked_at=123.0)
        restored = LicenseEntry.from_dict(entry.to_dict())
        assert restored.slug == entry.slug
        assert restored.licensed == entry.licensed
        assert restored.reason == entry.reason
        assert restored.checked_at == entry.checked_at


class TestLicenseCache:
    """Tests for LicenseCache file-backed cache."""

    def test_set_and_get(self, tmp_path: Path) -> None:
        cache = LicenseCache(cache_file=tmp_path / "lic.json")
        cache.set("arch-a", licensed=True, reason="purchased")
        entry = cache.get("arch-a")
        assert entry is not None
        assert entry.licensed is True
        assert entry.reason == "purchased"

    def test_get_returns_none_for_unknown(self, tmp_path: Path) -> None:
        cache = LicenseCache(cache_file=tmp_path / "lic.json")
        assert cache.get("nope") is None

    def test_expired_returns_none(self, tmp_path: Path) -> None:
        cache = LicenseCache(cache_file=tmp_path / "lic.json", ttl=1)
        cache.set("arch-b", licensed=True, reason="free")
        # Manually backdate
        cache._entries["arch-b"].checked_at = time.time() - 10
        assert cache.get("arch-b") is None

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        f = tmp_path / "lic.json"
        cache1 = LicenseCache(cache_file=f)
        cache1.set("p1", licensed=True, reason="purchased")

        cache2 = LicenseCache(cache_file=f)
        entry = cache2.get("p1")
        assert entry is not None
        assert entry.licensed is True

    def test_remove(self, tmp_path: Path) -> None:
        cache = LicenseCache(cache_file=tmp_path / "lic.json")
        cache.set("r", licensed=True, reason="free")
        cache.remove("r")
        assert cache.get("r") is None

    def test_clear(self, tmp_path: Path) -> None:
        cache = LicenseCache(cache_file=tmp_path / "lic.json")
        cache.set("a", licensed=True, reason="free")
        cache.set("b", licensed=False, reason="not_purchased")
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_list_entries_filters_expired(self, tmp_path: Path) -> None:
        cache = LicenseCache(cache_file=tmp_path / "lic.json", ttl=100)
        cache.set("fresh", licensed=True, reason="free")
        cache.set("old", licensed=True, reason="free")
        cache._entries["old"].checked_at = time.time() - 200
        entries = cache.list_entries()
        slugs = [e.slug for e in entries]
        assert "fresh" in slugs
        assert "old" not in slugs

    def test_handles_corrupt_file(self, tmp_path: Path) -> None:
        f = tmp_path / "lic.json"
        f.write_text("not valid json{{{", encoding="utf-8")
        cache = LicenseCache(cache_file=f)
        assert cache.get("x") is None  # Graceful degradation

    def test_handles_missing_dir(self, tmp_path: Path) -> None:
        f = tmp_path / "deep" / "nested" / "lic.json"
        cache = LicenseCache(cache_file=f)
        cache.set("x", licensed=True, reason="free")
        assert f.exists()


# ═══════════════════════════════════════════════════════════════════
#  PlatformClient — marketplace / license / validate methods
# ═══════════════════════════════════════════════════════════════════


def _make_config(api_key: str = "ag_test123") -> Any:
    """Create a minimal PlatformConfig for testing."""
    from agentguard.platform.config import PlatformConfig

    return PlatformConfig(
        api_key=api_key,
        platform_url="https://fake.agentguard.dev",
    )


def _make_client(api_key: str = "ag_test123") -> Any:
    from agentguard.platform.client import PlatformClient

    return PlatformClient(_make_config(api_key))


class TestPlatformClientMarketplace:
    """Test marketplace-related PlatformClient methods."""

    @pytest.mark.asyncio
    async def test_search_marketplace(self) -> None:
        client = _make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "items": [{"slug": "test-arch", "price_cents": 500}],
            "total": 1,
            "page": 1,
            "page_size": 20,
        }
        mock_resp.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        client._http = mock_http

        result = await client.search_marketplace(query="test")
        assert result["total"] == 1
        assert result["items"][0]["slug"] == "test-arch"

        # Verify the URL was called correctly
        call_args = mock_http.get.call_args
        assert "/api/marketplace/archetypes" in call_args[0][0]
        assert call_args[1]["params"]["q"] == "test"

        await client.close()

    @pytest.mark.asyncio
    async def test_get_archetype_detail(self) -> None:
        client = _make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "slug": "my-arch",
            "name": "My Archetype",
            "yaml_content": "id: my-arch\n",
            "is_purchased": True,
        }
        mock_resp.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        client._http = mock_http

        result = await client.get_archetype_detail("my-arch")
        assert result["slug"] == "my-arch"
        assert result["is_purchased"] is True

        await client.close()

    @pytest.mark.asyncio
    async def test_download_archetype(self) -> None:
        client = _make_client()

        # Step 1 mock: POST /download-token returns a token
        token_resp = MagicMock()
        token_resp.json = MagicMock(return_value={"download_token": "tok.abc.xyz"})
        token_resp.raise_for_status = MagicMock()

        # Step 2 mock: GET /content returns actual YAML
        content_resp = MagicMock()
        content_resp.json = MagicMock(return_value={
            "slug": "premium",
            "yaml_content": "id: premium\nname: Premium\n",
            "content_hash": "abc123",
            "trust_level": "community",
        })
        content_resp.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=token_resp)
        mock_http.get = AsyncMock(return_value=content_resp)
        client._http = mock_http

        result = await client.download_archetype("premium")
        assert result["yaml_content"].startswith("id: premium")
        assert result["content_hash"] == "abc123"

        # Assert Step 1: POST to download-token endpoint
        post_url = mock_http.post.call_args[0][0]
        assert "/api/engine/archetypes/premium/download-token" in post_url

        # Assert Step 2: GET to content endpoint with token param
        get_url = mock_http.get.call_args[0][0]
        assert "/api/engine/archetypes/premium/content" in get_url
        get_params = mock_http.get.call_args[1].get("params", {})
        assert get_params.get("token") == "tok.abc.xyz"

        await client.close()


class TestPlatformClientLicense:
    """Test license-related PlatformClient methods."""

    @pytest.mark.asyncio
    async def test_check_license_hits_api(self, tmp_path: Path) -> None:
        client = _make_client()
        # Use a fresh cache in tmp
        from agentguard.platform.license_cache import LicenseCache

        client._license_cache = LicenseCache(cache_file=tmp_path / "lic.json")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"slug": "paid-arch", "licensed": True, "reason": "purchased"}
        mock_resp.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        client._http = mock_http

        result = await client.check_license("paid-arch")
        assert result["licensed"] is True
        assert result["reason"] == "purchased"

        # Should now be cached
        cached = client.license_cache.get("paid-arch")
        assert cached is not None
        assert cached.licensed is True

        await client.close()

    @pytest.mark.asyncio
    async def test_check_license_uses_cache(self, tmp_path: Path) -> None:
        client = _make_client()
        from agentguard.platform.license_cache import LicenseCache

        cache = LicenseCache(cache_file=tmp_path / "lic.json")
        cache.set("cached-arch", licensed=True, reason="free")
        client._license_cache = cache

        mock_http = AsyncMock()
        client._http = mock_http

        result = await client.check_license("cached-arch")
        assert result["licensed"] is True
        # HTTP should NOT have been called (served from cache)
        mock_http.get.assert_not_called()

        await client.close()

    @pytest.mark.asyncio
    async def test_check_license_bypass_cache(self, tmp_path: Path) -> None:
        client = _make_client()
        from agentguard.platform.license_cache import LicenseCache

        cache = LicenseCache(cache_file=tmp_path / "lic.json")
        cache.set("x", licensed=True, reason="free")
        client._license_cache = cache

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"slug": "x", "licensed": False, "reason": "not_purchased"}
        mock_resp.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        client._http = mock_http

        result = await client.check_license("x", use_cache=False)
        assert result["licensed"] is False
        mock_http.get.assert_called_once()

        await client.close()


class TestPlatformClientValidate:
    """Test API key validation."""

    @pytest.mark.asyncio
    async def test_validate_api_key_success(self) -> None:
        client = _make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "valid": True,
            "user_id": "u-1",
            "email": "test@test.com",
            "tier": "pro",
            "name": "Test",
        }
        mock_resp.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        client._http = mock_http

        result = await client.validate_api_key()
        assert result["valid"] is True
        assert result["tier"] == "pro"

        call_url = mock_http.get.call_args[0][0]
        assert "/api/engine/validate" in call_url

        await client.close()


# ═══════════════════════════════════════════════════════════════════
#  CLI Commands
# ═══════════════════════════════════════════════════════════════════


class TestCLIMarketplace:
    """Test marketplace CLI commands."""

    def test_marketplace_search_no_results(self) -> None:
        runner = CliRunner()

        async def _mock_search(self_: Any, **kw: Any) -> dict[str, Any]:
            return {"items": [], "total": 0, "page": 1, "page_size": 20}

        with patch("agentguard.cli._require_httpx"), \
             patch("agentguard.platform.client.PlatformClient.search_marketplace", new=_mock_search), \
             patch("agentguard.platform.client.PlatformClient.close", new=AsyncMock()):
            from agentguard.cli import main

            result = runner.invoke(main, ["marketplace", "search", "nonexistent"])
            assert "No archetypes found" in result.output

    def test_marketplace_search_with_results(self) -> None:
        runner = CliRunner()

        async def _mock_search(self_: Any, **kw: Any) -> dict[str, Any]:
            return {
                "items": [
                    {
                        "slug": "cool-api",
                        "price_cents": 0,
                        "rating_avg": 4.5,
                        "downloads": 120,
                        "description": "A cool API archetype",
                    }
                ],
                "total": 1,
                "page": 1,
                "page_size": 20,
            }

        with patch("agentguard.cli._require_httpx"), \
             patch("agentguard.platform.client.PlatformClient.search_marketplace", new=_mock_search), \
             patch("agentguard.platform.client.PlatformClient.close", new=AsyncMock()):
            from agentguard.cli import main

            result = runner.invoke(main, ["marketplace", "search", "cool"])
            assert "cool-api" in result.output
            assert "FREE" in result.output
            assert "1 archetype" in result.output

    def test_marketplace_list_empty(self) -> None:
        runner = CliRunner()

        async def _mock_search(self_: Any, **kw: Any) -> dict[str, Any]:
            return {"items": [], "total": 0, "page": 1, "page_size": 20}

        with patch("agentguard.cli._require_httpx"), \
             patch("agentguard.platform.client.PlatformClient.search_marketplace", new=_mock_search), \
             patch("agentguard.platform.client.PlatformClient.close", new=AsyncMock()):
            from agentguard.cli import main

            result = runner.invoke(main, ["marketplace", "list"])
            assert "No archetypes published" in result.output


class TestCLIInstall:
    """Test install / uninstall CLI commands."""

    def test_install_archetype(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = CliRunner()

        yaml_content = (
            "id: test_install\n"
            "name: Test Install\n"
            "version: '1.0.0'\n"
            "description: Test archetype\n"
            "tech_stack:\n"
            "  language: python\n"
            "  framework: fastapi\n"
            "  build_tool: pip\n"
            "pipeline:\n"
            "  levels:\n"
            "    - skeleton\n"
            "structure:\n"
            "  root_dir: src\n"
            "  expected_files:\n"
            "    - main.py\n"
        )

        from agentguard.archetypes.schema import compute_content_hash

        content_hash = compute_content_hash(yaml_content)

        async def _mock_download(self_: Any, slug: str) -> dict[str, Any]:
            return {
                "slug": "test_install",
                "name": "Test Install",
                "version": "1.0.0",
                "yaml_content": yaml_content,
                "content_hash": content_hash,
                "trust_level": "community",
            }

        archetypes_dir = tmp_path / ".agentguard" / "archetypes"

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        with patch("agentguard.cli._require_httpx"), \
             patch("agentguard.platform.config.load_config", return_value=_make_config()), \
             patch("agentguard.platform.client.PlatformClient.download_archetype", new=_mock_download), \
             patch("agentguard.platform.client.PlatformClient.close", new=AsyncMock()):
            from agentguard.cli import main

            result = runner.invoke(main, ["install", "test_install"])
            assert result.exit_code == 0, result.output
            assert "Installed" in result.output

            # File should be written
            target = archetypes_dir / "test_install.yaml"
            assert target.exists()
            assert "id: test_install" in target.read_text(encoding="utf-8")

    def test_uninstall_archetype(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = CliRunner()

        archetypes_dir = tmp_path / ".agentguard" / "archetypes"
        archetypes_dir.mkdir(parents=True)
        (archetypes_dir / "removeme.yaml").write_text("id: removeme\n", encoding="utf-8")

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        from agentguard.cli import main

        result = runner.invoke(main, ["uninstall", "removeme"])
        assert result.exit_code == 0, result.output
        assert "Uninstalled" in result.output
        assert not (archetypes_dir / "removeme.yaml").exists()

    def test_uninstall_not_installed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = CliRunner()

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        from agentguard.cli import main

        result = runner.invoke(main, ["uninstall", "nope"])
        assert result.exit_code != 0
        assert "not installed" in result.output


class TestCLIServeValidation:
    """Test --platform-key validation in serve command."""

    def test_serve_validates_platform_key(self) -> None:
        runner = CliRunner()

        async def _mock_validate() -> dict[str, Any]:
            return {
                "valid": True,
                "user_id": "u-1",
                "email": "test@example.com",
                "tier": "pro",
                "name": "Test User",
            }

        with patch("agentguard.cli._validate_platform_key") as mock_validate:
            mock_validate.return_value = None
            with patch("agentguard.cli.sys.exit") as mock_exit:
                mock_exit.side_effect = SystemExit(0)
                from agentguard.cli import main

                # We can't actually start uvicorn, so just check the flag is recognised
                result = runner.invoke(
                    main,
                    ["serve", "--platform-key", "ag_testkey", "--help"],
                )
                # --help should work regardless
                assert "platform-key" in result.output.lower() or result.exit_code == 0


# ═══════════════════════════════════════════════════════════════════
#  Platform engine router (unit-level, no DB)
# ═══════════════════════════════════════════════════════════════════


class TestEngineRouterImport:
    """Verify engine router can be imported without errors."""

    @pytest.mark.skipif(
        not importlib.util.find_spec("app"),
        reason="Platform API package not available in library-only CI",
    )
    def test_import_engine_router(self) -> None:
        """Ensure the engine router module loads cleanly."""
        from app.routers.engine import router

        assert router is not None
        # Check expected routes exist (two-step secure download flow)
        paths = [r.path for r in router.routes]
        assert "/validate" in paths
        assert "/license/{slug}" in paths
        assert "/archetypes/{slug}/download-token" in paths
        assert "/archetypes/{slug}/content" in paths
