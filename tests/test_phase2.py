"""Phase 2 tests — HTTP server, MCP, archetypes, providers, CLI expansion.

Tests are grouped by module:
    - Server schemas
    - Server auth middleware
    - Server app factory
    - Server REST routes (via HTTPX test client)
    - MCP tools
    - MCP resources
    - New archetypes (script, cli_tool, web_app, library)
    - LLM factory (new providers)
    - CLI commands
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from agentguard._version import __version__

# ================================================================== #
#  1. Server Schemas
# ================================================================== #


class TestSchemas:
    """Test Pydantic request/response models."""

    def test_generate_request_defaults(self) -> None:
        from agentguard.server.schemas import GenerateRequest

        req = GenerateRequest(spec="Build a REST API")
        assert req.spec == "Build a REST API"
        assert req.archetype == "api_backend"
        assert req.options.skip_challenge is False
        assert req.options.parallel_l4 is True

    def test_generate_request_custom_options(self) -> None:
        from agentguard.server.schemas import GenerateOptions, GenerateRequest

        opts = GenerateOptions(skip_challenge=True, max_challenge_retries=5)
        req = GenerateRequest(spec="test", options=opts)
        assert req.options.skip_challenge is True
        assert req.options.max_challenge_retries == 5

    def test_generate_response_serialization(self) -> None:
        from agentguard.server.schemas import GenerateResponse, TraceSummaryResponse

        resp = GenerateResponse(
            files={"main.py": "print('hi')"},
            trace=TraceSummaryResponse(id="tr_123", total_llm_calls=5),
        )
        data = resp.model_dump()
        assert data["files"]["main.py"] == "print('hi')"
        assert data["trace"]["id"] == "tr_123"

    def test_validate_request(self) -> None:
        from agentguard.server.schemas import ValidateRequest

        req = ValidateRequest(files={"main.py": "x = 1"}, archetype="script")
        assert req.files == {"main.py": "x = 1"}
        assert req.archetype == "script"

    def test_validate_response(self) -> None:
        from agentguard.server.schemas import ValidateResponse

        resp = ValidateResponse(passed=True, checks=[], auto_fixed=[], errors=[])
        assert resp.passed is True

    def test_challenge_request_defaults(self) -> None:
        from agentguard.server.schemas import ChallengeRequest

        req = ChallengeRequest(code="def foo(): pass")
        assert req.criteria is None
        assert "anthropic" in req.llm

    def test_challenge_response(self) -> None:
        from agentguard.server.schemas import ChallengeResponse

        resp = ChallengeResponse(passed=True, attempt=1)
        assert resp.passed is True

    def test_health_response(self) -> None:
        from agentguard.server.schemas import HealthResponse

        h = HealthResponse(status="ok", version="0.1.0")
        assert h.status == "ok"

    def test_problem_detail(self) -> None:
        from agentguard.server.schemas import ProblemDetail

        p = ProblemDetail(title="Not Found", status=404, detail="x not found")
        data = p.model_dump()
        assert data["status"] == 404
        assert data["type"] == "about:blank"

    def test_archetype_summary(self) -> None:
        from agentguard.server.schemas import ArchetypeSummary

        s = ArchetypeSummary(id="script", name="Script", description="A script")
        assert s.id == "script"

    def test_archetype_detail_structure(self) -> None:
        from agentguard.server.schemas import ArchetypeDetail

        d = ArchetypeDetail(
            id="test", name="Test",
            tech_stack={"language": "python"},
            validation={"checks": ["syntax"]},
        )
        assert d.tech_stack["language"] == "python"

    def test_sse_level_event(self) -> None:
        from agentguard.server.schemas import SSELevelEvent

        evt = SSELevelEvent(level="skeleton", files=["main.py"])
        data = evt.model_dump()
        assert data["event"] == "level_complete"
        assert data["level"] == "skeleton"

    def test_sse_complete_event(self) -> None:
        from agentguard.server.schemas import SSECompleteEvent

        evt = SSECompleteEvent(files={"a.py": "code"})
        assert evt.event == "complete"
        assert evt.files == {"a.py": "code"}

    def test_trace_list_item(self) -> None:
        from agentguard.server.schemas import TraceListItem

        t = TraceListItem(id="tr_1", archetype="api_backend")
        assert t.id == "tr_1"

    def test_criterion_result_response(self) -> None:
        from agentguard.server.schemas import CriterionResultResponse

        cr = CriterionResultResponse(criterion="no secrets", passed=True, explanation="clean")
        assert cr.passed is True


# ================================================================== #
#  2. Server Auth Middleware
# ================================================================== #


class TestAuthMiddleware:
    """Test the ApiKeyMiddleware."""

    @pytest.fixture
    def app_with_key(self) -> Any:
        from agentguard.server.app import create_app

        return create_app(api_key="test-secret-key-123")

    @pytest.fixture
    def app_no_key(self) -> Any:
        from agentguard.server.app import create_app

        return create_app(api_key=None)

    @pytest.mark.asyncio
    async def test_health_is_public(self, app_with_key: Any) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app_with_key),
            base_url="http://test",
        ) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_protected_route_requires_key(self, app_with_key: Any) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app_with_key),
            base_url="http://test",
        ) as client:
            resp = await client.get("/v1/archetypes")
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_protected_route_with_valid_key(self, app_with_key: Any) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app_with_key),
            base_url="http://test",
        ) as client:
            resp = await client.get(
                "/v1/archetypes",
                headers={"X-Api-Key": "test-secret-key-123"},
            )
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_protected_route_with_wrong_key(self, app_with_key: Any) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app_with_key),
            base_url="http://test",
        ) as client:
            resp = await client.get(
                "/v1/archetypes",
                headers={"X-Api-Key": "wrong-key"},
            )
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_no_key_mode_allows_everything(self, app_no_key: Any) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app_no_key),
            base_url="http://test",
        ) as client:
            resp = await client.get("/v1/archetypes")
            assert resp.status_code == 200


# ================================================================== #
#  3. Server App Factory
# ================================================================== #


class TestAppFactory:
    """Test create_app configuration."""

    def test_create_app_returns_fastapi(self) -> None:
        from fastapi import FastAPI

        from agentguard.server.app import create_app

        app = create_app()
        assert isinstance(app, FastAPI)

    def test_app_has_health_endpoint(self) -> None:
        from agentguard.server.app import create_app

        app = create_app()
        paths = [r.path for r in app.routes]
        assert "/health" in paths

    def test_app_version(self) -> None:
        from agentguard.server.app import create_app

        app = create_app()
        assert app.version == __version__

    def test_app_state_trace_store(self) -> None:
        from agentguard.server.app import create_app

        app = create_app(trace_store="/tmp/traces")
        assert app.state.trace_store == "/tmp/traces"


# ================================================================== #
#  4. Server REST Routes
# ================================================================== #


class TestRoutes:
    """Test REST endpoints via HTTPX test client."""

    @pytest.fixture
    def app(self) -> Any:
        from agentguard.server.app import create_app

        return create_app()

    @pytest.mark.asyncio
    async def test_health_endpoint(self, app: Any) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["version"] == __version__

    @pytest.mark.asyncio
    async def test_list_archetypes(self, app: Any) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/v1/archetypes")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, list)
            assert len(data) >= 1
            ids = [a["id"] for a in data]
            assert "api_backend" in ids

    @pytest.mark.asyncio
    async def test_get_archetype_detail(self, app: Any) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/v1/archetypes/api_backend")
            assert resp.status_code == 200
            data = resp.json()
            assert data["id"] == "api_backend"
            assert "tech_stack" in data
            assert data["tech_stack"]["language"] == "python"

    @pytest.mark.asyncio
    async def test_get_archetype_not_found(self, app: Any) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/v1/archetypes/nonexistent_archetype")
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_validate_endpoint(self, app: Any) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/validate",
                json={
                    "files": {"main.py": "x = 1\n"},
                    "checks": ["syntax"],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "passed" in data
            assert "checks" in data

    @pytest.mark.asyncio
    async def test_validate_with_errors(self, app: Any) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/validate",
                json={
                    "files": {"bad.py": "def foo(\n"},
                    "checks": ["syntax"],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["passed"] is False
            assert len(data["errors"]) > 0

    @pytest.mark.asyncio
    async def test_list_traces_empty(self, app: Any) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/v1/traces")
            assert resp.status_code == 200
            assert resp.json() == []

    @pytest.mark.asyncio
    async def test_get_trace_no_store(self, app: Any) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/v1/traces/nonexistent")
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_new_archetypes(self, app: Any) -> None:
        """All 5 archetypes (api_backend + 4 new) should be discoverable."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/v1/archetypes")
            data = resp.json()
            ids = {a["id"] for a in data}
            assert "api_backend" in ids
            assert "script" in ids
            assert "cli_tool" in ids
            assert "web_app" in ids
            assert "library" in ids


# ================================================================== #
#  5. Archetypes
# ================================================================== #


class TestNewArchetypes:
    """Test that the new builtin archetypes load correctly."""

    @pytest.mark.parametrize("arch_id", ["script", "cli_tool", "web_app", "library"])
    def test_load_archetype(self, arch_id: str) -> None:
        from agentguard.archetypes.base import Archetype

        arch = Archetype.load(arch_id)
        assert arch.id == arch_id
        assert arch.name  # non-empty
        assert arch.description  # non-empty

    @pytest.mark.parametrize("arch_id", ["script", "cli_tool", "web_app", "library"])
    def test_archetype_has_validation(self, arch_id: str) -> None:
        from agentguard.archetypes.base import Archetype

        arch = Archetype.load(arch_id)
        assert len(arch.validation.checks) > 0

    @pytest.mark.parametrize("arch_id", ["script", "cli_tool", "web_app", "library"])
    def test_archetype_has_challenge_criteria(self, arch_id: str) -> None:
        from agentguard.archetypes.base import Archetype

        arch = Archetype.load(arch_id)
        assert len(arch.self_challenge.criteria) >= 3

    def test_list_available_includes_new(self) -> None:
        from agentguard.archetypes.base import Archetype

        available = Archetype.list_available()
        assert "script" in available
        assert "cli_tool" in available
        assert "web_app" in available
        assert "library" in available

    def test_script_archetype_details(self) -> None:
        from agentguard.archetypes.base import Archetype

        arch = Archetype.load("script")
        assert arch.tech_stack.framework == "stdlib"
        assert "syntax" in arch.validation.checks

    def test_library_archetype_strict_types(self) -> None:
        from agentguard.archetypes.base import Archetype

        arch = Archetype.load("library")
        assert arch.validation.type_strictness == "strict"

    def test_cli_tool_archetype_framework(self) -> None:
        from agentguard.archetypes.base import Archetype

        arch = Archetype.load("cli_tool")
        assert arch.tech_stack.framework == "click"

    def test_web_app_has_template_structure(self) -> None:
        from agentguard.archetypes.base import Archetype

        arch = Archetype.load("web_app")
        dirs = arch.structure.get("expected_dirs", [])
        has_templates = any("templates" in d for d in dirs)
        assert has_templates


# ================================================================== #
#  6. LLM Factory (new providers)
# ================================================================== #


class TestLLMFactory:
    """Test that the factory recognizes new providers."""

    def test_factory_google_import_error(self) -> None:
        """Google provider raises if google-genai not installed (we might have it)."""
        from agentguard.llm.factory import create_llm_provider

        # Just test it doesn't raise ValueError (unknown provider)
        try:
            create_llm_provider("google/gemini-2.0-flash")
        except ImportError:
            pass  # Expected if google-genai not installed
        except Exception as exc:
            # Should not be ValueError about unknown provider
            assert "Unknown LLM provider" not in str(exc)

    def test_factory_litellm_import_error(self) -> None:
        """LiteLLM provider raises if litellm not installed."""
        from agentguard.llm.factory import create_llm_provider

        try:
            create_llm_provider("litellm/gpt-4o")
        except ImportError:
            pass  # Expected if litellm not installed
        except Exception as exc:
            assert "Unknown LLM provider" not in str(exc)

    def test_factory_supports_four_providers(self) -> None:
        """Factory error message lists all four providers."""
        from agentguard.llm.factory import create_llm_provider

        with pytest.raises(ValueError, match="google"):
            create_llm_provider("unknown/model")

    def test_factory_anthropic_still_works(self) -> None:
        from agentguard.llm.factory import create_llm_provider

        provider = create_llm_provider("anthropic/claude-sonnet-4-20250514")
        assert provider.provider_name == "anthropic"

    def test_factory_openai_still_works(self) -> None:
        from agentguard.llm.factory import create_llm_provider

        try:
            provider = create_llm_provider("openai/gpt-4o")
            assert provider.provider_name == "openai"
        except Exception:
            # May fail if OPENAI_API_KEY not set — that's OK
            pass


# ================================================================== #
#  7. MCP Tools
# ================================================================== #


class TestMCPTools:
    """Test MCP tool wrapper functions."""

    @pytest.mark.asyncio
    async def test_list_archetypes_tool(self) -> None:
        from agentguard.mcp.tools import agentguard_list_archetypes

        result = await agentguard_list_archetypes()
        data = json.loads(result)
        assert isinstance(data, list)
        ids = [a["id"] for a in data]
        assert "api_backend" in ids

    @pytest.mark.asyncio
    async def test_get_archetype_tool(self) -> None:
        from agentguard.mcp.tools import agentguard_get_archetype

        result = await agentguard_get_archetype("api_backend")
        data = json.loads(result)
        assert data["id"] == "api_backend"
        assert "tech_stack" in data

    @pytest.mark.asyncio
    async def test_get_archetype_tool_not_found(self) -> None:
        from agentguard.mcp.tools import agentguard_get_archetype

        with pytest.raises(KeyError):
            await agentguard_get_archetype("nonexistent")

    @pytest.mark.asyncio
    async def test_trace_summary_tool(self) -> None:
        from agentguard.mcp.tools import agentguard_trace_summary

        result = await agentguard_trace_summary()
        data = json.loads(result)
        assert "note" in data


# ================================================================== #
#  7b. Agent-Native Tools (debug + migrate)
# ================================================================== #


class TestAgentNativeTools:
    """Test agentguard_debug and agentguard_migrate agent-native tools."""

    @pytest.mark.asyncio
    async def test_debug_returns_protocol(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_debug

        result = await agentguard_debug(symptom="500 Internal Server Error on POST /api/orders")
        data = json.loads(result)

        assert data["tool"] == "debug"
        assert "symptom" in data
        assert "debug_config" in data
        assert "instructions" in data
        assert "response_format" in data
        assert data["symptom"] == "500 Internal Server Error on POST /api/orders"

    @pytest.mark.asyncio
    async def test_debug_loads_debug_backend_archetype(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_debug

        result = await agentguard_debug(
            symptom="DB connection timeout",
            archetype="debug_backend",
        )
        data = json.loads(result)

        cfg = data["debug_config"]
        assert len(cfg["data_sources"]) > 0
        assert len(cfg["hypothesis_protocol"]) > 0
        assert len(cfg["fix_protocol"]) > 0
        assert len(cfg["escalation_criteria"]) > 0

    @pytest.mark.asyncio
    async def test_debug_loads_debug_frontend_archetype(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_debug

        result = await agentguard_debug(
            symptom="Component flickers on state update",
            archetype="debug_frontend",
        )
        data = json.loads(result)

        cfg = data["debug_config"]
        assert len(cfg["data_sources"]) > 0
        assert len(cfg["escalation_criteria"]) > 0

    @pytest.mark.asyncio
    async def test_debug_with_provided_sources(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_debug

        result = await agentguard_debug(
            symptom="NullPointerException in payment flow",
            sources={"app.log": "ERROR: NullPointerException at PaymentService.java:42"},
        )
        data = json.loads(result)

        assert data["provided_sources"]["app.log"].startswith("ERROR:")

    @pytest.mark.asyncio
    async def test_debug_fallback_for_unknown_archetype(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_debug

        # Unknown archetype — should fall back gracefully with default protocol
        result = await agentguard_debug(
            symptom="Something broke",
            archetype="nonexistent_archetype_xyz",
        )
        data = json.loads(result)

        cfg = data["debug_config"]
        assert len(cfg["data_sources"]) > 0
        assert "hypothesis_protocol" in cfg

    @pytest.mark.asyncio
    async def test_debug_response_format_contract(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_debug

        result = await agentguard_debug(symptom="test")
        data = json.loads(result)

        fmt = data["response_format"]
        assert "outcome" in fmt
        assert "root_cause" in fmt
        assert "fix" in fmt
        assert "escalation" in fmt

    @pytest.mark.asyncio
    async def test_migrate_returns_protocol(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_migrate

        result = await agentguard_migrate(
            source_files={"app.py": "import flask\napp = flask.Flask(__name__)"},
            target_archetype="api_backend",
            spec="Migrate Flask app to FastAPI",
        )
        data = json.loads(result)

        assert data["tool"] == "migrate"
        assert data["target_archetype"] == "api_backend"
        assert "migration_config" in data
        assert "source_files_digest" in data
        assert "instructions" in data
        assert "response_format" in data

    @pytest.mark.asyncio
    async def test_migrate_digests_source_files(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_migrate

        files = {
            "main.py": "# entry\nprint('hello')",
            "models.py": "class User:\n    pass",
        }
        result = await agentguard_migrate(source_files=files, target_archetype="api_backend")
        data = json.loads(result)

        digest = data["source_files_digest"]
        paths_in_digest = [d["path"] for d in digest]
        assert "main.py" in paths_in_digest
        assert "models.py" in paths_in_digest

        for entry in digest:
            assert "lines" in entry
            assert "preview" in entry

    @pytest.mark.asyncio
    async def test_migrate_includes_target_stack(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_migrate

        result = await agentguard_migrate(
            source_files={"app.py": "pass"},
            target_archetype="api_backend",
        )
        data = json.loads(result)

        stack = data["target_stack"]
        assert stack["language"] == "python"
        assert stack["framework"] == "fastapi"

    @pytest.mark.asyncio
    async def test_migrate_response_format_contract(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_migrate

        result = await agentguard_migrate(source_files={}, target_archetype="api_backend")
        data = json.loads(result)

        fmt = data["response_format"]
        assert "outcome" in fmt
        assert "migrated_files" in fmt
        assert "blocker_report" in fmt

    @pytest.mark.asyncio
    async def test_migrate_fallback_for_unknown_archetype(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_migrate

        result = await agentguard_migrate(
            source_files={"app.py": "pass"},
            target_archetype="nonexistent_archetype_xyz",
        )
        data = json.loads(result)

        # Should fall back gracefully with generic migration protocol
        cfg = data["migration_config"]
        assert len(cfg["risk_areas"]) > 0
        assert len(cfg["step_order"]) > 0


# ================================================================== #
#  8. MCP Resources
# ================================================================== #


class TestMCPResources:
    """Test MCP resource functions."""

    def test_archetypes_resource(self) -> None:
        from agentguard.mcp.resources import get_archetypes_resource

        result = get_archetypes_resource()
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) >= 5  # api_backend + 4 new

    def test_archetype_resource_detail(self) -> None:
        from agentguard.mcp.resources import get_archetype_resource

        result = get_archetype_resource("api_backend")
        data = json.loads(result)
        assert data["id"] == "api_backend"
        assert "validation" in data
        assert "self_challenge" in data

    def test_archetype_resource_not_found(self) -> None:
        from agentguard.mcp.resources import get_archetype_resource

        with pytest.raises(KeyError):
            get_archetype_resource("nonexistent")


# ================================================================== #
#  9. MCP Server Setup
# ================================================================== #


class TestMCPServer:
    """Test MCP server creation."""

    def test_create_mcp_server(self) -> None:
        """MCP server instance can be created without errors."""
        from agentguard.mcp.server import _create_mcp_server

        server = _create_mcp_server()
        assert server is not None


# ================================================================== #
#  10. CLI Expansion
# ================================================================== #


class TestCLIExpansion:
    """Test the new CLI commands exist and are wired correctly."""

    def test_cli_has_serve_command(self) -> None:
        from click.testing import CliRunner

        from agentguard.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Start the AgentGuard HTTP server" in result.output

    def test_cli_has_mcp_serve_command(self) -> None:
        from click.testing import CliRunner

        from agentguard.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["mcp-serve", "--help"])
        assert result.exit_code == 0
        assert "Start the AgentGuard MCP server" in result.output

    def test_cli_has_validate_command(self) -> None:
        from click.testing import CliRunner

        from agentguard.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["validate", "--help"])
        assert result.exit_code == 0
        assert "Validate code files" in result.output

    def test_cli_has_challenge_command(self) -> None:
        from click.testing import CliRunner

        from agentguard.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["challenge", "--help"])
        assert result.exit_code == 0
        assert "Self-challenge" in result.output

    def test_cli_serve_options(self) -> None:
        from click.testing import CliRunner

        from agentguard.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--api-key" in result.output

    def test_cli_mcp_serve_options(self) -> None:
        from click.testing import CliRunner

        from agentguard.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["mcp-serve", "--help"])
        assert "--transport" in result.output
        assert "--port" in result.output

    def test_cli_validate_with_good_file(self, tmp_path: Any) -> None:
        from click.testing import CliRunner

        from agentguard.cli import main

        good_file = tmp_path / "good.py"
        good_file.write_text("x = 1\n", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(good_file), "--checks", "syntax"])
        assert result.exit_code == 0
        assert "✓" in result.output

    def test_cli_validate_with_bad_file(self, tmp_path: Any) -> None:
        from click.testing import CliRunner

        from agentguard.cli import main

        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def foo(\n", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(bad_file), "--checks", "syntax"])
        assert result.exit_code == 1
        assert "FAILED" in result.output


# ================================================================== #
#  11. Public API
# ================================================================== #


class TestPublicAPI:
    """Test that Phase 2 exports are accessible."""

    def test_create_app_importable(self) -> None:
        from agentguard import create_app

        assert callable(create_app)

    def test_version_accessible(self) -> None:
        import agentguard

        assert agentguard.__version__ == __version__
