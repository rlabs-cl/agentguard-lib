"""Phase 3 tests — TypeScript SDK structure, framework integrations, examples, MCP config.

Tests are grouped by module:
    - TypeScript SDK file structure
    - Framework integrations (langgraph, crewai, openhands)
    - Example projects structure
    - MCP configuration
    - MCP tools (non-LLM, end-to-end)
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentguard.llm.types import CostEstimate

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# When running from lib/ inside the monorepo, some tests need the monorepo root
_MONOREPO_ROOT = PROJECT_ROOT.parent if PROJECT_ROOT.name == "lib" else PROJECT_ROOT
_IN_MONOREPO = (PROJECT_ROOT.name == "lib") and (_MONOREPO_ROOT / ".git").is_dir()

_skip_monorepo = pytest.mark.skipif(
    not _IN_MONOREPO or not (_MONOREPO_ROOT / "sdks").is_dir(),
    reason="Monorepo-only test (sdks/, .vscode/ not in standalone library)",
)


# ================================================================== #
#  1. TypeScript SDK Structure
# ================================================================== #


@_skip_monorepo
class TestTypeScriptSDK:
    """Verify that the TypeScript SDK files exist with correct structure."""

    SDK_ROOT = _MONOREPO_ROOT / "sdks" / "typescript"

    def test_package_json_exists(self) -> None:
        pkg = self.SDK_ROOT / "package.json"
        assert pkg.exists(), "sdks/typescript/package.json missing"
        data = json.loads(pkg.read_text(encoding="utf-8"))
        assert data["name"] == "@agentguard/sdk"
        assert data["version"] == "0.1.0"
        assert data["type"] == "module"

    def test_tsconfig_exists(self) -> None:
        cfg = self.SDK_ROOT / "tsconfig.json"
        assert cfg.exists(), "sdks/typescript/tsconfig.json missing"
        data = json.loads(cfg.read_text(encoding="utf-8"))
        assert data["compilerOptions"]["strict"] is True

    def test_source_files_exist(self) -> None:
        expected = [
            "src/types.ts",
            "src/errors.ts",
            "src/streaming.ts",
            "src/client.ts",
            "src/index.ts",
        ]
        for rel_path in expected:
            p = self.SDK_ROOT / rel_path
            assert p.exists(), f"sdks/typescript/{rel_path} missing"

    def test_types_ts_exports_interfaces(self) -> None:
        content = (self.SDK_ROOT / "src" / "types.ts").read_text(encoding="utf-8")
        required = [
            "AgentGuardConfig",
            "GenerateRequest",
            "GenerateResult",
            "ValidateRequest",
            "ValidationReport",
            "ChallengeRequest",
            "ChallengeResult",
            "HealthResponse",
            "ProblemDetail",
        ]
        for name in required:
            assert name in content, f"types.ts missing interface: {name}"

    def test_client_ts_exports_class(self) -> None:
        content = (self.SDK_ROOT / "src" / "client.ts").read_text(encoding="utf-8")
        assert "class AgentGuard" in content
        methods = ["generate", "validate", "challenge", "health", "listArchetypes"]
        for method in methods:
            assert method in content, f"client.ts missing method: {method}"

    def test_errors_ts_exports_classes(self) -> None:
        content = (self.SDK_ROOT / "src" / "errors.ts").read_text(encoding="utf-8")
        classes = [
            "AgentGuardError",
            "AgentGuardAPIError",
            "AgentGuardTimeoutError",
            "AgentGuardConnectionError",
        ]
        for cls in classes:
            assert cls in content, f"errors.ts missing class: {cls}"

    def test_streaming_ts_exports(self) -> None:
        content = (self.SDK_ROOT / "src" / "streaming.ts").read_text(encoding="utf-8")
        assert "streamSSE" in content
        assert "parseSSEEvent" in content

    def test_index_reexports(self) -> None:
        content = (self.SDK_ROOT / "src" / "index.ts").read_text(encoding="utf-8")
        for module in ["types", "errors", "streaming", "client"]:
            assert module in content, f"index.ts missing reexport from {module}"

    def test_sdk_readme_exists(self) -> None:
        readme = self.SDK_ROOT / "README.md"
        assert readme.exists(), "sdks/typescript/README.md missing"
        content = readme.read_text(encoding="utf-8")
        assert "@agentguard/sdk" in content


# ================================================================== #
#  2. Framework Integrations — LangGraph
# ================================================================== #


class TestLangGraphIntegration:
    """Test LangGraph integration nodes."""

    def test_module_imports(self) -> None:
        from agentguard.integrations.langgraph import (
            agentguard_challenge_node,
            agentguard_generate_node,
            agentguard_validate_node,
        )
        assert callable(agentguard_generate_node)
        assert callable(agentguard_validate_node)
        assert callable(agentguard_challenge_node)

    @pytest.mark.asyncio
    async def test_generate_node_calls_pipeline(self) -> None:
        mock_result = MagicMock()
        mock_result.files = {"main.py": "print('hi')"}
        mock_result.trace = None
        mock_result.total_cost = CostEstimate(input_cost=Decimal("0.03"), output_cost=Decimal("0.02"))

        with patch("agentguard.pipeline.Pipeline") as MockPipe:
            instance = MockPipe.return_value
            instance.generate = AsyncMock(return_value=mock_result)

            from agentguard.integrations.langgraph import agentguard_generate_node

            state = {"spec": "Build a calculator", "archetype": "script"}
            result = await agentguard_generate_node(state)

            assert result["files"] == {"main.py": "print('hi')"}
            assert result["generation_cost"] == 0.05
            instance.generate.assert_awaited_once_with("Build a calculator")

    @pytest.mark.asyncio
    async def test_validate_node_returns_passed(self) -> None:
        mock_report = MagicMock()
        mock_report.passed = True
        mock_report.errors = []
        mock_report.auto_fixed = []

        with patch("agentguard.validation.validator.Validator") as MockVal, \
             patch("agentguard.archetypes.base.Archetype"):
            MockVal.return_value.check.return_value = mock_report

            from agentguard.integrations.langgraph import agentguard_validate_node

            state = {"files": {"main.py": "x = 1"}, "archetype": "script"}
            result = await agentguard_validate_node(state)

            assert result["validation_passed"] is True
            assert result["validation_errors"] == []

    @pytest.mark.asyncio
    async def test_challenge_node_returns_result(self) -> None:
        mock_result = MagicMock()
        mock_result.passed = True
        mock_result.feedback = "All good"
        mock_result.grounding_violations = []

        with patch("agentguard.llm.factory.create_llm_provider"), \
             patch("agentguard.challenge.challenger.SelfChallenger") as MockChallenger:
            MockChallenger.return_value.challenge = AsyncMock(return_value=mock_result)

            from agentguard.integrations.langgraph import agentguard_challenge_node

            state = {"files": {"main.py": "x = 1"}, "criteria": ["no bugs"]}
            result = await agentguard_challenge_node(state)

            assert result["challenge_passed"] is True
            assert result["challenge_feedback"] == "All good"


# ================================================================== #
#  3. Framework Integrations — CrewAI
# ================================================================== #


class TestCrewAIIntegration:
    """Test CrewAI integration tools."""

    def test_module_imports(self) -> None:
        from agentguard.integrations.crewai import (
            agentguard_challenge,
            agentguard_generate,
            agentguard_validate,
        )
        assert callable(agentguard_generate)
        assert callable(agentguard_validate)
        assert callable(agentguard_challenge)

    def test_validate_sync_no_llm(self) -> None:
        """validate is sync and does NOT need an LLM — can test directly."""
        from agentguard.integrations.crewai import agentguard_validate

        result = agentguard_validate(
            files_json='{"hello.py": "print(42)\\n"}',
            archetype="script",
        )
        data = json.loads(result)
        assert "passed" in data
        assert isinstance(data["passed"], bool)
        assert isinstance(data["errors"], list)

    def test_validate_catches_syntax_error(self) -> None:
        from agentguard.integrations.crewai import agentguard_validate

        result = agentguard_validate(
            files_json='{"broken.py": "def f(\\n"}',
        )
        data = json.loads(result)
        assert data["passed"] is False
        assert len(data["errors"]) > 0

    def test_generate_calls_pipeline(self) -> None:
        mock_result = MagicMock()
        mock_result.files = {"main.py": "print('hi')"}

        with patch("agentguard.pipeline.Pipeline") as MockPipe:
            instance = MockPipe.return_value
            instance.generate = AsyncMock(return_value=mock_result)

            from agentguard.integrations.crewai import agentguard_generate

            result = agentguard_generate(spec="Build X", archetype="script")
            data = json.loads(result)
            assert "main.py" in data


# ================================================================== #
#  4. Framework Integrations — OpenHands
# ================================================================== #


class TestOpenHandsIntegration:
    """Test OpenHands micro-agent integration."""

    def test_module_imports(self) -> None:
        from agentguard.integrations.openhands import (
            MICRO_AGENT_DESCRIPTION,
        )
        assert "agentguard" in MICRO_AGENT_DESCRIPTION.lower()

    def test_micro_agent_result_to_json(self) -> None:
        from agentguard.integrations.openhands import MicroAgentResult

        r = MicroAgentResult(action="test", success=True, data={"key": "value"})
        j = json.loads(r.to_json())
        assert j["action"] == "test"
        assert j["success"] is True
        assert j["data"]["key"] == "value"
        assert j["error"] is None

    @pytest.mark.asyncio
    async def test_generate_action(self) -> None:
        mock_result = MagicMock()
        mock_result.files = {"main.py": "print('hello')"}
        mock_result.total_cost = Decimal("0.05")

        with patch("agentguard.pipeline.Pipeline") as MockPipe:
            instance = MockPipe.return_value
            instance.generate = AsyncMock(return_value=mock_result)

            from agentguard.integrations.openhands import AgentGuardMicroAgent

            agent = AgentGuardMicroAgent()
            result = await agent.run(instruction="Build a hello world", action="generate")

            assert result.success is True
            assert result.action == "generate"
            assert "main.py" in result.data["files"]

    @pytest.mark.asyncio
    async def test_validate_action(self) -> None:
        from agentguard.integrations.openhands import AgentGuardMicroAgent

        agent = AgentGuardMicroAgent()
        result = await agent.run(
            instruction="Validate code",
            action="validate",
            files={"hello.py": "print(42)\n"},
            archetype="script",
        )
        assert result.action == "validate"
        assert isinstance(result.success, bool)

    @pytest.mark.asyncio
    async def test_unknown_action(self) -> None:
        from agentguard.integrations.openhands import AgentGuardMicroAgent

        agent = AgentGuardMicroAgent()
        result = await agent.run(instruction="test", action="unknown_action")

        assert result.success is False
        assert "Unknown action" in result.error

    @pytest.mark.asyncio
    async def test_error_handling(self) -> None:
        with patch("agentguard.pipeline.Pipeline") as MockPipe:
            instance = MockPipe.return_value
            instance.generate = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

            from agentguard.integrations.openhands import AgentGuardMicroAgent

            agent = AgentGuardMicroAgent()
            result = await agent.run(instruction="test", action="generate")

            assert result.success is False
            assert "LLM unavailable" in result.error


# ================================================================== #
#  5. Example Projects Structure
# ================================================================== #


class TestExampleProjects:
    """Verify example project files exist and are well-formed."""

    EXAMPLES_ROOT = PROJECT_ROOT / "examples"

    def test_examples_readme_exists(self) -> None:
        readme = self.EXAMPLES_ROOT / "README.md"
        assert readme.exists()
        content = readme.read_text(encoding="utf-8")
        assert "basic_generation" in content

    def test_basic_generation_example(self) -> None:
        path = self.EXAMPLES_ROOT / "basic_generation.py"
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Pipeline" in content
        assert "generate" in content
        assert "async" in content

    def test_validation_workflow_example(self) -> None:
        path = self.EXAMPLES_ROOT / "validation_workflow.py"
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Validator" in content
        assert "check" in content

    def test_langgraph_pipeline_example(self) -> None:
        path = self.EXAMPLES_ROOT / "langgraph_pipeline.py"
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "StateGraph" in content or "langgraph" in content
        assert "agentguard_generate_node" in content


# ================================================================== #
#  6. MCP Configuration
# ================================================================== #


@_skip_monorepo
class TestMCPConfiguration:
    """Verify MCP configuration for VS Code / Copilot."""

    def test_mcp_json_exists(self) -> None:
        mcp_json = _MONOREPO_ROOT / ".vscode" / "mcp.json"
        assert mcp_json.exists(), ".vscode/mcp.json missing"

    def test_mcp_json_valid(self) -> None:
        mcp_json = _MONOREPO_ROOT / ".vscode" / "mcp.json"
        data = json.loads(mcp_json.read_text(encoding="utf-8"))
        assert "servers" in data
        assert "agentguard" in data["servers"]
        server = data["servers"]["agentguard"]
        assert server["type"] == "stdio"
        assert "mcp-serve" in server["args"]

    def test_mcp_server_creates_successfully(self) -> None:
        """Verify the MCP server object can be created with all tools."""
        from agentguard.mcp.server import _create_mcp_server

        mcp = _create_mcp_server()
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        # Utility tools
        assert "validate" in tool_names
        assert "list_archetypes" in tool_names
        assert "get_archetype" in tool_names
        assert "trace_summary" in tool_names
        # Agent-native tools (no API key)
        assert "skeleton" in tool_names
        assert "contracts" in tool_names
        assert "wiring" in tool_names
        assert "logic" in tool_names
        assert "get_challenge_criteria" in tool_names
        assert "contracts_and_wiring" in tool_names
        assert "digest" in tool_names
        # Full-pipeline tools (require API key)
        assert "generate" in tool_names
        assert "challenge" in tool_names
        # Benchmark tool
        assert "benchmark" in tool_names
        assert len(tool_names) == 14


# ================================================================== #
#  7. MCP Tools End-to-End (no LLM needed)
# ================================================================== #


class TestMCPToolsE2E:
    """Run MCP tool functions directly — no LLM calls needed."""

    @pytest.mark.asyncio
    async def test_list_archetypes(self) -> None:
        from agentguard.mcp.tools import agentguard_list_archetypes

        result = await agentguard_list_archetypes()
        data = json.loads(result)
        ids = [a["id"] for a in data]
        assert "api_backend" in ids
        assert "script" in ids
        assert "cli_tool" in ids
        assert "library" in ids
        assert "web_app" in ids

    @pytest.mark.asyncio
    async def test_get_archetype(self) -> None:
        from agentguard.mcp.tools import agentguard_get_archetype

        result = await agentguard_get_archetype(name="script")
        data = json.loads(result)
        assert data["id"] == "script"
        assert data["tech_stack"]["language"] == "python"
        assert "skeleton" in data["pipeline_levels"]

    @pytest.mark.asyncio
    async def test_validate_passing(self) -> None:
        from agentguard.mcp.tools import agentguard_validate

        result = await agentguard_validate(
            files={"hello.py": "def greet() -> str:\n    return 'hi'\n"},
            archetype="script",
        )
        data = json.loads(result)
        assert data["passed"] is True

    @pytest.mark.asyncio
    async def test_validate_failing(self) -> None:
        from agentguard.mcp.tools import agentguard_validate

        result = await agentguard_validate(
            files={"broken.py": "def f(\n"},
            archetype="script",
        )
        data = json.loads(result)
        assert data["passed"] is False
        assert len(data["errors"]) > 0

    @pytest.mark.asyncio
    async def test_get_archetype_not_found(self) -> None:
        from agentguard.mcp.tools import agentguard_get_archetype

        with pytest.raises(KeyError):
            await agentguard_get_archetype(name="nonexistent_arch")


# ================================================================== #
#  8. MCP Resources
# ================================================================== #


class TestMCPResourcesE2E:
    """Test MCP resources directly."""

    def test_archetypes_resource(self) -> None:
        from agentguard.mcp.resources import get_archetypes_resource

        result = get_archetypes_resource()
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) >= 5
        ids = [a["id"] for a in data]
        assert "api_backend" in ids

    def test_archetype_resource(self) -> None:
        from agentguard.mcp.resources import get_archetype_resource

        result = get_archetype_resource("api_backend")
        data = json.loads(result)
        assert data["id"] == "api_backend"
        assert "tech_stack" in data
        assert "pipeline" in data or "pipeline_levels" in data

    def test_archetype_resource_all_archetypes(self) -> None:
        """Ensure every archetype loads as a resource."""
        from agentguard.mcp.resources import get_archetype_resource, get_archetypes_resource

        listing = json.loads(get_archetypes_resource())
        for arch in listing:
            detail = json.loads(get_archetype_resource(arch["id"]))
            assert detail["id"] == arch["id"]
            assert "tech_stack" in detail


# ================================================================== #
#  9. Agent-Native MCP Tools (no API key needed)
# ================================================================== #


class TestAgentNativeTools:
    """Test agent-native MCP tools — these return structured prompts, no LLM calls."""

    @pytest.mark.asyncio
    async def test_skeleton_returns_instructions(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_skeleton

        result = await agentguard_skeleton(spec="Build a calculator", archetype="script")
        data = json.loads(result)
        assert data["level"] == "L1 — Skeleton"
        assert data["archetype"] == "script"
        assert len(data["instructions"]) >= 1
        assert data["tech_stack"]["language"] == "python"
        assert "next_step" in data

    @pytest.mark.asyncio
    async def test_contracts_returns_per_file_prompts(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_contracts

        skeleton = json.dumps([
            {"path": "main.py", "purpose": "Entry point"},
            {"path": "utils.py", "purpose": "Utility functions"},
        ])
        result = await agentguard_contracts(
            spec="Build a calculator", skeleton_json=skeleton, archetype="script",
        )
        data = json.loads(result)
        assert data["level"] == "L2 — Contracts"
        assert data["file_count"] == 2
        assert len(data["files"]) == 2
        assert data["files"][0]["file"] == "main.py"
        assert len(data["files"][0]["instructions"]) >= 1

    @pytest.mark.asyncio
    async def test_wiring_returns_per_file_prompts(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_wiring

        contracts = json.dumps({
            "main.py": "from utils import add\ndef main(): raise NotImplementedError",
            "utils.py": "def add(a: int, b: int) -> int: raise NotImplementedError",
        })
        result = await agentguard_wiring(contracts_json=contracts, archetype="script")
        data = json.loads(result)
        assert data["level"] == "L3 — Wiring"
        assert data["file_count"] == 2
        assert len(data["files"]) == 2

    @pytest.mark.asyncio
    async def test_logic_returns_function_prompt(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_logic

        result = await agentguard_logic(
            file_path="main.py",
            file_code="def add(a: int, b: int) -> int:\n    raise NotImplementedError\n",
            function_name="add",
            archetype="script",
        )
        data = json.loads(result)
        assert data["level"] == "L4 — Logic"
        assert "add" in data["description"]
        assert len(data["instructions"]) >= 1

    @pytest.mark.asyncio
    async def test_challenge_criteria_returns_criteria(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_get_challenge_criteria

        result = await agentguard_get_challenge_criteria(archetype="script")
        data = json.loads(result)
        assert data["level"] == "Self-Challenge"
        assert data["archetype"] == "script"
        assert data["criteria_count"] > 0
        assert isinstance(data["criteria"], list)
        assert "review_format" in data

    @pytest.mark.asyncio
    async def test_challenge_criteria_with_extras(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_get_challenge_criteria

        result = await agentguard_get_challenge_criteria(
            archetype="api_backend",
            extra_criteria=["Must use async/await", "Must have rate limiting"],
        )
        data = json.loads(result)
        assert "Must use async/await" in data["criteria"]
        assert "Must have rate limiting" in data["criteria"]
        assert data["criteria_count"] >= 2

    @pytest.mark.asyncio
    async def test_skeleton_different_archetypes(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_skeleton

        for arch_id in ["script", "api_backend", "cli_tool"]:
            result = await agentguard_skeleton(spec="test", archetype=arch_id)
            data = json.loads(result)
            assert data["archetype"] == arch_id

    # ── v1.1 enhancement tests ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_skeleton_includes_maturity_and_tiers(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_skeleton

        result = await agentguard_skeleton(spec="Build a React app", archetype="react_spa")
        data = json.loads(result)
        assert data["maturity"] == "production"
        assert "file_tiers" in data
        assert set(data["file_tiers"].keys()) == {"config", "foundation", "feature", "infrastructure"}
        assert "interface_summary_hint" in data
        assert isinstance(data["infrastructure_files"], list)

    @pytest.mark.asyncio
    async def test_skeleton_maturity_override(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_skeleton

        result = await agentguard_skeleton(
            spec="Build a React app", archetype="react_spa", maturity="enterprise",
        )
        data = json.loads(result)
        assert data["maturity"] == "enterprise"

    @pytest.mark.asyncio
    async def test_contracts_and_wiring_merged(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_contracts_and_wiring

        skeleton = json.dumps([
            {"path": "tsconfig.json", "purpose": "TypeScript config", "tier": "config"},
            {"path": "src/types.ts", "purpose": "Shared types", "tier": "foundation"},
            {"path": "src/App.tsx", "purpose": "Root component", "tier": "feature"},
        ])
        result = await agentguard_contracts_and_wiring(
            spec="Build a React app", skeleton_json=skeleton, archetype="react_spa",
        )
        data = json.loads(result)
        assert data["level"] == "L2+L3 — Contracts & Wiring (merged)"
        # Config file should be excluded
        assert data["file_count"] == 2
        files_by_path = {f["file"]: f for f in data["files"]}
        # Foundation file: contracts only, no wiring
        assert files_by_path["src/types.ts"]["tier"] == "foundation"
        assert files_by_path["src/types.ts"]["wiring_instructions"] is None
        # Feature file: contracts + wiring
        assert files_by_path["src/App.tsx"]["tier"] == "feature"
        assert files_by_path["src/App.tsx"]["wiring_instructions"] is not None
        # Anti-patterns present
        assert isinstance(data["anti_patterns"], list)
        assert len(data["anti_patterns"]) >= 1

    @pytest.mark.asyncio
    async def test_contracts_legacy_has_deprecation(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_contracts

        skeleton = json.dumps([
            {"path": "main.py", "purpose": "Entry point"},
        ])
        result = await agentguard_contracts(
            spec="Build a calc", skeleton_json=skeleton, archetype="script",
        )
        data = json.loads(result)
        assert "deprecation" in data

    @pytest.mark.asyncio
    async def test_wiring_legacy_has_deprecation(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_wiring

        contracts = json.dumps({
            "main.py": "def main(): raise NotImplementedError",
        })
        result = await agentguard_wiring(contracts_json=contracts, archetype="script")
        data = json.loads(result)
        assert "deprecation" in data

    @pytest.mark.asyncio
    async def test_challenge_criteria_includes_maturity(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_get_challenge_criteria

        result = await agentguard_get_challenge_criteria(archetype="react_spa")
        data = json.loads(result)
        assert data["maturity"] == "production"
        assert data["criteria_count"] >= 20  # react_spa has 30
        assert "tip" in data

    @pytest.mark.asyncio
    async def test_digest_returns_compact_summary(self) -> None:
        from agentguard.mcp.agent_tools import agentguard_digest

        files = json.dumps({
            "src/App.tsx": (
                "import React from 'react';\n"
                "import { useApp } from './context/AppContext';\n"
                "export function App() {\n"
                "  return <div role='main' aria-label='app'>Hello</div>;\n"
                "}\n"
            ),
            "src/types.ts": (
                "export interface Service {\n"
                "  id: string;\n"
                "  name: string;\n"
                "}\n"
            ),
        })
        result = await agentguard_digest(files_json=files, archetype="react_spa")
        data = json.loads(result)
        assert data["level"] == "Project Digest"
        assert data["cross_cutting"]["total_files"] == 2
        assert data["cross_cutting"]["total_lines"] > 0
        assert data["cross_cutting"]["has_a11y"] is True
        assert len(data["files"]) == 2
        # Check per-file digest has expected shape
        for fd in data["files"]:
            assert "path" in fd
            assert "lines" in fd
            assert "exports" in fd
            assert "patterns" in fd

    @pytest.mark.asyncio
    async def test_react_spa_archetype_loads(self) -> None:
        from agentguard.archetypes.base import Archetype

        arch = Archetype.load("react_spa")
        assert arch.id == "react_spa"
        assert arch.tech_stack.language == "typescript"
        assert arch.tech_stack.framework == "react"
        assert arch.maturity == "production"
        assert len(arch.self_challenge.criteria) >= 25
        assert len(arch.infrastructure_files) >= 1

    @pytest.mark.asyncio
    async def test_archetype_maturity_field(self) -> None:
        from agentguard.archetypes.base import Archetype

        # Production archetypes
        for arch_id in ["api_backend", "web_app", "cli_tool", "react_spa"]:
            arch = Archetype.load(arch_id)
            assert arch.maturity == "production"

        # Prototype archetype
        script = Archetype.load("script")
        assert script.maturity == "prototype"

    @pytest.mark.asyncio
    async def test_get_archetype_exposes_maturity(self) -> None:
        from agentguard.mcp.tools import agentguard_get_archetype

        result = await agentguard_get_archetype(name="react_spa")
        data = json.loads(result)
        assert data["maturity"] == "production"
        assert "infrastructure_files" in data


# ================================================================== #
#  10. Integration Package Structure
# ================================================================== #


class TestIntegrationPackage:
    """Verify integration package files exist."""

    INTEGRATIONS_ROOT = PROJECT_ROOT / "agentguard" / "integrations"

    def test_init_exists(self) -> None:
        assert (self.INTEGRATIONS_ROOT / "__init__.py").exists()

    def test_langgraph_module_exists(self) -> None:
        assert (self.INTEGRATIONS_ROOT / "langgraph.py").exists()

    def test_crewai_module_exists(self) -> None:
        assert (self.INTEGRATIONS_ROOT / "crewai.py").exists()

    def test_openhands_module_exists(self) -> None:
        assert (self.INTEGRATIONS_ROOT / "openhands.py").exists()

    def test_all_modules_importable(self) -> None:
        import importlib
        for mod in ["langgraph", "crewai", "openhands"]:
            m = importlib.import_module(f"agentguard.integrations.{mod}")
            assert m is not None
