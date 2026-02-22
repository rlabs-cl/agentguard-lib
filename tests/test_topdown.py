"""Tests for the top-down generation module."""

from __future__ import annotations

import json

import pytest

from agentguard.topdown.contracts import generate_contracts
from agentguard.topdown.generator import TopDownGenerator
from agentguard.topdown.logic import (
    _extract_stubs,
    _has_not_implemented,
    generate_logic,
)
from agentguard.topdown.skeleton import generate_skeleton
from agentguard.topdown.types import ContractsResult, FileEntry, SkeletonResult, WiringResult
from agentguard.topdown.wiring import generate_wiring
from agentguard.tracing.tracer import Tracer

# --- Helper tests (synchronous) ---

class TestStubExtraction:
    def test_has_not_implemented(self):
        code = '''
def greet(name: str) -> str:
    """Say hello."""
    raise NotImplementedError
'''
        assert _has_not_implemented(code)

    def test_no_not_implemented(self):
        code = '''
def greet(name: str) -> str:
    return f"Hello {name}"
'''
        assert not _has_not_implemented(code)

    def test_extract_stubs(self):
        code = '''
def implemented(x: int) -> int:
    return x + 1

def stub_func(name: str) -> str:
    """Greet someone."""
    raise NotImplementedError

async def async_stub(data: dict) -> list:
    raise NotImplementedError
'''
        stubs = _extract_stubs("test.py", code)
        assert len(stubs) == 2
        names = {s.function_name for s in stubs}
        assert "stub_func" in names
        assert "async_stub" in names

    def test_extract_stubs_syntax_error(self):
        code = "this is not valid python {{{"
        stubs = _extract_stubs("bad.py", code)
        assert stubs == []


# --- Async generation tests ---

@pytest.mark.asyncio
class TestSkeletonGeneration:
    async def test_generate_skeleton(self, mock_llm_factory, api_backend_archetype):
        skeleton_json = json.dumps([
            {"path": "main.py", "purpose": "Application entry point"},
            {"path": "models.py", "purpose": "Data models"},
            {"path": "routes.py", "purpose": "API routes"},
        ])
        llm = mock_llm_factory([skeleton_json])
        tracer = Tracer()
        tracer.new_trace(archetype="test", spec="test")

        result = await generate_skeleton(
            "A simple API", api_backend_archetype, llm, tracer
        )

        assert len(result.files) == 3
        assert result.files[0].path == "main.py"
        assert llm.call_count == 1

    async def test_skeleton_handles_markdown_fences(self, mock_llm_factory, api_backend_archetype):
        skeleton_json = '```json\n' + json.dumps([
            {"path": "app.py", "purpose": "Main app"},
        ]) + '\n```'
        llm = mock_llm_factory([skeleton_json])
        tracer = Tracer()
        tracer.new_trace(archetype="test", spec="test")

        result = await generate_skeleton(
            "A simple app", api_backend_archetype, llm, tracer
        )
        assert len(result.files) == 1


@pytest.mark.asyncio
class TestContractsGeneration:
    async def test_generate_contracts(self, mock_llm_factory, api_backend_archetype):
        skeleton = SkeletonResult(files=[
            FileEntry(path="main.py", purpose="Entry point"),
            FileEntry(path="README.md", purpose="Documentation"),
        ])

        contract_code = '''"""Main entry point."""

from fastapi import FastAPI

app = FastAPI()

def start() -> None:
    raise NotImplementedError
'''
        llm = mock_llm_factory([contract_code])
        tracer = Tracer()
        tracer.new_trace(archetype="test", spec="test")

        result = await generate_contracts(
            "A simple API", skeleton, api_backend_archetype, llm, tracer
        )

        # Only .py files get contracts generated
        assert "main.py" in result.files
        assert "README.md" not in result.files
        assert llm.call_count == 1  # Only called for main.py


@pytest.mark.asyncio
class TestWiringGeneration:
    async def test_generate_wiring(self, mock_llm_factory, api_backend_archetype):
        contracts = ContractsResult(
            files={
                "main.py": 'from routes import router\n\ndef start():\n    raise NotImplementedError\n',
                "routes.py": 'from fastapi import APIRouter\n\nrouter = APIRouter()\n',
            },
            skeleton=SkeletonResult(files=[
                FileEntry(path="main.py", purpose="Entry"),
                FileEntry(path="routes.py", purpose="Routes"),
            ]),
        )

        wired_main = 'from routes import router\n\ndef start():\n    raise NotImplementedError\n'
        wired_routes = 'from fastapi import APIRouter\n\nrouter = APIRouter()\n'

        llm = mock_llm_factory([wired_main, wired_routes])
        tracer = Tracer()
        tracer.new_trace(archetype="test", spec="test")

        result = await generate_wiring(contracts, api_backend_archetype, llm, tracer)
        assert "main.py" in result.files
        assert "routes.py" in result.files


@pytest.mark.asyncio
class TestLogicGeneration:
    async def test_generate_logic(self, mock_llm_factory, api_backend_archetype):
        wiring = WiringResult(
            files={
                "main.py": 'def start():\n    """Start the server."""\n    raise NotImplementedError\n',
            },
            contracts=ContractsResult(
                files={"main.py": "stub"},
                skeleton=SkeletonResult(files=[FileEntry(path="main.py", purpose="Entry")]),
            ),
        )

        implemented = 'import uvicorn\n\ndef start():\n    """Start the server."""\n    uvicorn.run("app:app", host="0.0.0.0", port=8000)\n'
        llm = mock_llm_factory([implemented])
        tracer = Tracer()
        tracer.new_trace(archetype="test", spec="test")

        result = await generate_logic(wiring, api_backend_archetype, llm, tracer)
        assert "main.py" in result.files
        assert "NotImplementedError" not in result.files["main.py"]


@pytest.mark.asyncio
class TestTopDownGenerator:
    async def test_full_generation(self, mock_llm_factory, api_backend_archetype):
        """Integration test: run all 4 levels with mock LLM."""
        # Prepare responses for each level
        skeleton_json = json.dumps([
            {"path": "main.py", "purpose": "Entry point"},
        ])
        contract_code = 'def start():\n    """Start."""\n    raise NotImplementedError\n'
        wired_code = 'def start():\n    """Start."""\n    raise NotImplementedError\n'
        logic_code = 'def start():\n    """Start."""\n    print("started")\n'

        llm = mock_llm_factory([skeleton_json, contract_code, wired_code, logic_code])
        tracer = Tracer()
        tracer.new_trace(archetype="test", spec="test")

        gen = TopDownGenerator(
            archetype=api_backend_archetype,
            llm=llm,
            tracer=tracer,
            parallel_logic=False,
        )

        result = await gen.generate("A simple app")
        assert result.skeleton is not None
        assert result.contracts is not None
        assert result.wiring is not None
        assert result.logic is not None
        assert len(result.files) >= 1
