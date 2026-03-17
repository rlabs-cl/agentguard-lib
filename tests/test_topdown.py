"""Tests for the top-down generation module — prompt rendering and helpers."""

from __future__ import annotations

import json

from agentguard.topdown.logic import (
    _extract_stubs,
    _has_not_implemented,
)
from agentguard.topdown.skeleton import _parse_skeleton_response
from agentguard.topdown.types import FileEntry, SkeletonResult


# --- Helper tests (synchronous, no LLM) ---

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


class TestSkeletonParsing:
    def test_parse_skeleton_json(self):
        content = json.dumps([
            {"path": "main.py", "purpose": "Entry point"},
            {"path": "models.py", "purpose": "Data models"},
        ])
        files = _parse_skeleton_response(content)
        assert len(files) == 2
        assert files[0].path == "main.py"
        assert files[1].purpose == "Data models"

    def test_parse_skeleton_with_fences(self):
        content = '```json\n' + json.dumps([
            {"path": "app.py", "purpose": "Main app"},
        ]) + '\n```'
        files = _parse_skeleton_response(content)
        assert len(files) == 1
        assert files[0].path == "app.py"

    def test_parse_skeleton_with_file_key(self):
        """Handles alternative 'file' key instead of 'path'."""
        content = json.dumps([
            {"file": "main.py", "description": "Entry point"},
        ])
        files = _parse_skeleton_response(content)
        assert len(files) == 1
        assert files[0].path == "main.py"
        assert files[0].purpose == "Entry point"
