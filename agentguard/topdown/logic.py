"""L4 Logic — render logic prompts and extract stubs for the calling agent."""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentguard.prompts.registry import get_prompt_registry

if TYPE_CHECKING:
    from agentguard.archetypes.base import Archetype

logger = logging.getLogger(__name__)


@dataclass
class FunctionStub:
    """A function that needs implementation."""

    file_path: str
    function_name: str
    signature: str
    docstring: str


def render_logic_prompt(
    file_path: str,
    file_code: str,
    function_name: str,
    archetype: Archetype,
    dependency_files: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Render the L4 logic prompt for implementing a single function.

    Returns rendered messages for the calling agent to use with its own LLM.
    Does not call any LLM itself.

    Args:
        file_path: Path of the file containing the function.
        file_code: Current code of the file.
        function_name: Name of the function to implement.
        archetype: Project archetype.
        dependency_files: Optional dict of {path: code} for dependency context.

    Returns:
        List of message dicts (role/content) for the logic prompt.
    """
    stubs = _extract_stubs(file_path, file_code)
    stub = next((s for s in stubs if s.function_name == function_name), None)

    deps: list[str] = []
    if dependency_files:
        deps = _collect_dependencies_from_files(file_path, dependency_files)

    prompt_registry = get_prompt_registry()
    template = prompt_registry.get("logic")

    messages = template.render(
        language=archetype.tech_stack.language,
        file_path=file_path,
        file_code=file_code,
        function_name=function_name,
        function_signature=stub.signature if stub else function_name,
        function_docstring=stub.docstring if stub else "",
        dependencies=deps,
        reference_patterns="",
    )

    return messages


def extract_stubs(file_path: str, code: str) -> list[FunctionStub]:
    """Extract functions that have `raise NotImplementedError` as their body.

    Public wrapper around _extract_stubs for external use.
    """
    return _extract_stubs(file_path, code)


def _extract_stubs(file_path: str, code: str) -> list[FunctionStub]:
    """Extract functions that have `raise NotImplementedError` as their body."""
    stubs: list[FunctionStub] = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        logger.warning("Could not parse %s for stub extraction", file_path)
        return stubs

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _body_is_not_implemented(node):
            sig = _get_function_signature(node, code)
            doc = ast.get_docstring(node) or ""
            stubs.append(FunctionStub(
                file_path=file_path,
                function_name=node.name,
                signature=sig,
                docstring=doc,
            ))

    return stubs


def _body_is_not_implemented(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if a function body is just `raise NotImplementedError`."""
    body = node.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    if len(body) == 1 and isinstance(body[0], ast.Raise):
        exc = body[0].exc
        if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
            return exc.func.id == "NotImplementedError"
        if isinstance(exc, ast.Name):
            return exc.id == "NotImplementedError"

    return False


def _get_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef, source: str) -> str:
    """Get the function signature line from source."""
    lines = source.split("\n")
    if node.lineno <= len(lines):
        sig_lines: list[str] = []
        for i in range(node.lineno - 1, min(node.lineno + 5, len(lines))):
            sig_lines.append(lines[i])
            if ":" in lines[i] and not lines[i].strip().startswith("#"):
                break
        return "\n".join(sig_lines).strip()
    return node.name


def _collect_dependencies_from_files(file_path: str, files: dict[str, str]) -> list[str]:
    """Collect function signatures from other files as dependency context."""
    deps: list[str] = []
    for other_path, other_code in files.items():
        if other_path == file_path:
            continue
        try:
            tree = ast.parse(other_code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig = _get_function_signature(node, other_code)
                    doc = ast.get_docstring(node) or ""
                    dep_text = f"# From {other_path}:\n{sig}"
                    if doc:
                        dep_text += f'\n    """{doc}"""'
                    deps.append(dep_text)
        except SyntaxError:
            continue
    return deps


def _has_not_implemented(code: str) -> bool:
    """Quick check if code contains NotImplementedError."""
    return "NotImplementedError" in code


def _clean_code_response(content: str) -> str:
    """Strip markdown fences and whitespace from LLM code output."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip() + "\n"
