"""L4 Logic — implement function bodies one at a time."""

from __future__ import annotations

import ast
import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentguard.llm.types import GenerationConfig
from agentguard.prompts.registry import get_prompt_registry
from agentguard.topdown.types import LogicResult, WiringResult
from agentguard.tracing.trace import SpanType

if TYPE_CHECKING:
    from agentguard.archetypes.base import Archetype
    from agentguard.llm.base import LLMProvider
    from agentguard.tracing.tracer import Tracer

logger = logging.getLogger(__name__)


@dataclass
class FunctionStub:
    """A function that needs implementation."""

    file_path: str
    function_name: str
    signature: str
    docstring: str


async def generate_logic(
    wiring: WiringResult,
    archetype: Archetype,
    llm: LLMProvider,
    tracer: Tracer,
    parallel: bool = True,
) -> LogicResult:
    """L4: Implement function bodies.

    For each file with NotImplementedError stubs, asks the LLM to implement
    the function bodies. Can run in parallel per-file.

    Args:
        wiring: L3 wiring result.
        archetype: Project archetype.
        llm: LLM provider.
        tracer: Tracer for recording spans.
        parallel: Whether to generate in parallel (per file).

    Returns:
        LogicResult with fully implemented files.
    """
    files: dict[str, str] = {}

    with tracer.span("L4_logic", SpanType.LEVEL) as _level_span:
        if parallel:
            # Generate all files in parallel
            tasks = [
                _implement_file(file_path, file_code, wiring, archetype, llm, tracer)
                for file_path, file_code in wiring.files.items()
                if _has_not_implemented(file_code)
            ]
            results = await asyncio.gather(*tasks)
            for file_path, code in results:
                files[file_path] = code
        else:
            for file_path, file_code in wiring.files.items():
                if _has_not_implemented(file_code):
                    _, code = await _implement_file(
                        file_path, file_code, wiring, archetype, llm, tracer
                    )
                    files[file_path] = code

    # Include files that didn't need implementation (no NotImplementedError)
    for file_path, file_code in wiring.files.items():
        if file_path not in files:
            files[file_path] = file_code

    logger.info("L4 logic: %d files with implementations", len(files))
    return LogicResult(files=files, wiring=wiring)


async def _implement_file(
    file_path: str,
    file_code: str,
    wiring: WiringResult,
    archetype: Archetype,
    llm: LLMProvider,
    tracer: Tracer,
) -> tuple[str, str]:
    """Implement all functions in a single file."""
    stubs = _extract_stubs(file_path, file_code)

    if not stubs:
        return file_path, file_code

    # Collect dependency signatures from other files
    deps = _collect_dependencies(file_path, wiring)

    prompt_registry = get_prompt_registry()
    template = prompt_registry.get("logic")

    current_code = file_code

    for stub in stubs:
        messages = template.render(
            language=archetype.tech_stack.language,
            file_path=file_path,
            file_code=current_code,
            function_name=stub.function_name,
            function_signature=stub.signature,
            function_docstring=stub.docstring,
            dependencies=deps,
            reference_patterns="",  # TODO: load from archetype
        )

        with tracer.span(f"llm_logic_{file_path}::{stub.function_name}", SpanType.LLM_CALL) as llm_span:
            response = await llm.generate(
                messages,
                config=GenerationConfig(temperature=0.0, max_tokens=4096),
            )
            tracer.record_llm_response(llm_span, response)

        current_code = _clean_code_response(response.content)
        logger.info("L4 logic: implemented %s in %s", stub.function_name, file_path)

    return file_path, current_code


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
            # Build signature string
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
    # Skip the docstring if present
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
        # Grab from def/async def line to the colon
        sig_lines: list[str] = []
        for i in range(node.lineno - 1, min(node.lineno + 5, len(lines))):
            sig_lines.append(lines[i])
            if ":" in lines[i] and not lines[i].strip().startswith("#"):
                break
        return "\n".join(sig_lines).strip()
    return node.name


def _collect_dependencies(file_path: str, wiring: WiringResult) -> list[str]:
    """Collect function signatures from other files as dependency context."""
    deps: list[str] = []
    for other_path, other_code in wiring.files.items():
        if other_path == file_path:
            continue
        # Extract just the function signatures (first 2 lines of each function)
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
