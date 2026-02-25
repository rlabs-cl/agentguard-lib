"""MCP tool definitions for AgentGuard.

Each function is registered as an MCP tool on the ``FastMCP`` server.
The functions themselves are thin wrappers that call into the
``Pipeline``, ``Validator``, and ``SelfChallenger`` classes.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


async def agentguard_generate(
    spec: str,
    archetype: str = "api_backend",
    llm: str = "",  # ignored — the calling agent uses its own LLM
) -> str:
    """Return structured generation instructions for the calling agent.

    Instead of calling an internal LLM, this returns the prompts and
    workflow the calling agent should follow to generate code itself using
    its own configured LLM.  The agent always does the thinking.

    Returns a JSON object with next-step instructions for the full pipeline:
    skeleton → contracts_and_wiring → logic → validate.
    """
    return json.dumps(
        {
            "tool": "generate",
            "description": (
                "Structured generation workflow for you (the calling agent) to execute "
                "using your own LLM. Do NOT delegate to any external model — you generate "
                "all code yourself by following these steps in order."
            ),
            "spec": spec,
            "archetype": archetype,
            "steps": [
                {
                    "step": 1,
                    "action": "Call `skeleton` with the spec and archetype to get the file tree.",
                },
                {
                    "step": 2,
                    "action": (
                        "Call `contracts_and_wiring` with the skeleton JSON to get "
                        "typed stubs and import wiring instructions per file."
                    ),
                },
                {
                    "step": 3,
                    "action": (
                        "For each non-trivial function, call `logic` to get implementation "
                        "instructions, then write the full function body yourself."
                    ),
                },
                {
                    "step": 4,
                    "action": "Call `validate` on all generated files to catch syntax/import issues.",
                },
                {
                    "step": 5,
                    "action": (
                        "Call `get_challenge_criteria` for the archetype, then self-review "
                        "your generated code against each criterion and fix any issues."
                    ),
                },
            ],
            "next_step": "Call `skeleton` now with spec and archetype to begin.",
        },
        indent=2,
    )


_PREREQUISITES: dict[str, dict[str, str | list[str]]] = {
    "python": {
        "runtime": "python >= 3.9",
        "tools": ["ruff (linter)", "mypy (type checker)", "pytest (testing)"],
        "install_hint": "pip install ruff mypy pytest",
    },
    "typescript": {
        "runtime": "node >= 18",
        "tools": ["tsc (type checker)", "eslint (linter)", "jest or vitest (testing)"],
        "install_hint": "npm install -D typescript eslint jest",
    },
    "javascript": {
        "runtime": "node >= 18",
        "tools": ["eslint (linter)", "jest or vitest (testing)"],
        "install_hint": "npm install -D eslint jest",
    },
    "java": {
        "runtime": "java >= 17",
        "tools": ["maven or gradle (build)", "checkstyle (linter)", "junit (testing)"],
        "install_hint": "Install JDK 17+ and Maven/Gradle from your package manager",
    },
    "go": {
        "runtime": "go >= 1.21",
        "tools": ["go vet (linter)", "staticcheck", "go test (testing)"],
        "install_hint": "Install Go from https://go.dev/dl/",
    },
    "rust": {
        "runtime": "rustup / cargo",
        "tools": ["cargo check", "clippy (linter)", "cargo test (testing)"],
        "install_hint": "Install via rustup: https://rustup.rs/",
    },
    "csharp": {
        "runtime": "dotnet >= 8",
        "tools": ["dotnet build", "roslyn analysers", "xunit or nunit (testing)"],
        "install_hint": "Install .NET SDK from https://dotnet.microsoft.com/",
    },
    "ruby": {
        "runtime": "ruby >= 3.1",
        "tools": ["rubocop (linter)", "rspec (testing)"],
        "install_hint": "gem install rubocop rspec",
    },
}

_CHECK_CRITERIA: dict[str, dict[str, str]] = {
    "syntax": {
        "description": "Every file is syntactically valid and can be parsed/compiled.",
        "what_to_check": (
            "Read each file and verify there are no syntax errors: unclosed brackets, "
            "invalid tokens, malformed expressions, missing colons/semicolons where required "
            "by the language. If a native tool is available (python ast, tsc --noEmit, "
            "cargo check, etc.) run it and include the output."
        ),
        "severity": "blocking",
    },
    "imports": {
        "description": "All imports/requires/uses resolve to modules that exist in the project or are declared as dependencies.",
        "what_to_check": (
            "For each file, list every import statement and verify: (1) internal imports "
            "reference a file that exists in the provided file tree, (2) external packages "
            "are declared in the dependency manifest (package.json, pyproject.toml, go.mod, "
            "Cargo.toml, pom.xml, etc.). Flag any import that references a non-existent path "
            "or undeclared package."
        ),
        "severity": "blocking",
    },
    "types": {
        "description": "Type annotations are present and consistent on all public interfaces.",
        "what_to_check": (
            "Check that: (1) all public functions/methods have parameter and return type "
            "annotations, (2) type annotations are consistent across call sites (passed arg "
            "types match declared param types), (3) no use of `any` or equivalent unless "
            "explicitly justified. If a type checker is available (mypy, tsc) run it."
        ),
        "severity": "blocking",
    },
    "lint": {
        "description": "Code follows the conventions and rules of the archetype's configured linter.",
        "what_to_check": (
            "Apply the linting rules for the archetype's linter (e.g. ruff for Python, "
            "eslint for TypeScript/JS, clippy for Rust). Flag: unused imports, unused "
            "variables, shadowed names, inconsistent naming conventions, lines over 120 "
            "chars, missing docstrings on public API. If the linter tool is available, run "
            "it and include the output verbatim."
        ),
        "severity": "warning",
    },
    "structure": {
        "description": "The generated file tree matches the expected directories and entry points defined in the archetype.",
        "what_to_check": (
            "Compare the provided file paths against the archetype's expected_dirs and "
            "expected_files. Flag any required path that is absent. Note any extra files "
            "that seem misplaced or redundant."
        ),
        "severity": "blocking",
    },
}


async def agentguard_validate(
    files: dict[str, str],
    archetype: str = "api_backend",
) -> str:
    """Return a structured validation prompt for the calling agent.

    The agent performs the actual review using its own LLM and knowledge of
    the language.  No internal LLM or external tool is invoked here.

    Returns a JSON object containing:
    - prerequisites: what the local environment should have for full validation
    - criteria: scored checklist the agent must evaluate
    - response_format: exact JSON shape the agent must return
    - files: the code to review
    """
    from agentguard.archetypes.base import Archetype

    try:
        arch = Archetype.load(archetype)
        lang = arch.tech_stack.language.lower()
        linter = arch.tech_stack.linter
        type_checker = arch.tech_stack.type_checker
        checks = arch.validation.checks
        expected_dirs = arch.structure.get("expected_dirs", [])
        expected_files = arch.structure.get("expected_files", [])
    except Exception:
        lang = "python"
        linter = "ruff"
        type_checker = "mypy"
        checks = ["syntax", "lint", "types", "imports", "structure"]
        expected_dirs = []
        expected_files = []

    prerequisites = _PREREQUISITES.get(lang, {
        "runtime": lang,
        "tools": [linter, type_checker],
        "install_hint": f"Set up a {lang} development environment with {linter} and {type_checker}.",
    })

    criteria = []
    for check_id in checks:
        if check_id in _CHECK_CRITERIA:
            c = _CHECK_CRITERIA[check_id]
            criteria.append({
                "id": check_id,
                "description": c["description"],
                "what_to_check": c["what_to_check"],
                "severity": c["severity"],
                "score_scale": {
                    "0": "Critical failure — blocking issue that must be fixed before merging",
                    "1": "Warning — issue present but project can still run",
                    "2": "Acceptable — minor imperfection, non-blocking",
                    "3": "Clean — fully satisfies the criterion",
                },
            })

    return json.dumps(
        {
            "tool": "validate",
            "description": (
                "Validation task for you (the calling agent). Review the provided files "
                "against each criterion using your own LLM and language knowledge. "
                "YOU perform every check — do not delegate to another model."
            ),
            "archetype": archetype,
            "language": lang,
            "prerequisites": {
                "note": (
                    f"For full static analysis of {lang} code, your local environment "
                    f"should have: {prerequisites.get('runtime')}. "
                    f"Expected tools: {', '.join(str(t) for t in prerequisites.get('tools', []))}. "
                    f"If not installed: {prerequisites.get('install_hint')} — "
                    "the agent review still covers all criteria without them."
                ),
                "runtime": prerequisites.get("runtime"),
                "tools": prerequisites.get("tools"),
            },
            "expected_structure": {
                "dirs": expected_dirs,
                "files": expected_files,
            },
            "criteria": criteria,
            "response_format": {
                "description": "Return this exact JSON structure after completing your review.",
                "schema": {
                    "passed": "boolean — true only if no criterion with severity=blocking scored below 2",
                    "blocking_failures": "integer — count of blocking criteria that scored 0 or 1",
                    "criteria_results": [
                        {
                            "criterion_id": "string — matches the id field above",
                            "score": "integer 0-3",
                            "level": "one of: critical_fail | warning | acceptable | clean",
                            "explanation": "string — what you found, be specific (file and line if relevant)",
                            "fix_suggestion": "string | null — concrete fix if score < 3, else null",
                        }
                    ],
                    "overall_notes": "string — summary of the most important findings",
                },
            },
            "files": files,
        },
        indent=2,
    )


async def agentguard_challenge(
    code: str,
    criteria: list[str] | None = None,
    llm: str = "",  # ignored — the calling agent uses its own LLM
) -> str:
    """Return a self-review prompt for the calling agent.

    Instead of calling an internal LLM, this packages the code and criteria
    into a structured review task that the calling agent executes itself using
    its own configured LLM.  The agent always does the thinking.

    Returns a JSON object with the code and criteria for the agent to review.
    """
    review_criteria = criteria or [
        "No hardcoded secrets, credentials, or environment-specific values",
        "All imports are used and resolvable",
        "Error handling present on all I/O and external calls",
        "Functions have type annotations",
        "No dead code or TODO stubs left in production paths",
        "Consistent naming conventions throughout",
    ]
    return json.dumps(
        {
            "tool": "challenge",
            "description": (
                "Self-review task for you (the calling agent). Review the provided code "
                "against each criterion using your own judgment. Do NOT delegate to any "
                "external model — you perform the review."
            ),
            "instructions": (
                "For each criterion below, assess the code and respond with: "
                "PASS or FAIL, a one-sentence explanation, and a suggested fix if FAIL. "
                "Then provide an overall verdict and a summary of required changes."
            ),
            "criteria": review_criteria,
            "code_to_review": code,
        },
        indent=2,
    )


async def agentguard_list_archetypes() -> str:
    """List all available project archetypes with their descriptions."""
    from agentguard.archetypes.registry import get_archetype_registry

    registry = get_archetype_registry()
    archetypes = []
    for arch_id in registry.list_available():
        entry = registry.get_entry(arch_id)
        archetypes.append(
            {
                "id": entry.archetype.id,
                "name": entry.archetype.name,
                "description": entry.archetype.description,
                "trust_level": entry.trust_level.value,
                "content_hash": entry.content_hash,
            }
        )
    return json.dumps(archetypes, indent=2)


async def agentguard_get_archetype(name: str) -> str:
    """Get detailed configuration for a specific archetype.

    Includes tech stack, validation rules, and challenge criteria.
    """
    from agentguard.archetypes.registry import get_archetype_registry

    registry = get_archetype_registry()
    entry = registry.get_entry(name)
    arch = entry.archetype

    return json.dumps(
        {
            "id": arch.id,
            "name": arch.name,
            "description": arch.description,
            "version": arch.version,
            "maturity": getattr(arch, "maturity", "production"),
            "trust_level": entry.trust_level.value,
            "content_hash": entry.content_hash,
            "tech_stack": {
                "language": arch.tech_stack.language,
                "framework": arch.tech_stack.framework,
                "database": arch.tech_stack.database,
                "testing": arch.tech_stack.testing,
            },
            "pipeline_levels": arch.pipeline.levels,
            "validation_checks": arch.validation.checks,
            "challenge_criteria": arch.self_challenge.criteria,
            "infrastructure_files": getattr(arch, "infrastructure_files", []),
        },
        indent=2,
    )


async def agentguard_trace_summary(trace_id: str | None = None) -> str:
    """Get a summary of a generation trace: LLM calls, cost, validation results.

    If trace_id is omitted, returns info about the last trace (if available).
    """
    return json.dumps(
        {
            "note": "Trace lookup requires a trace store. "
            "Use the HTTP server with --trace-store for persistent traces.",
            "trace_id": trace_id,
        },
        indent=2,
    )


async def agentguard_benchmark(
    archetype: str = "api_backend",
    model: str = "",
    category: str | None = None,
    budget: float = 10.0,
) -> str:
    """Run a comparative benchmark for an archetype.

    Generates code WITH and WITHOUT AgentGuard across 5 complexity levels,
    evaluating enterprise and operational readiness. Returns the full
    benchmark report as JSON.

    Note: ``model`` is optional — if omitted, falls back to the model
    string configured on the pipeline / server.  When calling from an
    agent that already has an LLM, prefer the agent-native
    ``benchmark`` + ``benchmark_evaluate`` tools instead.
    """
    from agentguard.benchmark.catalog import get_default_specs
    from agentguard.benchmark.runner import BenchmarkRunner
    from agentguard.benchmark.types import BenchmarkConfig

    cat = category or archetype
    specs = get_default_specs(cat)
    config = BenchmarkConfig(specs=specs, model=model, budget_ceiling_usd=budget)
    runner = BenchmarkRunner(archetype=archetype, config=config, llm=model or None)
    report = await runner.run()
    return report.to_json()
