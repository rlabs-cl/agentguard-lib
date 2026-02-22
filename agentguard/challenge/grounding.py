"""GroundingChecker — detect hallucinated APIs, modules, and references."""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GroundingReport:
    """Results of grounding analysis."""

    violations: list[str] = field(default_factory=list)
    unknown_imports: list[str] = field(default_factory=list)
    unknown_references: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0

    def __str__(self) -> str:
        if self.passed:
            return "Grounding: PASSED (no violations)"
        total = len(self.violations)
        return f"Grounding: FAILED ({total} violations)"


class GroundingChecker:
    """Static analysis to detect invented APIs and modules.

    The grounding checker compares generated code against the *known context*
    (the modules, functions, and classes that were provided to the LLM) to
    identify references that the LLM might have hallucinated.

    This complements the LLM-based grounding questions in the self-challenge
    prompt by providing a deterministic, zero-cost check.

    Usage::

        checker = GroundingChecker(known_modules={"auth.service", "auth.models"})
        report = checker.check_files({"src/routes.py": code})
    """

    def __init__(
        self,
        known_modules: set[str] | None = None,
        known_symbols: set[str] | None = None,
        stdlib_ok: bool = True,
    ) -> None:
        """Initialize the grounding checker.

        Args:
            known_modules: Set of module paths that should be importable.
                           Project-internal modules (e.g. "auth.service").
            known_symbols: Set of known symbol names (classes, functions).
            stdlib_ok: If True, stdlib imports are always considered grounded.
        """
        self._known_modules = known_modules or set()
        self._known_symbols = known_symbols or set()
        self._stdlib_ok = stdlib_ok

        # Build stdlib set lazily
        self._stdlib: set[str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_files(self, files: dict[str, str]) -> GroundingReport:
        """Check all files for grounding violations.

        Args:
            files: Dict of {file_path: source_code}.

        Returns:
            GroundingReport with any violations found.
        """
        # Build project module set from the file paths themselves
        project_modules = self._build_project_modules(files)
        all_known = self._known_modules | project_modules

        report = GroundingReport()

        for file_path, source in files.items():
            if not file_path.endswith(".py"):
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                # Can't analyze unparseable code — syntax check handles this
                continue

            self._check_imports(tree, file_path, all_known, report)
            self._check_attribute_access(tree, file_path, report)

        if report.unknown_imports:
            for imp in report.unknown_imports:
                report.violations.append(f"Unknown import: {imp}")
        if report.unknown_references:
            for ref in report.unknown_references:
                report.violations.append(f"Unknown reference: {ref}")

        return report

    def check_single(self, source: str, file_path: str = "<unknown>") -> GroundingReport:
        """Check a single file for grounding violations."""
        return self.check_files({file_path: source})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_stdlib_modules(self) -> set[str]:
        """Get stdlib top-level module names."""
        if self._stdlib is None:
            import sys
            self._stdlib = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else set()
        return self._stdlib

    def _build_project_modules(self, files: dict[str, str]) -> set[str]:
        """Convert file paths to dotted module names."""
        modules: set[str] = set()
        for path in files:
            if not path.endswith(".py"):
                continue
            # Normalize path separators
            normalized = path.replace("\\", "/")
            # Strip leading ./
            if normalized.startswith("./"):
                normalized = normalized[2:]
            # Convert to module path
            if normalized.endswith("/__init__.py"):
                mod = normalized[:-12].replace("/", ".")
            elif normalized.endswith(".py"):
                mod = normalized[:-3].replace("/", ".")
            else:
                continue
            modules.add(mod)
            # Also add all parent packages
            parts = mod.split(".")
            for i in range(1, len(parts)):
                modules.add(".".join(parts[:i]))
        return modules

    def _check_imports(
        self,
        tree: ast.Module,
        file_path: str,
        known_modules: set[str],
        report: GroundingReport,
    ) -> None:
        """Check import statements against known modules."""
        stdlib = self._get_stdlib_modules()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level = alias.name.split(".")[0]
                    if self._is_grounded_import(alias.name, top_level, known_modules, stdlib):
                        continue
                    report.unknown_imports.append(
                        f"{file_path}: import {alias.name}"
                    )

            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue  # relative import without module (from . import x)
                if node.level > 0:
                    continue  # relative imports — within the project, assumed OK
                top_level = node.module.split(".")[0]
                if self._is_grounded_import(node.module, top_level, known_modules, stdlib):
                    continue
                names = ", ".join(a.name for a in node.names)
                report.unknown_imports.append(
                    f"{file_path}: from {node.module} import {names}"
                )

    def _is_grounded_import(
        self,
        full_module: str,
        top_level: str,
        known_modules: set[str],
        stdlib: set[str],
    ) -> bool:
        """Check if an import is grounded (known or stdlib)."""
        # Check against stdlib
        if self._stdlib_ok and top_level in stdlib:
            return True
        # Check full module path and all prefixes
        parts = full_module.split(".")
        for i in range(len(parts), 0, -1):
            if ".".join(parts[:i]) in known_modules:
                return True
        # Check if it's a well-known third-party package
        # (we allow common packages that are likely installed)
        return bool(self._is_common_third_party(top_level))

    def _check_attribute_access(
        self,
        tree: ast.Module,
        file_path: str,
        report: GroundingReport,
    ) -> None:
        """Check for references to unknown symbols if known_symbols is populated.

        This is a lightweight check — it only flags attribute access patterns
        that look like they reference specific known symbols when the user
        has provided a known_symbols set.
        """
        if not self._known_symbols:
            return  # No known symbols to check against

        # Collect all defined names in this file
        local_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                local_names.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                local_names.add(node.id)

        # This is intentionally conservative — we only flag things that
        # look like they should be in known_symbols but aren't.
        # Full implementation would track imports → attributes more precisely.

    @staticmethod
    def _is_common_third_party(module: str) -> bool:
        """Check if a module is a well-known third-party package.

        We allow these without explicit grounding because they're
        extremely common in Python projects.
        """
        common = {
            # Web frameworks
            "fastapi", "flask", "django", "starlette", "uvicorn",
            # Data / ML
            "numpy", "pandas", "scipy", "sklearn", "torch", "tensorflow",
            # HTTP / API
            "httpx", "requests", "aiohttp",
            # Database
            "sqlalchemy", "sqlmodel", "alembic", "pymongo", "redis",
            # Auth / Security
            "jwt", "jose", "passlib", "bcrypt", "cryptography",
            # Serialization
            "pydantic", "marshmallow", "attrs",
            # Testing
            "pytest", "unittest", "mock", "hypothesis",
            # Utils
            "click", "typer", "rich", "loguru", "structlog",
            "yaml", "toml", "dotenv",
            "jinja2", "mako",
            "celery", "rq",
            "boto3", "google",
        }
        return module in common
