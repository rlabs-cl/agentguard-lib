"""Import check — verify all imports resolve to real modules."""

from __future__ import annotations

import ast
import importlib.util
import logging
import sys
import time

from agentguard.validation.types import CheckResult, Severity, ValidationError

logger = logging.getLogger(__name__)


def check_imports(files: dict[str, str]) -> CheckResult:
    """Check that all imports in generated Python files are resolvable.

    For each import, checks:
    1. Is it a standard library module?
    2. Is it an installed third-party package?
    3. Is it a local project import (relative to other generated files)?

    Args:
        files: Dict of {file_path: file_content}.

    Returns:
        CheckResult with phantom import errors.
    """
    start = time.perf_counter()
    py_files = {k: v for k, v in files.items() if k.endswith(".py")}

    if not py_files:
        return CheckResult(
            check="imports",
            passed=True,
            details="No Python files to check",
            duration_ms=0,
        )

    # Build set of project-internal module names
    project_modules = _build_project_module_set(py_files)

    errors: list[ValidationError] = []

    for file_path, content in py_files.items():
        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError:
            continue  # Syntax check handles this

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not _can_resolve(alias.name, project_modules):
                        errors.append(
                            ValidationError(
                                check="imports",
                                file_path=file_path,
                                line=node.lineno,
                                message=f"Cannot resolve import '{alias.name}'",
                                severity=Severity.ERROR,
                            )
                        )

            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    # Relative import — check within project
                    base = _resolve_relative_import(file_path, node.module or "", node.level)
                    if base and not _is_project_module(base, project_modules):
                        errors.append(
                            ValidationError(
                                check="imports",
                                file_path=file_path,
                                line=node.lineno,
                                message=f"Cannot resolve relative import '.{'.' * (node.level - 1)}{node.module or ''}'",
                                severity=Severity.ERROR,
                            )
                        )
                elif node.module:
                    if not _can_resolve(node.module, project_modules):
                        errors.append(
                            ValidationError(
                                check="imports",
                                file_path=file_path,
                                line=node.lineno,
                                message=f"Cannot resolve import '{node.module}'",
                                severity=Severity.ERROR,
                            )
                        )

    duration_ms = int((time.perf_counter() - start) * 1000)
    return CheckResult(
        check="imports",
        passed=len(errors) == 0,
        details=f"Checked imports in {len(py_files)} files, {len(errors)} unresolvable",
        errors=errors,
        duration_ms=duration_ms,
    )


def _build_project_module_set(files: dict[str, str]) -> set[str]:
    """Build a set of module dotted names from the project file paths.

    For example, 'src/myapp/models/user.py' → {'src', 'src.myapp', 'src.myapp.models', 'src.myapp.models.user'}
    """
    modules: set[str] = set()
    for file_path in files:
        if not file_path.endswith(".py"):
            continue
        # Convert path to module name
        parts = file_path.replace("\\", "/").split("/")
        # Remove .py from last part
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1].removesuffix(".py")

        # Add all prefixes as valid modules
        for i in range(1, len(parts) + 1):
            modules.add(".".join(parts[:i]))

    return modules


def _can_resolve(module_name: str, project_modules: set[str]) -> bool:
    """Check if a module name can be resolved.

    Checks project modules first, then stdlib/installed packages.
    """
    # Check project-internal modules
    if _is_project_module(module_name, project_modules):
        return True

    # Check top-level module in stdlib/installed
    top_level = module_name.split(".")[0]
    return _is_importable(top_level)


def _is_project_module(module_name: str, project_modules: set[str]) -> bool:
    """Check if module_name matches any project-internal module."""
    # Direct match
    if module_name in project_modules:
        return True
    # Check if top-level matches (e.g. 'models.user' when 'models' is in project)
    parts = module_name.split(".")
    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in project_modules:
            return True
    return False


def _is_importable(module_name: str) -> bool:
    """Check if a top-level module is importable (stdlib or installed)."""
    # Fast check: is it in sys.modules?
    if module_name in sys.modules:
        return True

    # Check if it's findable
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _resolve_relative_import(
    file_path: str, module: str, level: int
) -> str | None:
    """Resolve a relative import to an absolute module name."""
    parts = file_path.replace("\\", "/").split("/")

    # Remove filename
    if parts[-1].endswith(".py"):
        parts = parts[:-1]

    # Go up `level` directories
    if level > len(parts):
        return None
    base_parts = parts[: -level] if level > 0 else parts

    if module:
        return ".".join(base_parts) + "." + module
    return ".".join(base_parts) if base_parts else None
