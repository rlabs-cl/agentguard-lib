"""Auto-fix — automatically fix trivial issues in generated code."""

from __future__ import annotations

import ast
import logging
import subprocess
import tempfile
from pathlib import Path

from agentguard.validation.types import AutoFix

logger = logging.getLogger(__name__)


def autofix(files: dict[str, str]) -> tuple[dict[str, str], list[AutoFix]]:
    """Apply automatic fixes to generated code.

    Currently handles:
    - Remove unused imports (via AST analysis)
    - Ensure trailing newline
    - Format with Ruff (if available)

    Args:
        files: Dict of {file_path: file_content}.

    Returns:
        Tuple of (fixed_files, list_of_fixes_applied).
    """
    fixed_files = dict(files)
    all_fixes: list[AutoFix] = []

    for file_path in list(fixed_files.keys()):
        if not file_path.endswith(".py"):
            continue

        content = fixed_files[file_path]

        # Fix 1: Ensure trailing newline
        if content and not content.endswith("\n"):
            content += "\n"
            all_fixes.append(AutoFix(
                check="formatting",
                file_path=file_path,
                description="Added trailing newline",
            ))

        # Fix 2: Remove unused imports
        content, import_fixes = _remove_unused_imports(file_path, content)
        all_fixes.extend(import_fixes)

        fixed_files[file_path] = content

    # Fix 3: Run ruff format on all files (if available)
    fixed_files, format_fixes = _ruff_format(fixed_files)
    all_fixes.extend(format_fixes)

    return fixed_files, all_fixes


def _remove_unused_imports(file_path: str, content: str) -> tuple[str, list[AutoFix]]:
    """Remove imports that are never referenced in the code.

    Uses AST analysis to find imported names that aren't used anywhere.
    Preserves imports that are re-exported (__all__) or are side-effect imports.
    """
    fixes: list[AutoFix] = []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return content, fixes

    # Collect all imported names
    imported_names: dict[str, ast.stmt] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                imported_names[name] = node
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("__"):
                continue  # Skip dunder imports
            for alias in node.names:
                if alias.name == "*":
                    continue  # Don't touch star imports
                name = alias.asname or alias.name
                imported_names[name] = node

    if not imported_names:
        return content, fixes

    # Collect all used names (excluding the import statements themselves)
    used_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            # For chained attribute access like os.path, collect the root
            root: ast.expr = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                used_names.add(root.id)

    # Check for __all__ — those exports count as "used"
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            used_names.add(elt.value)

    # Find unused imports
    unused = set(imported_names.keys()) - used_names

    if not unused:
        return content, fixes

    # Remove unused import lines
    lines = content.split("\n")
    lines_to_remove: set[int] = set()

    for name in unused:
        node = imported_names[name]
        # Only remove if the entire import statement is unused
        # (don't partially remove `from x import a, b` if only `a` is unused)
        if isinstance(node, ast.Import) and len(node.names) == 1:
            lines_to_remove.add(node.lineno - 1)
            fixes.append(AutoFix(
                check="unused-import",
                file_path=file_path,
                description=f"Removed unused import '{name}'",
            ))
        elif isinstance(node, ast.ImportFrom) and len(node.names) == 1:
            lines_to_remove.add(node.lineno - 1)
            fixes.append(AutoFix(
                check="unused-import",
                file_path=file_path,
                description=f"Removed unused import '{name}' from '{node.module}'",
            ))

    if lines_to_remove:
        new_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]
        content = "\n".join(new_lines)

    return content, fixes


def _ruff_format(files: dict[str, str]) -> tuple[dict[str, str], list[AutoFix]]:
    """Run ruff format on files. Returns formatted files and list of fixes."""
    fixed = dict(files)
    fixes: list[AutoFix] = []

    py_files = {k: v for k, v in files.items() if k.endswith(".py")}
    if not py_files:
        return fixed, fixes

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            for file_path, content in py_files.items():
                full_path = tmpdir_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content, encoding="utf-8")

            # Run ruff format
            subprocess.run(
                ["ruff", "format", str(tmpdir_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Read back formatted files
            for file_path in py_files:
                full_path = tmpdir_path / file_path
                if full_path.exists():
                    formatted = full_path.read_text(encoding="utf-8")
                    if formatted != files[file_path]:
                        fixed[file_path] = formatted
                        fixes.append(AutoFix(
                            check="formatting",
                            file_path=file_path,
                            description="Reformatted with Ruff",
                        ))

    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug("Ruff format not available or timed out: %s", e)

    return fixed, fixes
