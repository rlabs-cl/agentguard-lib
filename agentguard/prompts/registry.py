"""Prompt registry — loads and manages prompt templates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agentguard.prompts.template import PromptTemplate

_BUILTIN_DIR = Path(__file__).parent / "builtin"


class PromptRegistry:
    """Registry for loading and caching prompt templates."""

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        self._loaded_builtins = False

    def get(self, template_id: str) -> PromptTemplate:
        """Get a prompt template by ID.

        Loads builtins on first access.

        Raises:
            KeyError: If template not found.
        """
        if not self._loaded_builtins:
            self._load_builtins()

        if template_id not in self._templates:
            raise KeyError(f"Prompt template '{template_id}' not found. Available: {list(self._templates.keys())}")

        return self._templates[template_id]

    def register(self, template: PromptTemplate) -> None:
        """Register a custom template (overrides builtins)."""
        self._templates[template.id] = template

    def list_available(self) -> list[str]:
        """List all available template IDs."""
        if not self._loaded_builtins:
            self._load_builtins()
        return sorted(self._templates.keys())

    def _load_builtins(self) -> None:
        """Load all YAML templates from the builtin directory."""
        if not _BUILTIN_DIR.exists():
            self._loaded_builtins = True
            return

        for yaml_file in _BUILTIN_DIR.glob("*.yaml"):
            data: dict[str, Any] = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
            template = PromptTemplate.from_dict(data)
            self._templates[template.id] = template

        self._loaded_builtins = True


# Module-level singleton
_default_registry: PromptRegistry | None = None


def get_prompt_registry() -> PromptRegistry:
    """Get the default prompt registry singleton."""
    global _default_registry
    if _default_registry is None:
        _default_registry = PromptRegistry()
    return _default_registry
