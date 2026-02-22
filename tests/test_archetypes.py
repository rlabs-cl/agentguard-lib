"""Tests for the archetypes module."""

from __future__ import annotations

import pytest

from agentguard.archetypes.base import Archetype
from agentguard.archetypes.registry import ArchetypeRegistry, get_archetype_registry


class TestArchetype:
    def test_load_api_backend(self):
        arch = Archetype.load("api_backend")
        assert arch.id == "api_backend"
        assert arch.tech_stack.language == "python"
        assert arch.tech_stack.framework == "fastapi"

    def test_load_unknown_raises(self):
        with pytest.raises(KeyError):
            Archetype.load("nonexistent_archetype_xyz")

    def test_list_available(self):
        available = Archetype.list_available()
        assert "api_backend" in available

    def test_pipeline_config(self):
        arch = Archetype.load("api_backend")
        assert len(arch.pipeline.levels) == 4
        assert arch.pipeline.levels == ["skeleton", "contracts", "wiring", "logic"]

    def test_validation_config(self):
        arch = Archetype.load("api_backend")
        assert arch.validation is not None
        checks = arch.validation.checks
        assert len(checks) > 0

    def test_challenge_config(self):
        arch = Archetype.load("api_backend")
        assert arch.self_challenge is not None
        assert len(arch.self_challenge.criteria) > 0

    def test_expected_structure(self):
        arch = Archetype.load("api_backend")
        text = arch.get_expected_structure_text()
        assert len(text) > 0


class TestArchetypeRegistry:
    def test_registry_loads_builtins(self):
        reg = get_archetype_registry()
        arch = reg.get("api_backend")
        assert arch.id == "api_backend"

    def test_registry_unknown_raises(self):
        reg = ArchetypeRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")
