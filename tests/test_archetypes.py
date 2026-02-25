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


class TestDebugArchetypes:
    """Tests for debug_backend and debug_frontend builtin archetypes."""

    def test_debug_backend_loads(self):
        arch = Archetype.load("debug_backend")
        assert arch.id == "debug_backend"
        assert arch.tech_stack.language == "python"

    def test_debug_frontend_loads(self):
        arch = Archetype.load("debug_frontend")
        assert arch.id == "debug_frontend"
        assert arch.tech_stack.language == "typescript"

    def test_debug_archetypes_in_registry(self):
        available = Archetype.list_available()
        assert "debug_backend" in available
        assert "debug_frontend" in available

    def test_debug_backend_has_debug_config(self):
        arch = Archetype.load("debug_backend")
        assert len(arch.debug_config.data_sources) > 0
        assert len(arch.debug_config.hypothesis_protocol) > 0
        assert len(arch.debug_config.fix_protocol) > 0
        assert len(arch.debug_config.escalation_criteria) > 0

    def test_debug_frontend_has_debug_config(self):
        arch = Archetype.load("debug_frontend")
        assert len(arch.debug_config.data_sources) > 0
        assert len(arch.debug_config.escalation_criteria) > 0

    def test_archetypes_without_debug_config_have_empty_defaults(self):
        arch = Archetype.load("api_backend")
        # api_backend has no debug_config block — should default to empty lists
        assert arch.debug_config.data_sources == []
        assert arch.debug_config.escalation_criteria == []

    def test_archetypes_without_migration_config_have_empty_defaults(self):
        arch = Archetype.load("api_backend")
        assert arch.migration_config.risk_areas == []
        assert arch.migration_config.step_order == []


class TestDebugMigrationSchema:
    """Tests for DebugConfig and MigrationConfig dataclasses."""

    def test_debug_config_defaults(self):
        from agentguard.archetypes.base import DebugConfig

        cfg = DebugConfig()
        assert cfg.data_sources == []
        assert cfg.hypothesis_protocol == []
        assert cfg.fix_protocol == []
        assert cfg.escalation_criteria == []

    def test_migration_config_defaults(self):
        from agentguard.archetypes.base import MigrationConfig

        cfg = MigrationConfig()
        assert cfg.risk_areas == []
        assert cfg.concern_protocol == []
        assert cfg.incompatibility_signals == []
        assert cfg.step_order == []

