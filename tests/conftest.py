"""conftest.py — shared test fixtures for AgentGuard."""

from __future__ import annotations

import pytest

from agentguard.archetypes.base import Archetype


@pytest.fixture
def api_backend_archetype() -> Archetype:
    """Load the builtin api_backend archetype."""
    return Archetype.load("api_backend")
