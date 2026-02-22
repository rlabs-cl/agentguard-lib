"""Tests for the prompts module."""

from __future__ import annotations

import jinja2
import pytest

from agentguard.prompts.registry import PromptRegistry, get_prompt_registry
from agentguard.prompts.template import PromptTemplate


class TestPromptTemplate:
    def test_render_simple(self):
        tpl = PromptTemplate(
            id="test",
            version="1.0.0",
            description="A test template",
            system_template="You are a {{ role }}.",
            user_template="Generate {{ thing }}.",
        )
        messages = tpl.render(role="coder", thing="code")
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert "coder" in messages[0].content
        assert messages[1].role == "user"
        assert "code" in messages[1].content

    def test_render_no_system(self):
        tpl = PromptTemplate(
            id="test",
            version="1.0.0",
            description="No system",
            system_template="",
            user_template="Hello {{ name }}.",
        )
        messages = tpl.render(name="world")
        assert len(messages) == 1
        assert messages[0].role == "user"

    def test_render_strict_undefined(self):
        tpl = PromptTemplate(
            id="test",
            version="1.0.0",
            description="Strict",
            system_template="",
            user_template="Hello {{ missing_var }}.",
        )
        with pytest.raises(jinja2.UndefinedError):
            tpl.render()

    def test_from_dict(self):
        d = {
            "id": "test",
            "version": "1.0.0",
            "description": "Test template",
            "system": "sys",
            "user": "usr",
        }
        tpl = PromptTemplate.from_dict(d)
        assert tpl.id == "test"
        assert tpl.system_template == "sys"


class TestPromptRegistry:
    def test_register_and_get(self):
        reg = PromptRegistry()
        tpl = PromptTemplate(id="custom", version="1.0.0", description="Custom", user_template="Hello")
        reg.register(tpl)
        assert reg.get("custom") is tpl

    def test_get_unknown_raises(self):
        reg = PromptRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_builtin_templates_loaded(self):
        reg = get_prompt_registry()
        available = reg.list_available()
        assert "skeleton" in available
        assert "contracts" in available
        assert "wiring" in available
        assert "logic" in available

    def test_builtin_skeleton_renders(self):
        reg = get_prompt_registry()
        tpl = reg.get("skeleton")
        messages = tpl.render(
            spec="A user auth API",
            archetype_name="api_backend",
            language="python",
            framework="fastapi",
            expected_structure="main.py\nmodels.py",
        )
        assert len(messages) >= 1
        assert "user auth" in messages[-1].content.lower() or "user auth" in messages[-1].content
