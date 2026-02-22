"""PromptTemplate — versioned Jinja2-based prompt rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jinja2

from agentguard.llm.types import Message


@dataclass
class PromptTemplate:
    """A versioned prompt template with system and user parts.

    Templates use Jinja2 syntax for variable substitution.
    """

    id: str
    version: str
    description: str = ""
    system_template: str = ""
    user_template: str = ""
    variables: list[str] = field(default_factory=list)

    _jinja_env: jinja2.Environment = field(
        default_factory=lambda: jinja2.Environment(
            undefined=jinja2.StrictUndefined,
            keep_trailing_newline=True,
        ),
        repr=False,
    )

    def render(self, **kwargs: Any) -> list[Message]:
        """Render the template into a list of Messages.

        Args:
            **kwargs: Template variables to substitute.

        Returns:
            List of Message objects (system + user).

        Raises:
            jinja2.UndefinedError: If a required variable is missing.
        """
        messages: list[Message] = []

        if self.system_template:
            system_tmpl = self._jinja_env.from_string(self.system_template)
            system_text = system_tmpl.render(**kwargs).strip()
            if system_text:
                messages.append(Message(role="system", content=system_text))

        if self.user_template:
            user_tmpl = self._jinja_env.from_string(self.user_template)
            user_text = user_tmpl.render(**kwargs).strip()
            if user_text:
                messages.append(Message(role="user", content=user_text))

        return messages

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptTemplate:
        """Create a PromptTemplate from a dictionary (e.g. loaded from YAML)."""
        return cls(
            id=data["id"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            system_template=data.get("system", ""),
            user_template=data.get("user", ""),
            variables=data.get("variables", []),
        )
