"""AgentGuard — A quality-assurance engine for LLM-generated code."""

from agentguard._version import __version__
from agentguard.archetypes.base import Archetype
from agentguard.context.recipe import ContextBundle, ContextEngine, ContextRecipe
from agentguard.context.window import TokenWindow
from agentguard.prompts.template import PromptTemplate
from agentguard.tracing.trace import Span, SpanType, Trace
from agentguard.tracing.tracer import Tracer
from agentguard.validation.types import CheckResult, ValidationError, ValidationReport
from agentguard.validation.validator import Validator

__all__ = [
    "__version__",
    # Archetypes
    "Archetype",
    # Tracing
    "Tracer",
    "Trace",
    "Span",
    "SpanType",
    # Prompts
    "PromptTemplate",
    # Validation
    "ValidationReport",
    "CheckResult",
    "ValidationError",
    "Validator",
    # Context
    "ContextBundle",
    "ContextEngine",
    "ContextRecipe",
    "TokenWindow",
    # Platform integration — lazy imports
    "PlatformClient",
    "PlatformConfig",
]


# Lazy import for platform client (optional httpx dependency)
def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "PlatformClient":
        from agentguard.platform.client import PlatformClient
        return PlatformClient
    if name == "PlatformConfig":
        from agentguard.platform.config import PlatformConfig
        return PlatformConfig
    raise AttributeError(f"module 'agentguard' has no attribute {name!r}")
