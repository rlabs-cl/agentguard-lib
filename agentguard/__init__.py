"""AgentGuard — A quality-assurance engine for LLM-generated code."""

from agentguard._version import __version__
from agentguard.archetypes.base import Archetype
from agentguard.challenge.challenger import SelfChallenger
from agentguard.challenge.grounding import GroundingChecker
from agentguard.challenge.types import ChallengeResult, CriterionResult
from agentguard.context.recipe import ContextBundle, ContextEngine, ContextRecipe
from agentguard.context.window import TokenWindow
from agentguard.llm.base import LLMProvider
from agentguard.llm.factory import create_llm_provider
from agentguard.llm.types import CostEstimate, LLMResponse, TokenUsage
from agentguard.pipeline import Pipeline
from agentguard.prompts.template import PromptTemplate
from agentguard.topdown.generator import TopDownGenerator
from agentguard.tracing.trace import Span, SpanType, Trace
from agentguard.tracing.tracer import Tracer
from agentguard.validation.types import CheckResult, ValidationError, ValidationReport
from agentguard.validation.validator import Validator

__all__ = [
    "__version__",
    # LLM
    "LLMProvider",
    "LLMResponse",
    "TokenUsage",
    "CostEstimate",
    "create_llm_provider",
    # Archetypes
    "Archetype",
    # Tracing
    "Tracer",
    "Trace",
    "Span",
    "SpanType",
    # Prompts
    "PromptTemplate",
    # Generation
    "TopDownGenerator",
    "Pipeline",
    # Validation (Phase 1)
    "ValidationReport",
    "CheckResult",
    "ValidationError",
    "Validator",
    # Challenge (Phase 1)
    "ChallengeResult",
    "CriterionResult",
    "SelfChallenger",
    "GroundingChecker",
    # Context (Phase 1)
    "ContextBundle",
    "ContextEngine",
    "ContextRecipe",
    "TokenWindow",
    # Platform integration (Phase 5)
    "PlatformClient",
    "PlatformConfig",
    # Server (Phase 2) — lazy imports
    "create_app",
]


def create_app(**kwargs):  # type: ignore[no-untyped-def]
    """Lazy import for the HTTP server app factory."""
    from agentguard.server.app import create_app as _create_app

    return _create_app(**kwargs)


# Lazy import for platform client (optional httpx dependency)
def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "PlatformClient":
        from agentguard.platform.client import PlatformClient
        return PlatformClient
    if name == "PlatformConfig":
        from agentguard.platform.config import PlatformConfig
        return PlatformConfig
    raise AttributeError(f"module 'agentguard' has no attribute {name!r}")
