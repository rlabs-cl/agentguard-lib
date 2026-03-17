"""Backward compatibility shim for benchmark runner (v0.6.0).

LLM providers have been removed from the library. The benchmark runner
still accepts a model string for convenience, but it will raise a clear
error directing callers to pass an LLM instance instead.
"""

from __future__ import annotations

from typing import Any


def create_llm_provider(model: str) -> Any:
    """Raise an error — LLM providers are no longer bundled.

    In v0.6.0, callers must pass an already-instantiated LLM object
    (any object with an async ``generate(messages, config)`` method)
    rather than a model string.
    """
    raise ImportError(
        f"LLM providers were removed in AgentGuard v0.6.0. "
        f"Cannot auto-create provider for model string '{model}'. "
        f"Pass an already-instantiated LLM object to the BenchmarkRunner "
        f"instead of a model string."
    )
