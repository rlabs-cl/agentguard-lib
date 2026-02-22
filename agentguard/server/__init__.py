"""Server module — FastAPI HTTP wrapper around the AgentGuard engine."""

from agentguard.server.app import create_app

__all__ = ["create_app"]
