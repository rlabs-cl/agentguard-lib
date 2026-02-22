"""FastAPI application factory — creates the AgentGuard HTTP server."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agentguard._version import __version__
from agentguard.server.auth import ApiKeyMiddleware
from agentguard.server.schemas import HealthResponse, ProblemDetail

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — startup / shutdown hooks."""
    logger.info("AgentGuard server starting (v%s)", __version__)
    yield
    logger.info("AgentGuard server shutting down")


def create_app(
    *,
    api_key: str | None = None,
    trace_store: str | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create the AgentGuard FastAPI application.

    Args:
        api_key: Optional API key for ``X-Api-Key`` header auth.
                 ``None`` disables authentication (local mode).
        trace_store: Directory to persist trace JSON files.
        cors_origins: Allowed CORS origins. ``["*"]`` if not set.

    Returns:
        Configured ``FastAPI`` instance.
    """
    app = FastAPI(
        title="AgentGuard",
        description="Quality-assured LLM code generation engine",
        version=__version__,
        lifespan=_lifespan,
    )

    # ── State shared with route handlers ──────────────────────────────
    app.state.trace_store = trace_store

    # ── CORS ──────────────────────────────────────────────────────────
    origins = cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── API key auth ──────────────────────────────────────────────────
    app.add_middleware(ApiKeyMiddleware, api_key=api_key)

    # ── RFC 7807 error handlers ───────────────────────────────────────

    @app.exception_handler(404)
    async def not_found(request: Request, exc: Any) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=ProblemDetail(
                type="about:blank",
                title="Not Found",
                status=404,
                detail=str(exc.detail) if hasattr(exc, "detail") else "Resource not found",
                instance=str(request.url),
            ).model_dump(),
        )

    @app.exception_handler(422)
    async def validation_error(request: Request, exc: Any) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ProblemDetail(
                type="about:blank",
                title="Validation Error",
                status=422,
                detail=str(exc),
                instance=str(request.url),
            ).model_dump(),
        )

    @app.exception_handler(500)
    async def internal_error(request: Request, exc: Any) -> JSONResponse:
        logger.exception("Internal server error")
        return JSONResponse(
            status_code=500,
            content=ProblemDetail(
                type="about:blank",
                title="Internal Server Error",
                status=500,
                detail="An unexpected error occurred.",
                instance=str(request.url),
            ).model_dump(),
        )

    # ── Health endpoint (always public) ───────────────────────────────

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    # ── Register REST routes ──────────────────────────────────────────
    from agentguard.server.routes import router  # noqa: E402

    app.include_router(router, prefix="/v1")

    return app
