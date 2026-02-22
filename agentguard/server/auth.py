"""API key authentication middleware."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.types import ASGIApp


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Simple API key middleware using X-Api-Key header.

    If *api_key* is None, authentication is disabled (local mode).
    """

    # Paths that never require authentication
    _PUBLIC_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json"})

    def __init__(self, app: ASGIApp, api_key: str | None = None) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # If no key configured, allow everything
        if self._api_key is None:
            return await call_next(request)

        # Public paths don't require auth
        if request.url.path in self._PUBLIC_PATHS:
            return await call_next(request)

        # Check header
        provided = request.headers.get("X-Api-Key", "")
        if not provided or not secrets.compare_digest(provided, self._api_key):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        return await call_next(request)
