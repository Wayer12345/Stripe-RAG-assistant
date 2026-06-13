"""FastAPI dependency providers for API runtime state."""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from app.application.api_query_service import ApiQueryService, ApiQueryWarmupResult
from app.utils.config import Settings


def get_settings(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Settings are not initialized.",
        )
    return settings


def get_api_query_service(request: Request) -> ApiQueryService:
    service = getattr(request.app.state, "api_query_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API query service is not initialized.",
        )
    return service


def get_warmup_result(request: Request) -> ApiQueryWarmupResult | None:
    return getattr(request.app.state, "api_warmup_result", None)
