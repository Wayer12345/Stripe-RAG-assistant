"""Health route definitions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import get_api_query_service, get_settings, get_warmup_result
from app.application.api_query_service import ApiQueryService, ApiQueryWarmupResult
from app.schemas.response import HealthResponse
from app.utils.config import Settings

router = APIRouter(tags=["health"])
SettingsDep = Annotated[Settings, Depends(get_settings)]
ApiServiceDep = Annotated[ApiQueryService, Depends(get_api_query_service)]
WarmupDep = Annotated[ApiQueryWarmupResult | None, Depends(get_warmup_result)]


@router.get("/health", response_model=HealthResponse)
def health_endpoint(
    request: Request,
    settings: SettingsDep,
    service: ApiServiceDep,
    warmup_result: WarmupDep,
) -> HealthResponse:
    """Return lightweight runtime health and warmup information."""
    dependencies: dict[str, Any] = {
        "service_initialized": service is not None,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if service.last_warmup_result is not None:
        generation_warm = service.last_warmup_result.components.get("generation", {})
        retrieval_warm = service.last_warmup_result.components.get("retrieval", {})
        dependencies["ollama"] = generation_warm.get("ollama_healthcheck_ok")
        dependencies["qdrant"] = retrieval_warm.get("qdrant_healthcheck_ok")

    app_state_warmup = warmup_result or getattr(request.app.state, "api_warmup_result", None)
    return HealthResponse(
        status="ok",
        app=settings.app.name,
        version=settings.api.version,
        warmup=app_state_warmup.__dict__ if app_state_warmup is not None else None,
        dependencies=dependencies,
    )
