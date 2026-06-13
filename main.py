"""FastAPI app entrypoint for stripe-rag-assistant."""

from __future__ import annotations

from contextlib import asynccontextmanager

from app.api import debug_router, health_router, index_router, query_router
from app.application.api_query_service import ApiQueryService
from app.utils.config import Settings, load_settings
from app.utils.logging import get_logger, setup_logging
from fastapi import FastAPI

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]

    settings: Settings = load_settings()

    setup_logging(settings)

    app.state.settings = settings
    app.state.api_query_service = None
    app.state.api_warmup_result = None

    if settings.api.enabled:
        service = ApiQueryService(settings=settings)
        app.state.api_query_service = service

        if settings.api.warmup_on_startup:
            warmup_result = service.warmup()
            app.state.api_warmup_result = warmup_result

            if warmup_result.status != "success" and settings.api.fail_startup_on_warmup_error:
                raise RuntimeError(
                    "ApiQueryService warmup failed and fail_startup_on_warmup_error=true."
                )

    logger.info(
        "FastAPI startup complete: app=%s version=%s warmup_on_startup=%s",
        settings.app.name,
        settings.api.version,
        settings.api.warmup_on_startup,
    )

    yield

    service = getattr(app.state, "api_query_service", None)
    if service is not None and settings.api.shutdown_on_exit:
        service.shutdown()

    logger.info("FastAPI shutdown complete.")


_settings = load_settings()
setup_logging(_settings)

app = FastAPI(
    title=_settings.api.title,
    version=_settings.api.version,
    debug=_settings.api.debug,
    lifespan=lifespan,
)
app.include_router(health_router)
app.include_router(query_router)
app.include_router(index_router)
app.include_router(debug_router)
