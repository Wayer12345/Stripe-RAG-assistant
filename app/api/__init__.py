"""API router exports."""

from app.api.routes_debug import router as debug_router
from app.api.routes_health import router as health_router
from app.api.routes_index import router as index_router
from app.api.routes_query import router as query_router

__all__ = ["debug_router", "health_router", "index_router", "query_router"]
