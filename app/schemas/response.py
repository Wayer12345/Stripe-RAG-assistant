"""Response schemas for API routes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class SourceResponse(BaseModel):
    """Serialized source fields returned by query API."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    url: str | None = None
    section: str | None = None
    chunk_id: str | None = None
    document_id: str | None = None
    support_score: float | None = None
    source_type: str | None = None
    source_name: str | None = None
    source_path: str | None = None


class QueryDebugResponse(BaseModel):
    """Debug section for query response when requested."""

    model_config = ConfigDict(extra="forbid")

    trace_paths: dict[str, str | None] | None = None
    latency_ms: dict[str, int] | None = None
    stage_counts: dict[str, int | bool] | None = None
    context: dict[str, Any] | None = None


class QueryResponse(BaseModel):
    """Top-level successful query response schema."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    answer: str
    confidence: str
    sources: list[SourceResponse]
    debug: QueryDebugResponse | None = None


class HealthResponse(BaseModel):
    """Health endpoint response schema."""

    model_config = ConfigDict(extra="forbid")

    status: str
    app: str
    version: str
    warmup: dict[str, Any] | None = None
    dependencies: dict[str, Any] | None = None
