"""Shared API envelope schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ErrorDetail(BaseModel):
    """Structured API error detail."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    request_id: str | None = None
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Structured API error response envelope."""

    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail


class IndexStatusResponse(BaseModel):
    """Index status summary for `/index/status`."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    collection_name: str
    collection_count: int | None = None
    embedding_model: str
    embedding_dimension: int | None = None
    index_manifest_path: str
    index_manifest: dict[str, Any] | None = None
