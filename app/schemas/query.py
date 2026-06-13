"""Request schemas for online query API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


class QueryFilters(BaseModel):
    """Supported metadata filters for query-time retrieval."""

    model_config = ConfigDict(extra="forbid")

    document_id: str | None = None
    source_type: str | None = None
    source_path: str | None = None
    url: str | None = None
    category: str | None = None
    section: str | None = None
    content_hash: str | None = None

    @field_validator(
        "document_id",
        "source_type",
        "source_path",
        "url",
        "category",
        "section",
        "content_hash",
        mode="before",
    )
    @classmethod
    def _normalize_text_filters(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            return _normalize_optional(value)
        return value

    def as_dict(self) -> dict[str, str]:
        payload = self.model_dump(mode="json", exclude_none=True)
        return {str(key): str(value) for key, value in payload.items()}


class QueryDebugOptions(BaseModel):
    """Optional debug toggles for query responses."""

    model_config = ConfigDict(extra="forbid")

    include_trace: bool = True
    include_stage_counts: bool = True
    include_latency: bool = True
    include_sources: bool = True
    include_retrieved_preview: bool = False
    include_context_preview: bool = False


class QueryRequest(BaseModel):
    """Incoming API request payload for `/query`."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1)
    filters: QueryFilters | None = None
    retrieve_top_k: int | None = None
    rerank_top_k_before: int | None = None
    rerank_top_k_after: int | None = None
    context_token_budget: int | None = None
    context_max_chunks: int | None = None
    debug: QueryDebugOptions | None = None
    write_trace: bool | None = None

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("question must not be empty.")
        return trimmed

    @field_validator(
        "retrieve_top_k",
        "rerank_top_k_before",
        "rerank_top_k_after",
        "context_token_budget",
        "context_max_chunks",
    )
    @classmethod
    def _validate_positive_overrides(cls, value: int | None, info: object) -> int | None:
        if value is not None and value <= 0:
            field_name = getattr(info, "field_name", "value")
            raise ValueError(f"{field_name} must be > 0 when provided.")
        return value

    @model_validator(mode="after")
    def _validate_rerank_bounds(self) -> QueryRequest:
        if (
            self.rerank_top_k_before is not None
            and self.rerank_top_k_after is not None
            and self.rerank_top_k_after > self.rerank_top_k_before
        ):
            raise ValueError("rerank_top_k_after must be <= rerank_top_k_before.")
        return self
