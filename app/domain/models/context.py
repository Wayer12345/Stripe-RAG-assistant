"""Domain model for the final LLM context bundle."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.models.retrieval_result import RetrievalResult
from app.domain.models.source import Source


class ContextBundle(BaseModel):
    """The assembled context passed to the generation layer.

    Validates that sources belong to included chunks, that the rendered
    context is present whenever chunks are provided, and that the token
    budget was respected.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str
    chunks: list[RetrievalResult]
    rendered_context: str
    token_count: int
    sources: list[Source] = Field(default_factory=list)
    # Maximum tokens the context builder was allowed to use.
    token_budget: int | None = None
    # True when context was cut short due to budget.
    truncated: bool = False
    # Chunk IDs that were candidates but excluded due to budget or policy.
    dropped_chunk_ids: list[str] = Field(default_factory=list)
    # Version tag for the rendered context format; useful for eval regression tracking.
    context_format_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def query_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty or whitespace-only.")
        return value

    @field_validator("token_count")
    @classmethod
    def token_count_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("token_count must be >= 0.")
        return value

    @field_validator("token_budget")
    @classmethod
    def token_budget_non_negative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("token_budget must be >= 0.")
        return value

    @field_validator("context_format_version")
    @classmethod
    def format_version_non_empty_when_provided(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("context_format_version must not be empty or whitespace-only.")
        return value

    @field_validator("dropped_chunk_ids")
    @classmethod
    def dropped_ids_no_empty_strings(cls, value: list[str]) -> list[str]:
        for chunk_id in value:
            if not chunk_id.strip():
                raise ValueError(
                    "dropped_chunk_ids must not contain empty or whitespace-only strings."
                )
        return value

    @model_validator(mode="after")
    def validate_bundle_consistency(self) -> "ContextBundle":
        if self.chunks and not self.rendered_context.strip():
            raise ValueError("rendered_context must be non-empty when chunks are present.")

        chunk_ids = {r.chunk_id for r in self.chunks}

        for source in self.sources:
            if source.chunk_id not in chunk_ids:
                raise ValueError(
                    f"Source chunk_id {source.chunk_id!r} is not present in chunks."
                )

        if self.token_budget is not None and self.token_count > self.token_budget:
            raise ValueError(
                f"token_count ({self.token_count}) exceeds token_budget ({self.token_budget})."
            )

        for dropped_id in self.dropped_chunk_ids:
            if dropped_id in chunk_ids:
                raise ValueError(
                    f"dropped_chunk_id {dropped_id!r} overlaps with an included chunk."
                )
        return self
