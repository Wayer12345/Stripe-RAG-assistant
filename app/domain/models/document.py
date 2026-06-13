"""Domain model for a normalized ingested document."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentProcessingStage(StrEnum):
    """Which pipeline stage this Document artifact represents."""

    RAW = "raw"
    PARSED = "parsed"
    CLEANED = "cleaned"


class Document(BaseModel):
    """A normalized document after ingestion, parsing, and cleaning.

    Validation enforces non-empty required strings and preserves all source
    metadata without performing any IO or hashing logic.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str
    source_type: str
    source_path: str | None = None
    url: str | None = None
    title: str | None = None
    # Optional stable logical identifier for the source (e.g. file stem, URL hash).
    source_id: str | None = None
    # Human-readable source name (e.g. filename, page title, guide name).
    source_name: str | None = None
    # Detected MIME type when available (e.g. "application/pdf", "text/html").
    source_mime_type: str | None = None
    processing_stage: DocumentProcessingStage = DocumentProcessingStage.PARSED
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_hash: str
    created_at: datetime

    @field_validator("id", "source_type", "text", "content_hash")
    @classmethod
    def must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field must not be empty or whitespace-only.")
        return value

    @field_validator("source_id", "source_name", "source_mime_type")
    @classmethod
    def optional_str_non_empty_when_provided(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Field must not be empty or whitespace-only when provided.")
        return value
