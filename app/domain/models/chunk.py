"""Domain model for a retrieval-ready document chunk."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Chunk(BaseModel):
    """A retrieval-ready fragment of a Document.

    Preserves source traceability through document_id and carries enough
    metadata for retrieval, eval, and citation without embedding logic.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str
    document_id: str
    text: str
    chunk_index: int
    token_count: int
    # Hash of chunk text; required for deduplication, embedding cache, index rebuilds.
    content_hash: str
    # Name of the chunking strategy that produced this chunk (e.g. "heading_aware").
    chunking_strategy: str | None = None
    # Ordered list of heading labels from document root to this chunk's section.
    heading_path: list[str] = Field(default_factory=list)
    # Human-readable section label for source attribution and context rendering.
    section: str | None = None
    # 1-based page number; relevant for PDFs.
    page_number: int | None = None
    # 1-based source line range; relevant for TXT/Markdown.
    start_line: int | None = None
    end_line: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "document_id", "text", "content_hash")
    @classmethod
    def must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field must not be empty or whitespace-only.")
        return value

    @field_validator("chunking_strategy", "section")
    @classmethod
    def optional_str_non_empty_when_provided(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Field must not be empty or whitespace-only when provided.")
        return value

    @field_validator("chunk_index")
    @classmethod
    def chunk_index_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("chunk_index must be >= 0.")
        return value

    @field_validator("token_count")
    @classmethod
    def token_count_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("token_count must be > 0.")
        return value

    @field_validator("heading_path")
    @classmethod
    def heading_path_no_empty_strings(cls, value: list[str]) -> list[str]:
        for heading in value:
            if not heading.strip():
                raise ValueError("heading_path must not contain empty or whitespace-only strings.")
        return value

    @field_validator("page_number")
    @classmethod
    def page_number_gte_one(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("page_number must be >= 1.")
        return value

    @field_validator("start_line", "end_line")
    @classmethod
    def line_numbers_gte_one(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("start_line and end_line must be >= 1.")
        return value

    @model_validator(mode="after")
    def validate_offsets(self) -> "Chunk":
        if self.char_start is not None and self.char_end is not None:
            if self.char_start < 0:
                raise ValueError("char_start must be >= 0.")
            if self.char_end < self.char_start:
                raise ValueError("char_end must be >= char_start.")
        if self.start_line is not None and self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line.")
        return self
