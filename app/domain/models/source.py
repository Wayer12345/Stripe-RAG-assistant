"""Domain model for a source that supports an answer."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Source(BaseModel):
    """A source document or chunk that supports a generated answer.

    Sources must come from retrieved chunks; they must never be invented.
    Supports both URL-based and local file-based sources.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    title: str
    url: str | None = None
    section: str | None = None
    chunk_id: str
    document_id: str
    support_score: float | None = None
    # Local-file source identity fields; may be absent for web-sourced documents.
    source_path: str | None = None
    source_type: str | None = None
    source_name: str | None = None
    # Ordered heading labels from document root to this chunk's section.
    heading_path: list[str] = Field(default_factory=list)
    # 1-based page number; relevant for PDFs.
    page_number: int | None = None
    # 1-based source line range; relevant for TXT/Markdown.
    start_line: int | None = None
    end_line: int | None = None

    @field_validator("title", "chunk_id", "document_id")
    @classmethod
    def must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field must not be empty or whitespace-only.")
        return value

    @field_validator("source_path", "source_type", "source_name")
    @classmethod
    def optional_str_non_empty_when_provided(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Field must not be empty or whitespace-only when provided.")
        return value

    @field_validator("support_score")
    @classmethod
    def score_in_range(cls, value: float | None) -> float | None:
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError("support_score must be between 0.0 and 1.0 inclusive.")
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
    def validate_line_range(self) -> "Source":
        if self.start_line is not None and self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line.")
        return self
