"""Domain interface contract for loading raw source documents."""

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RawDocument(BaseModel):
    """Raw source payload produced by a document loader.

    This value object intentionally stays inside the interface module so
    loaders can pass bytes to parsers without introducing a separate
    domain model file yet.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    source_type: str
    content: bytes
    source_path: str | None = None
    source_name: str | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_type")
    @classmethod
    def source_type_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source_type must not be empty or whitespace-only.")
        return value

    @field_validator("content")
    @classmethod
    def content_non_empty(cls, value: bytes) -> bytes:
        if not value:
            raise ValueError("content must not be empty.")
        return value

    @field_validator("source_path", "source_name", "mime_type")
    @classmethod
    def optional_fields_non_empty_when_provided(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Field must not be empty or whitespace-only when provided.")
        return value


class DocumentLoader(Protocol):
    """Loads raw source files into in-memory bytes payloads."""

    def load(self) -> list[RawDocument]:
        """Return loaded raw documents ready for parser selection."""
        ...

