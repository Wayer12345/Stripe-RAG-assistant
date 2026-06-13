"""Domain models for generated answers and confidence levels."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain.models.source import Source


class Confidence(StrEnum):
    """Allowed confidence buckets for a generated answer."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class GeneratedAnswer(BaseModel):
    """The output of the generation layer.

    Preserves raw LLM output for debugging and enforces that the answer
    string is non-empty when parsing succeeded.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    answer: str
    confidence: Confidence
    sources: list[Source] = Field(default_factory=list)
    raw_output: str
    parsed_successfully: bool
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("answer")
    @classmethod
    def answer_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must not be empty or whitespace-only.")
        return value
