"""Domain model for an evaluation test case."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Difficulty(StrEnum):
    """Difficulty level of an evaluation question."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVERSARIAL = "adversarial"


class EvalCaseType(StrEnum):
    """Primary taxonomy type for an eval question.

    Enables slicing metrics by question type (factual, procedural, etc.).
    """

    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    COMPARISON = "comparison"
    MULTI_HOP = "multi_hop"
    DEFINITION = "definition"
    CITATION_SENSITIVE = "citation_sensitive"
    UNANSWERABLE = "unanswerable"
    OOD = "ood"
    AMBIGUOUS = "ambiguous"
    TYPO = "typo"


class EvalCase(BaseModel):
    """A single evaluation test case with ground-truth labels.

    Early eval datasets may lack gold sources, so empty expected_sources
    is allowed even when is_answerable is True.  Expected chunk IDs are also
    optional to support incremental dataset enrichment.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str
    question: str
    expected_answer: str | None = None
    expected_sources: list[str] = Field(default_factory=list)
    # Gold chunk IDs for chunk-level retrieval eval; optional for early datasets.
    expected_chunk_ids: list[str] = Field(default_factory=list)
    # Gold source titles for title-level retrieval eval; optional for early datasets.
    expected_source_titles: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    difficulty: Difficulty
    # Primary taxonomy type; optional for backward compatibility.
    case_type: EvalCaseType | None = None
    is_answerable: bool
    # Free-form human notes for answer/citation evaluation.
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "question")
    @classmethod
    def must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field must not be empty or whitespace-only.")
        return value

    @field_validator("notes")
    @classmethod
    def notes_non_empty_when_provided(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("notes must not be empty or whitespace-only when provided.")
        return value

    @field_validator("tags")
    @classmethod
    def tags_no_empty_strings(cls, value: list[str]) -> list[str]:
        for tag in value:
            if not tag.strip():
                raise ValueError("tags must not contain empty or whitespace-only strings.")
        return value

    @field_validator("expected_sources")
    @classmethod
    def sources_no_empty_strings(cls, value: list[str]) -> list[str]:
        for source in value:
            if not source.strip():
                raise ValueError(
                    "expected_sources must not contain empty or whitespace-only strings."
                )
        return value

    @field_validator("expected_chunk_ids")
    @classmethod
    def chunk_ids_no_empty_strings(cls, value: list[str]) -> list[str]:
        for chunk_id in value:
            if not chunk_id.strip():
                raise ValueError(
                    "expected_chunk_ids must not contain empty or whitespace-only strings."
                )
        return value

    @field_validator("expected_source_titles")
    @classmethod
    def source_titles_no_empty_strings(cls, value: list[str]) -> list[str]:
        for title in value:
            if not title.strip():
                raise ValueError(
                    "expected_source_titles must not contain empty or whitespace-only strings."
                )
        return value
