"""Domain model for a candidate chunk returned by retrieval or reranking."""

import math
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.models.source import Source


class RetrievalMethod(StrEnum):
    """Which retrieval pathway produced this result."""

    DENSE = "dense"
    LEXICAL = "lexical"
    HYBRID = "hybrid"
    RERANKED = "reranked"
    FILTERED = "filtered"
    FALLBACK = "fallback"


class RetrievalResult(BaseModel):
    """A retrieval candidate with scores from each ranking stage.

    Preserves dense, lexical, reranker, and final scores so downstream layers
    can inspect or re-rank without losing signal.  Rank fields support eval
    metrics such as MRR, nDCG, and HitRate.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    chunk_id: str
    document_id: str
    text: str
    source: Source
    retrieval_score: float | None = None
    lexical_score: float | None = None
    dense_score: float | None = None
    reranker_score: float | None = None
    final_score: float
    # Which retrieval pathway produced this result.
    retrieval_method: RetrievalMethod | None = None
    # 1-based rank after the current retrieval/reranking stage.
    rank: int | None = None
    # 1-based rank before reranking or fusion, preserved for debug comparison.
    original_rank: int | None = None
    # BM25 / lexical matched terms, useful for debug and exact-term failure analysis.
    matched_terms: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunk_id", "document_id", "text")
    @classmethod
    def must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field must not be empty or whitespace-only.")
        return value

    @field_validator("final_score")
    @classmethod
    def final_score_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("final_score must be a finite number.")
        return value

    @field_validator("retrieval_score", "lexical_score", "dense_score", "reranker_score")
    @classmethod
    def optional_scores_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Score must be a finite number when provided.")
        return value

    @field_validator("rank", "original_rank")
    @classmethod
    def ranks_gte_one(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("rank and original_rank must be >= 1.")
        return value

    @field_validator("matched_terms")
    @classmethod
    def matched_terms_valid(cls, value: list[str]) -> list[str]:
        seen: list[str] = []
        seen_set: set[str] = set()
        for term in value:
            if not term.strip():
                raise ValueError("matched_terms must not contain empty or whitespace-only strings.")
            if term not in seen_set:
                seen.append(term)
                seen_set.add(term)
        return seen

    @model_validator(mode="after")
    def source_ids_consistent(self) -> "RetrievalResult":
        if self.source.chunk_id != self.chunk_id:
            raise ValueError(
                f"source.chunk_id ({self.source.chunk_id!r}) must equal "
                f"chunk_id ({self.chunk_id!r})."
            )
        if self.source.document_id != self.document_id:
            raise ValueError(
                f"source.document_id ({self.source.document_id!r}) must equal "
                f"document_id ({self.document_id!r})."
            )
        return self
