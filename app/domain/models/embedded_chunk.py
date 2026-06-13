"""Domain model for a chunk with an embedding vector."""

import math

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from app.domain.models.chunk import Chunk


class EmbeddedChunk(BaseModel):
    """A Chunk paired with a dense embedding vector.

    Validates vector integrity (dimension match, finite values) without
    importing numpy or any ML framework.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    chunk: Chunk
    vector: list[float]
    embedding_model: str
    embedding_dim: int
    normalized: bool

    @field_validator("embedding_model")
    @classmethod
    def model_name_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("embedding_model must not be empty or whitespace-only.")
        return value

    @field_validator("embedding_dim")
    @classmethod
    def dim_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("embedding_dim must be > 0.")
        return value

    @field_validator("vector")
    @classmethod
    def vector_non_empty_and_finite(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("vector must not be empty.")
        for v in value:
            if not math.isfinite(v):
                raise ValueError(f"vector contains a non-finite value: {v}.")
        return value

    @model_validator(mode="after")
    def dim_matches_vector_length(self) -> "EmbeddedChunk":
        if self.embedding_dim != len(self.vector):
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) does not match "
                f"len(vector) ({len(self.vector)})."
            )
        return self

    @staticmethod
    def validate_batch_consistency(
        embedded_chunks: "list[EmbeddedChunk]",
    ) -> "tuple[int, str, bool, list[tuple[EmbeddedChunk, Exception]]]":
        """Validate that all chunks in a batch share dimension, model, and normalisation.

        Args:
            embedded_chunks: The batch to validate. May be empty.

        Returns:
            A tuple of (inferred_dim, inferred_model, inferred_normalized, failures)
            where ``failures`` is a list of (chunk, exception) pairs for every
            chunk whose metadata is inconsistent with the first chunk.
        """
        if not embedded_chunks:
            return 0, "", False, []

        inferred_dim = len(embedded_chunks[0].vector)
        inferred_model = embedded_chunks[0].embedding_model
        inferred_normalized = embedded_chunks[0].normalized
        failures: list[tuple[EmbeddedChunk, Exception]] = []

        for item in embedded_chunks:
            exc: Exception | None = None
            if item.embedding_dim != len(item.vector):
                exc = ValueError(
                    f"embedding_dim={item.embedding_dim} does not match "
                    f"vector length={len(item.vector)}."
                )
            elif len(item.vector) != inferred_dim:
                exc = ValueError(
                    f"inconsistent vector length={len(item.vector)}; expected {inferred_dim}."
                )
            elif item.embedding_model != inferred_model:
                exc = ValueError(
                    f"inconsistent embedding_model={item.embedding_model!r}; "
                    f"expected {inferred_model!r}."
                )
            elif item.normalized != inferred_normalized:
                exc = ValueError(
                    f"inconsistent normalized={item.normalized}; "
                    f"expected {inferred_normalized}."
                )
            if exc is not None:
                failures.append((item, exc))

        return inferred_dim, inferred_model, inferred_normalized, failures
