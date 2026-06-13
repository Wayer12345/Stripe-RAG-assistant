"""Domain interface contract for vector-store operations."""

from typing import Any, Protocol

from app.domain.models.embedded_chunk import EmbeddedChunk
from app.domain.models.retrieval_result import RetrievalResult


class VectorStore(Protocol):
    """Vector index contract independent of any concrete backend."""

    def create_collection(self, *, recreate: bool = False) -> None:
        """Create the target collection, optionally recreating it."""
        ...

    def collection_exists(self) -> bool:
        """Return whether the target collection exists."""
        ...

    def validate_collection(self, embedding_dim: int) -> None:
        """Validate collection configuration against embedding dimensionality."""
        ...

    def upsert(self, chunks: list[EmbeddedChunk]) -> int:
        """Upsert embedded chunks and return successful upsert count."""
        ...

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search nearest neighbors and return typed retrieval results."""
        ...

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete indexed points by document identifier and return deleted count."""
        ...

    def count(self) -> int:
        """Return total indexed vector count."""
        ...

    def healthcheck(self) -> bool:
        """Return backend health status."""
        ...

