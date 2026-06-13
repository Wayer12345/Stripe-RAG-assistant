"""Domain interface contract for dense embeddings."""

from typing import Protocol


class Embedder(Protocol):
    """Embeds documents and queries into plain Python float vectors."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return vectors for document texts."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Return one vector for the user query."""
        ...

    def embedding_dim(self) -> int:
        """Return embedding dimensionality."""
        ...

    def model_name(self) -> str:
        """Return the embedding model identifier."""
        ...

