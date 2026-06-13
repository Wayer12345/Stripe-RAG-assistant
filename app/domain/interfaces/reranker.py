"""Domain interface contract for reranking retrieval candidates."""

from typing import Protocol

from app.domain.models.retrieval_result import RetrievalResult


class Reranker(Protocol):
    """Reranks retrieval candidates for a specific query."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Return reranked top-k retrieval results."""
        ...

