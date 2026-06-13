"""Domain interface contract for query-time retrieval."""

from typing import Any, Protocol

from app.domain.models.retrieval_result import RetrievalResult


class Retriever(Protocol):
    """Retrieves query candidates using dense, lexical, hybrid, or future strategies."""

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Return retrieval results for the query."""
        ...

