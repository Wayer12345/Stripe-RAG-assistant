"""Dense query-time retriever backed by embedder + vector store."""

from __future__ import annotations

from typing import Any

from app.domain.interfaces.embedder import Embedder
from app.domain.interfaces.retriever import Retriever
from app.domain.interfaces.vector_store import VectorStore
from app.domain.models.retrieval_result import RetrievalMethod, RetrievalResult
from app.utils.logging import get_logger

logger = get_logger(__name__)


class DenseRetriever(Retriever):
    """Retrieve dense candidates by embedding query then searching vector store."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        default_top_k: int = 30,
        warmup_qdrant_healthcheck_enabled: bool = True,
        warmup_embed_query_enabled: bool = True,
        warmup_embed_query_text: str = "warmup query",
        warmup_tiny_search_enabled: bool = False,
        warmup_tiny_search_top_k: int = 1,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be > 0.")
        if not warmup_embed_query_text.strip():
            raise ValueError("warmup_embed_query_text must not be empty.")
        if warmup_tiny_search_top_k <= 0:
            raise ValueError("warmup_tiny_search_top_k must be > 0.")
        self._embedder = embedder
        self._vector_store = vector_store
        self._default_top_k = default_top_k
        self._warmup_qdrant_healthcheck_enabled = warmup_qdrant_healthcheck_enabled
        self._warmup_embed_query_enabled = warmup_embed_query_enabled
        self._warmup_embed_query_text = warmup_embed_query_text.strip()
        self._warmup_tiny_search_enabled = warmup_tiny_search_enabled
        self._warmup_tiny_search_top_k = warmup_tiny_search_top_k

    def collection_exists(self) -> bool:
        """Return whether the backing vector store collection exists."""
        exists_fn = getattr(self._vector_store, "collection_exists", None)
        if callable(exists_fn):
            return bool(exists_fn())
        return True

    def close(self) -> None:
        """Close the backing vector store if it supports lifecycle management."""
        close_fn = getattr(self._vector_store, "close", None)
        if callable(close_fn):
            close_fn()

    def warmup(self) -> dict[str, Any]:
        """Warm retrieval path with optional healthcheck/embed/search probes."""
        qdrant_healthcheck_ok: bool | None = None
        embed_query_warmup_ok: bool | None = None
        tiny_search_warmup_ok: bool | None = None

        if self._warmup_qdrant_healthcheck_enabled:
            healthcheck = getattr(self._vector_store, "healthcheck", None)
            if callable(healthcheck):
                qdrant_healthcheck_ok = bool(healthcheck())

        if self._warmup_embed_query_enabled:
            _ = self._embedder.embed_query(self._warmup_embed_query_text)
            embed_query_warmup_ok = True

        if self._warmup_tiny_search_enabled:
            _ = self.retrieve(
                self._warmup_embed_query_text,
                top_k=self._warmup_tiny_search_top_k,
                filters=None,
            )
            tiny_search_warmup_ok = True

        status = (
            "success"
            if (
                qdrant_healthcheck_ok is not False
                and embed_query_warmup_ok is not False
                and tiny_search_warmup_ok is not False
            )
            else "failed"
        )
        return {
            "status": status,
            "qdrant_healthcheck_ok": qdrant_healthcheck_ok,
            "embed_query_warmup_ok": embed_query_warmup_ok,
            "tiny_search_warmup_ok": tiny_search_warmup_ok,
        }

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be > 0.")

        try:
            query_vector = self._embedder.embed_query(query)
        except Exception:
            logger.exception("Failed to embed query for dense retrieval.")
            raise

        return self.retrieve_by_vector(
            query=query,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )

    def retrieve_by_vector(
        self,
        *,
        query: str,
        query_vector: list[float],
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query must not be empty.")
        effective_top_k = self._default_top_k if top_k is None else top_k
        if effective_top_k <= 0:
            raise ValueError("top_k must be > 0.")
        if not query_vector:
            raise ValueError("query_vector must not be empty.")

        try:
            raw_results = self._vector_store.search(
                query_vector,
                top_k=effective_top_k,
                filters=filters,
            )
        except Exception:
            logger.exception("Qdrant dense search failed.")
            raise

        normalized_results = [
            self._normalize_dense_result(result=result, rank=rank)
            for rank, result in enumerate(raw_results, start=1)
        ]
        normalized_results.sort(
            key=lambda result: (
                result.dense_score if result.dense_score is not None else result.final_score
            ),
            reverse=True,
        )
        return [
            result.model_copy(update={"rank": rank})
            for rank, result in enumerate(normalized_results, start=1)
        ]

    @staticmethod
    def _normalize_dense_result(*, result: RetrievalResult, rank: int) -> RetrievalResult:
        dense_score = result.dense_score
        if dense_score is None:
            if result.retrieval_score is not None:
                dense_score = result.retrieval_score
            else:
                dense_score = result.final_score
        return result.model_copy(
            update={
                "dense_score": dense_score,
                "retrieval_score": dense_score,
                "lexical_score": None,
                "reranker_score": None,
                "final_score": dense_score,
                "retrieval_method": RetrievalMethod.DENSE,
                "rank": rank,
                "source": result.source.model_copy(update={"support_score": dense_score}),
            }
        )
