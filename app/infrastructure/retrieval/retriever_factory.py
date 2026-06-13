"""Factory for configured retrieval strategy implementations."""

from __future__ import annotations

import atexit
from threading import Lock
from typing import TYPE_CHECKING

from app.domain.interfaces.retriever import Retriever
from app.infrastructure.embeddings.embedder_factory import create_embedder
from app.infrastructure.retrieval.dense_retriever import DenseRetriever
from app.utils.config import Settings

if TYPE_CHECKING:
    from app.infrastructure.vector_stores.qdrant_store import QdrantStore

_RETRIEVER_CACHE_LOCK = Lock()
_RETRIEVER_CACHE: dict[tuple[object, ...], Retriever] = {}


def _retriever_cache_key(settings: Settings) -> tuple[object, ...]:
    return (
        settings.retrieval.strategy,
        settings.retrieval.dense_top_k,
        settings.embeddings.provider,
        settings.embeddings.model_name,
        settings.embeddings.batch_size,
        settings.embeddings.normalize_embeddings,
        settings.embeddings.prefix_mode,
        settings.vector_store.provider,
        settings.vector_store.mode,
        str(settings.vector_store.local_path),
        settings.vector_store.collection_name,
        settings.vector_store.distance,
        settings.vector_store.timeout,
        settings.vector_store.wait,
        settings.vector_store.upsert_batch_size,
        settings.api.warmup.qdrant_healthcheck_enabled,
        settings.api.warmup.retrieval_embed_query_enabled,
        settings.api.warmup.retrieval_embed_query_text,
        settings.api.warmup.retrieval_tiny_search_enabled,
        settings.api.warmup.retrieval_tiny_search_top_k,
    )


def create_retriever(settings: Settings) -> Retriever:
    """Create retriever from settings, validating strategy support."""
    strategy = settings.retrieval.strategy
    if strategy != "dense":
        raise ValueError(f"Unsupported retrieval strategy for this implementation: {strategy}")

    embedder = create_embedder(settings)
    from app.infrastructure.vector_stores.qdrant_store import QdrantStore

    vector_store: QdrantStore = QdrantStore(
        mode=settings.vector_store.mode,
        local_path=settings.vector_store.local_path,
        host=settings.vector_store.host,
        port=settings.vector_store.port,
        url=settings.vector_store.url,
        api_key=settings.vector_store.api_key,
        timeout=settings.vector_store.timeout,
        prefer_grpc=settings.vector_store.prefer_grpc,
        collection_name=settings.vector_store.collection_name,
        distance=settings.vector_store.distance,
        upsert_batch_size=settings.vector_store.upsert_batch_size,
        wait=settings.vector_store.wait,
        payload_indexes=settings.vector_store.payload_indexes,
    )
    warmup = settings.api.warmup
    return DenseRetriever(
        embedder=embedder,
        vector_store=vector_store,
        default_top_k=settings.retrieval.dense_top_k,
        warmup_qdrant_healthcheck_enabled=warmup.qdrant_healthcheck_enabled,
        warmup_embed_query_enabled=warmup.retrieval_embed_query_enabled,
        warmup_embed_query_text=warmup.retrieval_embed_query_text,
        warmup_tiny_search_enabled=warmup.retrieval_tiny_search_enabled,
        warmup_tiny_search_top_k=warmup.retrieval_tiny_search_top_k,
    )


def create_cached_retriever(settings: Settings) -> Retriever:
    """Create or reuse an in-process retriever instance for identical settings."""
    cache_key = _retriever_cache_key(settings)
    with _RETRIEVER_CACHE_LOCK:
        existing = _RETRIEVER_CACHE.get(cache_key)
        if existing is not None:
            return existing
        created = create_retriever(settings)
        _RETRIEVER_CACHE[cache_key] = created
        return created


def shutdown_retriever_cache() -> None:
    """Close cached retrievers and clear in-process cache."""
    with _RETRIEVER_CACHE_LOCK:
        cached_retrievers = list(_RETRIEVER_CACHE.values())
        _RETRIEVER_CACHE.clear()

    for retriever in cached_retrievers:
        vector_store = getattr(retriever, "_vector_store", None)
        close_method = getattr(vector_store, "close", None)
        if callable(close_method):
            close_method()


atexit.register(shutdown_retriever_cache)
