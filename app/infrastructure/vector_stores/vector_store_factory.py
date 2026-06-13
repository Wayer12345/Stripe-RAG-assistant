"""Factory for configured local vector store implementations."""

from __future__ import annotations

from typing import Any

from app.domain.interfaces.vector_store import VectorStore
from app.infrastructure.vector_stores.qdrant_store import QdrantStore

_SUPPORTED_MODES = {"embedded"}


def create_vector_store(
    settings: Any,
    *,
    collection_name: str,
    distance: str,
    upsert_batch_size: int,
    payload_indexes: dict[str, str],
) -> VectorStore:
    """Create and return a configured local Qdrant vector store.

    Args:
        settings: Project settings loaded via load_settings.
        collection_name: Qdrant collection to use.
        distance: Distance metric for the collection.
        upsert_batch_size: Number of points per upsert batch.
        payload_indexes: Payload field → index type mapping.

    Returns:
        A configured VectorStore backed by local Qdrant.

    Raises:
        ValueError: If the configured vector store mode is not supported.
    """
    mode = settings.vector_store.mode
    if mode not in _SUPPORTED_MODES:
        allowed = ", ".join(sorted(_SUPPORTED_MODES))
        raise ValueError(
            f"Unsupported vector store mode {mode!r}. "
            f"Allowed: {allowed}."
        )

    return QdrantStore(
        mode=mode,
        local_path=settings.vector_store.local_path,
        host=settings.vector_store.host,
        port=settings.vector_store.port,
        url=settings.vector_store.url,
        api_key=settings.vector_store.api_key,
        timeout=settings.vector_store.timeout,
        prefer_grpc=settings.vector_store.prefer_grpc,
        collection_name=collection_name,
        distance=distance,
        upsert_batch_size=upsert_batch_size,
        wait=settings.vector_store.wait,
        payload_indexes=payload_indexes,
    )
