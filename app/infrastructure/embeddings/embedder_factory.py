"""Factory for constructing configured embedder implementations."""

from __future__ import annotations

from app.domain.interfaces.embedder import Embedder
from app.infrastructure.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from app.utils.config import Settings


def create_embedder(settings: Settings) -> Embedder:
    """Create the configured embedder implementation."""
    provider = settings.embeddings.provider
    cache_enabled = settings.embeddings.cache_enabled
    cache_path = settings.embeddings.cache_path
    if provider != "sentence_transformers":
        raise ValueError(
            "Unsupported embeddings.provider value: "
            f"{provider!r}. Supported values: ['sentence_transformers']."
        )
    if cache_enabled and not str(cache_path).strip():
        raise ValueError("embeddings.cache_path must not be empty when cache is enabled.")

    return SentenceTransformerEmbedder(
        model_name=settings.embeddings.model_name,
        batch_size=settings.embeddings.batch_size,
        normalize_embeddings=settings.embeddings.normalize_embeddings,
        prefix_mode=settings.embeddings.prefix_mode,
    )
