"""Factory for constructing configured chunker implementations."""

from app.domain.interfaces.chunker import Chunker
from app.infrastructure.chunking.semantic_chunker import (
    SemanticChunker,
    SemanticChunkingOptions,
)
from app.utils.config import Settings

_SUPPORTED_STRATEGIES = {"semantic", "structure_aware"}


def create_chunker(settings: Settings) -> Chunker:
    """Build and return the configured chunker implementation."""
    strategy = settings.chunking.strategy
    if strategy not in _SUPPORTED_STRATEGIES:
        supported = ", ".join(sorted(_SUPPORTED_STRATEGIES))
        raise ValueError(
            f"Unsupported chunking.strategy={strategy!r}. Supported values: {supported}."
        )

    options = SemanticChunkingOptions(
        strategy=strategy,
        chunk_size_min=settings.chunking.chunk_size_min,
        chunk_size_max=settings.chunking.chunk_size_max,
        chunk_overlap=settings.chunking.chunk_overlap,
        min_chunk_chars=settings.chunking.min_chunk_chars,
        max_chunk_chars=settings.chunking.max_chunk_chars,
        overlap_chars=settings.chunking.overlap_chars,
        max_overlap_units=settings.chunking.max_overlap_units,
        use_semantic_boundaries=settings.chunking.use_semantic_boundaries,
        similarity_threshold=settings.chunking.similarity_threshold,
        boundary_embedding_model_name=settings.chunking.boundary_embedding_model_name,
        unit_embed_batch_size=settings.chunking.unit_embed_batch_size,
    )
    return SemanticChunker(options)
