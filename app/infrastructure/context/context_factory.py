"""Factory for configured context builder implementations."""

from __future__ import annotations

from app.infrastructure.context.context_builder import ContextBuilder
from app.utils.config import Settings


def create_context_builder(settings: Settings) -> ContextBuilder:
    """Create configured context builder core implementation."""
    context = settings.context

    if context.token_budget <= 0:
        raise ValueError("context.token_budget must be > 0")
    if context.max_chunks <= 0:
        raise ValueError("context.max_chunks must be > 0")
    if context.max_sources <= 0:
        raise ValueError("context.max_sources must be > 0")
    if context.min_chunk_tokens <= 0:
        raise ValueError("context.min_chunk_tokens must be > 0")
    if context.max_chunk_tokens <= 0:
        raise ValueError("context.max_chunk_tokens must be > 0")
    if not context.context_format_version.strip():
        raise ValueError("context.context_format_version must not be empty")

    return ContextBuilder(
        token_budget=context.token_budget,
        max_chunks=context.max_chunks,
        max_sources=context.max_sources,
        min_chunk_tokens=context.min_chunk_tokens,
        deduplicate_by=list(context.deduplicate_by),
        include_scores=context.include_scores,
        include_metadata=context.include_metadata,
        context_format_version=context.context_format_version,
        truncate_long_chunks=context.truncate_long_chunks,
        max_chunk_tokens=context.max_chunk_tokens,
    )
