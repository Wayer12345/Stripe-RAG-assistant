"""Factory for configured reranker implementations."""

from __future__ import annotations

from app.infrastructure.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.utils.config import Settings

_DEFAULT_PROVIDER = "cross_encoder"
_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def create_reranker(settings: Settings) -> CrossEncoderReranker:
    """Create configured reranker core implementation."""
    reranking = settings.reranking
    provider = (reranking.provider or _DEFAULT_PROVIDER).strip().lower()
    if provider != _DEFAULT_PROVIDER:
        raise ValueError(f"Unsupported reranker provider: {reranking.provider!r}")

    if reranking.top_k_before <= 0:
        raise ValueError("reranking.top_k_before must be > 0")
    if reranking.top_k_after <= 0:
        raise ValueError("reranking.top_k_after must be > 0")
    if reranking.top_k_after > reranking.top_k_before:
        raise ValueError("reranking.top_k_after must be <= reranking.top_k_before")
    if reranking.batch_size <= 0:
        raise ValueError("reranking.batch_size must be > 0")
    if reranking.max_query_chars <= 0:
        raise ValueError("reranking.max_query_chars must be > 0")
    if reranking.max_pair_chars <= 0:
        raise ValueError("reranking.max_pair_chars must be > 0")
    if reranking.latency_budget_ms <= 0:
        raise ValueError("reranking.latency_budget_ms must be > 0")

    model_name = reranking.model_name.strip() if reranking.model_name.strip() else _DEFAULT_MODEL
    return CrossEncoderReranker(
        model_name=model_name,
        batch_size=reranking.batch_size,
        top_k_before=reranking.top_k_before,
        top_k_after=reranking.top_k_after,
        max_query_chars=reranking.max_query_chars,
        max_pair_chars=reranking.max_pair_chars,
        warmup_enabled=reranking.warmup_enabled,
        cache_enabled=reranking.cache_enabled,
        cache_path=reranking.cache_path,
        latency_budget_ms=reranking.latency_budget_ms,
    )
