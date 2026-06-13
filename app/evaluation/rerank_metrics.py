"""Deterministic reranking metrics."""

from __future__ import annotations

from app.evaluation.retrieval_metrics import mrr_at_k, recall_at_k


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _deduplicate_preserve_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = raw.strip()
        if not value:
            continue
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def first_relevant_rank(ordered_ids: list[str], expected_ids: list[str]) -> int | None:
    """Return 1-based rank of first relevant id."""
    expected = set(_deduplicate_preserve_order(expected_ids))
    if not expected:
        return None
    ranked = _deduplicate_preserve_order(ordered_ids)
    for index, item in enumerate(ranked, start=1):
        if item in expected:
            return index
    return None


def rank_delta(before_ids: list[str], after_ids: list[str], expected_ids: list[str]) -> int | None:
    """Improvement in first relevant rank. Positive means better."""
    rank_before = first_relevant_rank(before_ids, expected_ids)
    rank_after = first_relevant_rank(after_ids, expected_ids)
    if rank_before is None and rank_after is None:
        return 0
    if rank_before is None or rank_after is None:
        return 0
    return rank_before - rank_after


def mrr_delta(
    before_ids: list[str],
    after_ids: list[str],
    expected_ids: list[str],
    k: int | None = None,
) -> float:
    """Difference between MRR after and before reranking."""
    before = mrr_at_k(before_ids, expected_ids, k=k)
    after = mrr_at_k(after_ids, expected_ids, k=k)
    return after - before


def expected_source_kept_rate(after_ids: list[str], expected_ids: list[str]) -> float:
    """Share of expected sources that survive in reranked output."""
    expected = _deduplicate_preserve_order(expected_ids)
    if not expected:
        return 0.0
    after_set = set(_deduplicate_preserve_order(after_ids))
    kept = sum(1 for value in expected if value in after_set)
    return _safe_divide(float(kept), float(len(expected)))


def top_k_after_recall(after_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Recall@k after reranking."""
    return recall_at_k(after_ids, expected_ids, k)


def latency_budget_exceeded_rate(flags: list[bool]) -> float:
    """Rate of reranking calls exceeding latency budget."""
    if not flags:
        return 0.0
    exceeded = sum(1 for flag in flags if flag)
    return _safe_divide(float(exceeded), float(len(flags)))


def cache_hit_rate(cache_hits: int, cache_misses: int) -> float:
    """Cache hit ratio over hits+misses."""
    total = cache_hits + cache_misses
    return _safe_divide(float(cache_hits), float(total))


def build_rerank_metrics(
    *,
    retrieved_chunk_ids_before: list[str],
    reranked_chunk_ids_after: list[str],
    expected_chunk_ids: list[str],
    k_values: list[int] | None = None,
    latency_budget_exceeded: bool | None = None,
    cache_hits: int | None = None,
    cache_misses: int | None = None,
) -> dict[str, float]:
    """Build per-case deterministic rerank metrics."""
    ks = [1, 3, 5, 10] if k_values is None else list(k_values)
    rank_before = first_relevant_rank(retrieved_chunk_ids_before, expected_chunk_ids)
    rank_after = first_relevant_rank(reranked_chunk_ids_after, expected_chunk_ids)
    delta = rank_delta(retrieved_chunk_ids_before, reranked_chunk_ids_after, expected_chunk_ids)

    metrics: dict[str, float] = {
        "rank_before": float(rank_before or 0),
        "rank_after": float(rank_after or 0),
        "rank_delta": float(delta or 0),
        "mrr_before": mrr_at_k(retrieved_chunk_ids_before, expected_chunk_ids, k=10),
        "mrr_after": mrr_at_k(reranked_chunk_ids_after, expected_chunk_ids, k=10),
        "mrr_delta": mrr_delta(
            retrieved_chunk_ids_before,
            reranked_chunk_ids_after,
            expected_chunk_ids,
            k=10,
        ),
        "kept_rate": expected_source_kept_rate(reranked_chunk_ids_after, expected_chunk_ids),
    }

    for k in ks:
        metrics[f"recall_after_at_{k}"] = top_k_after_recall(
            reranked_chunk_ids_after, expected_chunk_ids, k
        )

    metrics["latency_budget_exceeded"] = 1.0 if latency_budget_exceeded else 0.0
    metrics["cache_hit_rate"] = (
        cache_hit_rate(cache_hits, cache_misses)
        if cache_hits is not None and cache_misses is not None
        else 0.0
    )
    return metrics
