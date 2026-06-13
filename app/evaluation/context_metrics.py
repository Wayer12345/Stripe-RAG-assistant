"""Deterministic metrics for context-building quality."""

from __future__ import annotations

from collections.abc import Sequence

from app.evaluation.retrieval_metrics import recall_at_k, url_recall_at_k


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


def _rate(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    positives = sum(1 for flag in flags if flag)
    return _safe_divide(float(positives), float(len(flags)))


def _mean(values: Sequence[int | float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def context_chunk_recall(context_chunk_ids: list[str], expected_chunk_ids: list[str]) -> float:
    return recall_at_k(context_chunk_ids, expected_chunk_ids, k=max(1, len(context_chunk_ids)))


def context_document_recall(context_document_ids: list[str], expected_document_ids: list[str]) -> float:
    return recall_at_k(
        context_document_ids, expected_document_ids, k=max(1, len(context_document_ids))
    )


def context_url_recall(context_urls: list[str], expected_urls: list[str]) -> float:
    return url_recall_at_k(context_urls, expected_urls, k=max(1, len(context_urls)))


def expected_source_dropped_rate(
    retrieved_or_reranked_ids: list[str],
    context_ids: list[str],
    expected_ids: list[str],
) -> float:
    """Rate of expected ids dropped from context among those previously present."""
    expected = set(_deduplicate_preserve_order(expected_ids))
    if not expected:
        return 0.0
    upstream = set(_deduplicate_preserve_order(retrieved_or_reranked_ids))
    context = set(_deduplicate_preserve_order(context_ids))
    expected_seen_upstream = [value for value in expected if value in upstream]
    if not expected_seen_upstream:
        return 0.0
    dropped = sum(1 for value in expected_seen_upstream if value not in context)
    return _safe_divide(float(dropped), float(len(expected_seen_upstream)))


def token_budget_violation(token_count: int, token_budget: int) -> float:
    if token_budget <= 0:
        raise ValueError("token_budget must be > 0.")
    if token_count < 0:
        raise ValueError("token_count must be >= 0.")
    return 1.0 if token_count > token_budget else 0.0


def truncated_flag_rate(flags: list[bool]) -> float:
    return _rate(flags)


def empty_context_rate(token_counts: list[int]) -> float:
    if any(value < 0 for value in token_counts):
        raise ValueError("token_counts must not contain negative values.")
    return _rate([value == 0 for value in token_counts])


def dedup_rate(input_ids: list[str], output_ids: list[str]) -> float:
    if not input_ids:
        return 0.0
    unique_output_count = len(_deduplicate_preserve_order(output_ids))
    removed = max(0, len(input_ids) - unique_output_count)
    return _safe_divide(float(removed), float(len(input_ids)))


def avg_context_tokens(token_counts: list[int]) -> float:
    if any(value < 0 for value in token_counts):
        raise ValueError("token_counts must not contain negative values.")
    return _mean(token_counts)


def build_context_metrics(
    *,
    context_chunk_ids: list[str],
    context_document_ids: list[str],
    context_urls: list[str],
    expected_chunk_ids: list[str],
    expected_document_ids: list[str],
    expected_urls: list[str],
    reranked_chunk_ids: list[str] | None = None,
    token_count: int | None = None,
    token_budget: int | None = None,
    truncated: bool | None = None,
    input_chunk_ids: list[str] | None = None,
) -> dict[str, float]:
    """Build per-case context metrics."""
    reference_ids = reranked_chunk_ids if reranked_chunk_ids is not None else (input_chunk_ids or [])
    token_count_value = token_count or 0

    metrics: dict[str, float] = {
        "context_chunk_recall": context_chunk_recall(context_chunk_ids, expected_chunk_ids),
        "context_document_recall": context_document_recall(
            context_document_ids, expected_document_ids
        ),
        "context_url_recall": context_url_recall(context_urls, expected_urls),
        "expected_chunk_dropped_rate": expected_source_dropped_rate(
            reference_ids,
            context_chunk_ids,
            expected_chunk_ids,
        ),
        "token_budget_violation": (
            token_budget_violation(token_count_value, token_budget)
            if token_budget is not None
            else 0.0
        ),
        "truncated": 1.0 if truncated else 0.0,
        "empty_context": 1.0 if token_count_value <= 0 else 0.0,
        "dedup_rate": dedup_rate(input_chunk_ids or [], context_chunk_ids),
    }
    return metrics

