"""Deterministic retrieval metrics for eval records."""

from __future__ import annotations

import math
from urllib.parse import urlsplit, urlunsplit


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be > 0.")


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


def _normalize_url(url: str) -> str:
    normalized = url.strip()
    if not normalized:
        return ""
    split = urlsplit(normalized)
    scheme = split.scheme.lower()
    netloc = split.netloc.lower()
    path = split.path.rstrip("/")
    if not path:
        path = ""
    return urlunsplit((scheme, netloc, path, split.query, split.fragment))


def _normalize_url_list(urls: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in urls:
        normalized_value = _normalize_url(value)
        if normalized_value:
            normalized.append(normalized_value)
    return _deduplicate_preserve_order(normalized)


def hit_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Binary hit metric in top-k."""
    _validate_k(k)
    expected_set = set(_deduplicate_preserve_order(expected_ids))
    if not expected_set:
        return 0.0
    top_k = _deduplicate_preserve_order(retrieved_ids)[:k]
    return 1.0 if any(item in expected_set for item in top_k) else 0.0


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Recall of expected ids in top-k."""
    _validate_k(k)
    expected = _deduplicate_preserve_order(expected_ids)
    if not expected:
        return 0.0
    top_k = _deduplicate_preserve_order(retrieved_ids)[:k]
    hits = sum(1 for value in expected if value in set(top_k))
    return _safe_divide(float(hits), float(len(expected)))


def precision_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Precision over unique top-k retrieved ids."""
    _validate_k(k)
    top_k = _deduplicate_preserve_order(retrieved_ids)[:k]
    if not top_k:
        return 0.0
    expected_set = set(_deduplicate_preserve_order(expected_ids))
    if not expected_set:
        return 0.0
    hits = sum(1 for value in top_k if value in expected_set)
    return _safe_divide(float(hits), float(len(top_k)))


def mrr_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int | None = None) -> float:
    """Reciprocal rank of the first relevant item."""
    if k is not None:
        _validate_k(k)
    expected_set = set(_deduplicate_preserve_order(expected_ids))
    if not expected_set:
        return 0.0
    ranked = _deduplicate_preserve_order(retrieved_ids)
    if k is not None:
        ranked = ranked[:k]
    for index, item in enumerate(ranked, start=1):
        if item in expected_set:
            return 1.0 / float(index)
    return 0.0


def dcg_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Discounted cumulative gain with binary relevance."""
    _validate_k(k)
    expected_set = set(_deduplicate_preserve_order(expected_ids))
    if not expected_set:
        return 0.0
    ranked = _deduplicate_preserve_order(retrieved_ids)[:k]
    score = 0.0
    for index, item in enumerate(ranked, start=1):
        if item in expected_set:
            score += 1.0 / math.log2(index + 1)
    return score


def ndcg_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Normalized DCG at k."""
    _validate_k(k)
    expected = _deduplicate_preserve_order(expected_ids)
    if not expected:
        return 0.0
    actual = dcg_at_k(retrieved_ids, expected, k)
    ideal_count = min(k, len(expected))
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_count + 1))
    return _safe_divide(actual, ideal)


def chunk_recall_at_k(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str], k: int) -> float:
    return recall_at_k(retrieved_chunk_ids, expected_chunk_ids, k)


def document_recall_at_k(
    retrieved_document_ids: list[str], expected_document_ids: list[str], k: int
) -> float:
    return recall_at_k(retrieved_document_ids, expected_document_ids, k)


def url_recall_at_k(retrieved_urls: list[str], expected_urls: list[str], k: int) -> float:
    _validate_k(k)
    normalized_retrieved = _normalize_url_list(retrieved_urls)
    normalized_expected = _normalize_url_list(expected_urls)
    return recall_at_k(normalized_retrieved, normalized_expected, k)


def retrieval_empty_result_rate(results_counts: list[int]) -> float:
    """Rate of zero-result retrieval outcomes."""
    if not results_counts:
        return 0.0
    empty = sum(1 for value in results_counts if value <= 0)
    return _safe_divide(float(empty), float(len(results_counts)))


def build_retrieval_metrics(
    *,
    retrieved_chunk_ids: list[str],
    retrieved_document_ids: list[str],
    retrieved_urls: list[str],
    expected_chunk_ids: list[str],
    expected_document_ids: list[str],
    expected_urls: list[str],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Build per-case deterministic retrieval metrics."""
    ks = [1, 3, 5, 10] if k_values is None else list(k_values)
    for k in ks:
        _validate_k(k)

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"chunk_hit_at_{k}"] = hit_at_k(retrieved_chunk_ids, expected_chunk_ids, k)
        metrics[f"chunk_recall_at_{k}"] = recall_at_k(retrieved_chunk_ids, expected_chunk_ids, k)
        metrics[f"chunk_precision_at_{k}"] = precision_at_k(
            retrieved_chunk_ids, expected_chunk_ids, k
        )
        metrics[f"document_hit_at_{k}"] = hit_at_k(
            retrieved_document_ids, expected_document_ids, k
        )
        metrics[f"document_recall_at_{k}"] = recall_at_k(
            retrieved_document_ids, expected_document_ids, k
        )
        normalized_retrieved_urls = _normalize_url_list(retrieved_urls)
        normalized_expected_urls = _normalize_url_list(expected_urls)
        metrics[f"url_hit_at_{k}"] = hit_at_k(normalized_retrieved_urls, normalized_expected_urls, k)
        metrics[f"url_recall_at_{k}"] = recall_at_k(
            normalized_retrieved_urls,
            normalized_expected_urls,
            k,
        )
        metrics[f"chunk_ndcg_at_{k}"] = ndcg_at_k(retrieved_chunk_ids, expected_chunk_ids, k)

    metrics["chunk_mrr_at_10"] = mrr_at_k(retrieved_chunk_ids, expected_chunk_ids, k=10)
    return metrics
