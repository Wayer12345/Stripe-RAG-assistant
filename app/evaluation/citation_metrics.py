"""Deterministic citation and source-support metrics."""

from __future__ import annotations

from app.utils.constants import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_NONE


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


def valid_citation_rate(cited_ids: list[str], allowed_ids: list[str]) -> float:
    cited = _deduplicate_preserve_order(cited_ids)
    if not cited:
        return 0.0
    allowed = set(_deduplicate_preserve_order(allowed_ids))
    valid = sum(1 for value in cited if value in allowed)
    return _safe_divide(float(valid), float(len(cited)))


def invented_source_rate(cited_ids: list[str], allowed_ids: list[str]) -> float:
    cited = _deduplicate_preserve_order(cited_ids)
    if not cited:
        return 0.0
    allowed = set(_deduplicate_preserve_order(allowed_ids))
    invented = sum(1 for value in cited if value not in allowed)
    return _safe_divide(float(invented), float(len(cited)))


def citation_precision(cited_ids: list[str], expected_ids: list[str]) -> float:
    cited = _deduplicate_preserve_order(cited_ids)
    if not cited:
        return 0.0
    expected = set(_deduplicate_preserve_order(expected_ids))
    if not expected:
        return 0.0
    correct = sum(1 for value in cited if value in expected)
    return _safe_divide(float(correct), float(len(cited)))


def citation_recall(cited_ids: list[str], expected_ids: list[str]) -> float:
    cited = set(_deduplicate_preserve_order(cited_ids))
    expected = _deduplicate_preserve_order(expected_ids)
    if not expected:
        return 0.0
    covered = sum(1 for value in expected if value in cited)
    return _safe_divide(float(covered), float(len(expected)))


def answer_without_sources_flag(answer: str, cited_ids: list[str]) -> float:
    if not answer.strip():
        return 0.0
    cited = _deduplicate_preserve_order(cited_ids)
    return 1.0 if not cited else 0.0


def high_confidence_invalid_source_flag(
    confidence: str,
    cited_ids: list[str],
    allowed_ids: list[str],
) -> float:
    confidence_value = confidence.strip().lower()
    if confidence_value not in {CONFIDENCE_HIGH, CONFIDENCE_MEDIUM}:
        return 0.0
    return 1.0 if invented_source_rate(cited_ids, allowed_ids) > 0.0 else 0.0


def unsupported_citation_rate(cited_ids: list[str], supported_ids: list[str]) -> float:
    return invented_source_rate(cited_ids, supported_ids)


def build_citation_metrics(
    *,
    cited_chunk_ids: list[str],
    context_chunk_ids: list[str],
    expected_chunk_ids: list[str],
    answer: str,
    confidence: str,
) -> dict[str, float]:
    """Build per-case deterministic citation metrics."""
    no_answer = confidence.strip().lower() == CONFIDENCE_NONE or not answer.strip()
    answer_without_sources = (
        0.0 if no_answer else answer_without_sources_flag(answer, cited_chunk_ids)
    )
    return {
        "valid_citation_rate": valid_citation_rate(cited_chunk_ids, context_chunk_ids),
        "invented_source_rate": invented_source_rate(cited_chunk_ids, context_chunk_ids),
        "citation_precision": citation_precision(cited_chunk_ids, expected_chunk_ids),
        "citation_recall": citation_recall(cited_chunk_ids, expected_chunk_ids),
        "answer_without_sources": answer_without_sources,
        "high_confidence_invalid_source": high_confidence_invalid_source_flag(
            confidence, cited_chunk_ids, context_chunk_ids
        ),
        "unsupported_citation_rate": unsupported_citation_rate(
            cited_chunk_ids, context_chunk_ids
        ),
    }
