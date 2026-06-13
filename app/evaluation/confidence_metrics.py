"""Deterministic confidence and abstention metrics."""

from __future__ import annotations

from app.utils.constants import CONFIDENCE_HIGH, CONFIDENCE_LOW, CONFIDENCE_MEDIUM, CONFIDENCE_NONE

_ALLOWED = {CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW, CONFIDENCE_NONE}


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _normalize_confidence(confidence: str) -> str:
    value = confidence.strip().lower()
    if value in _ALLOWED:
        return value
    return CONFIDENCE_LOW


def confidence_distribution(confidences: list[str]) -> dict[str, float]:
    """Return normalized distribution over confidence buckets."""
    if not confidences:
        return {
            CONFIDENCE_HIGH: 0.0,
            CONFIDENCE_MEDIUM: 0.0,
            CONFIDENCE_LOW: 0.0,
            CONFIDENCE_NONE: 0.0,
        }
    normalized = [_normalize_confidence(value) for value in confidences]
    total = float(len(normalized))
    return {
        CONFIDENCE_HIGH: normalized.count(CONFIDENCE_HIGH) / total,
        CONFIDENCE_MEDIUM: normalized.count(CONFIDENCE_MEDIUM) / total,
        CONFIDENCE_LOW: normalized.count(CONFIDENCE_LOW) / total,
        CONFIDENCE_NONE: normalized.count(CONFIDENCE_NONE) / total,
    }


def abstention_rate(confidences: list[str]) -> float:
    normalized = [_normalize_confidence(value) for value in confidences]
    if not normalized:
        return 0.0
    abstained = sum(1 for value in normalized if value == CONFIDENCE_NONE)
    return _safe_divide(float(abstained), float(len(normalized)))


def high_confidence_rate(confidences: list[str]) -> float:
    normalized = [_normalize_confidence(value) for value in confidences]
    if not normalized:
        return 0.0
    highs = sum(1 for value in normalized if value == CONFIDENCE_HIGH)
    return _safe_divide(float(highs), float(len(normalized)))


def high_confidence_without_sources_rate(confidences: list[str], sources_counts: list[int]) -> float:
    total = min(len(confidences), len(sources_counts))
    if total == 0:
        return 0.0
    flagged = 0
    for index in range(total):
        confidence = _normalize_confidence(confidences[index])
        if confidence == CONFIDENCE_HIGH and sources_counts[index] <= 0:
            flagged += 1
    return _safe_divide(float(flagged), float(total))


def high_confidence_empty_context_rate(
    confidences: list[str],
    context_token_counts: list[int],
) -> float:
    total = min(len(confidences), len(context_token_counts))
    if total == 0:
        return 0.0
    flagged = 0
    for index in range(total):
        confidence = _normalize_confidence(confidences[index])
        if confidence == CONFIDENCE_HIGH and context_token_counts[index] <= 0:
            flagged += 1
    return _safe_divide(float(flagged), float(total))


def abstention_on_answerable_rate(confidences: list[str], expected_behaviors: list[str]) -> float:
    pairs = list(zip(confidences, expected_behaviors, strict=False))
    answerable = [pair for pair in pairs if pair[1].strip().lower() == "answer"]
    if not answerable:
        return 0.0
    abstained = sum(
        1
        for confidence, _behavior in answerable
        if _normalize_confidence(confidence) == CONFIDENCE_NONE
    )
    return _safe_divide(float(abstained), float(len(answerable)))


def answer_on_unanswerable_rate(confidences: list[str], expected_behaviors: list[str]) -> float:
    pairs = list(zip(confidences, expected_behaviors, strict=False))
    unanswerable = [pair for pair in pairs if pair[1].strip().lower() == "abstain"]
    if not unanswerable:
        return 0.0
    answered = sum(
        1
        for confidence, _behavior in unanswerable
        if _normalize_confidence(confidence) != CONFIDENCE_NONE
    )
    return _safe_divide(float(answered), float(len(unanswerable)))


def build_confidence_metrics(
    *,
    confidence: str,
    sources_total: int,
    context_token_count: int,
    expected_behavior: str,
) -> dict[str, float]:
    """Build per-case confidence metrics."""
    normalized = _normalize_confidence(confidence)
    expected_behavior_normalized = expected_behavior.strip().lower()
    return {
        "confidence_high": 1.0 if normalized == CONFIDENCE_HIGH else 0.0,
        "confidence_medium": 1.0 if normalized == CONFIDENCE_MEDIUM else 0.0,
        "confidence_low": 1.0 if normalized == CONFIDENCE_LOW else 0.0,
        "confidence_none": 1.0 if normalized == CONFIDENCE_NONE else 0.0,
        "abstained": 1.0 if normalized == CONFIDENCE_NONE else 0.0,
        "high_confidence_without_sources": (
            1.0 if normalized == CONFIDENCE_HIGH and sources_total <= 0 else 0.0
        ),
        "high_confidence_empty_context": (
            1.0 if normalized == CONFIDENCE_HIGH and context_token_count <= 0 else 0.0
        ),
        "abstention_on_answerable": (
            1.0
            if expected_behavior_normalized == "answer" and normalized == CONFIDENCE_NONE
            else 0.0
        ),
        "answer_on_unanswerable": (
            1.0
            if expected_behavior_normalized == "abstain" and normalized != CONFIDENCE_NONE
            else 0.0
        ),
    }
