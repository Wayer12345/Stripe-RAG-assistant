"""Deterministic metrics for robustness subsets."""

from __future__ import annotations

from app.utils.constants import CONFIDENCE_LOW, CONFIDENCE_NONE


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _normalize_type(sample_type: str) -> str:
    return sample_type.strip().lower()


def _normalize_confidence(confidence: str) -> str:
    return confidence.strip().lower()


def ood_abstention_rate(confidences: list[str], types: list[str]) -> float:
    pairs = list(zip(confidences, types, strict=False))
    targets = [pair for pair in pairs if _normalize_type(pair[1]) == "ood"]
    if not targets:
        return 0.0
    abstained = sum(1 for confidence, _ in targets if _normalize_confidence(confidence) == CONFIDENCE_NONE)
    return _safe_divide(float(abstained), float(len(targets)))


def unanswerable_unsupported_answer_rate(
    confidences: list[str],
    types: list[str],
    valid_sources_flags: list[bool] | None = None,
) -> float:
    pairs = list(zip(confidences, types, strict=False))
    target_indices = [idx for idx, (_c, t) in enumerate(pairs) if _normalize_type(t) == "unanswerable"]
    if not target_indices:
        return 0.0

    bad_answers = 0
    for index in target_indices:
        confidence = _normalize_confidence(confidences[index])
        answered = confidence != CONFIDENCE_NONE
        unsupported = (
            valid_sources_flags is None
            or index >= len(valid_sources_flags)
            or not valid_sources_flags[index]
        )
        if answered and unsupported:
            bad_answers += 1
    return _safe_divide(float(bad_answers), float(len(target_indices)))


def typo_answer_rate(confidences: list[str], types: list[str]) -> float:
    pairs = list(zip(confidences, types, strict=False))
    targets = [pair for pair in pairs if _normalize_type(pair[1]) == "typo"]
    if not targets:
        return 0.0
    answered = sum(1 for confidence, _ in targets if _normalize_confidence(confidence) != CONFIDENCE_NONE)
    return _safe_divide(float(answered), float(len(targets)))


def ambiguous_abstention_or_low_confidence_rate(confidences: list[str], types: list[str]) -> float:
    pairs = list(zip(confidences, types, strict=False))
    targets = [pair for pair in pairs if _normalize_type(pair[1]) == "ambiguous"]
    if not targets:
        return 0.0
    safe = sum(
        1
        for confidence, _ in targets
        if _normalize_confidence(confidence) in {CONFIDENCE_LOW, CONFIDENCE_NONE}
    )
    return _safe_divide(float(safe), float(len(targets)))


def adversarial_valid_source_rate(types: list[str], valid_source_flags: list[bool]) -> float:
    pairs = list(zip(types, valid_source_flags, strict=False))
    targets = [pair for pair in pairs if _normalize_type(pair[0]) == "adversarial"]
    if not targets:
        return 0.0
    valid = sum(1 for _type, flag in targets if flag)
    return _safe_divide(float(valid), float(len(targets)))


def build_robustness_metrics(
    *,
    sample_type: str,
    confidence: str,
    valid_sources: bool,
) -> dict[str, float]:
    """Build per-case robustness metrics."""
    sample_type_normalized = _normalize_type(sample_type)
    confidence_normalized = _normalize_confidence(confidence)
    return {
        "ood_abstained": (
            1.0
            if sample_type_normalized == "ood" and confidence_normalized == CONFIDENCE_NONE
            else 0.0
        ),
        "unanswerable_unsupported_answer": (
            1.0
            if sample_type_normalized == "unanswerable"
            and confidence_normalized != CONFIDENCE_NONE
            and not valid_sources
            else 0.0
        ),
        "typo_answered": (
            1.0
            if sample_type_normalized == "typo" and confidence_normalized != CONFIDENCE_NONE
            else 0.0
        ),
        "ambiguous_safe": (
            1.0
            if sample_type_normalized == "ambiguous"
            and confidence_normalized in {CONFIDENCE_LOW, CONFIDENCE_NONE}
            else 0.0
        ),
        "adversarial_valid_sources": (
            1.0 if sample_type_normalized == "adversarial" and valid_sources else 0.0
        ),
    }
