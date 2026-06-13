"""Deterministic generation output metrics."""

from __future__ import annotations

import re

from app.utils.constants import CONFIDENCE_HIGH, CONFIDENCE_LOW, CONFIDENCE_MEDIUM, CONFIDENCE_NONE

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_VALID_CONFIDENCE = {CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW, CONFIDENCE_NONE}


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def parsed_successfully_rate(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    parsed = sum(1 for flag in flags if flag)
    return _safe_divide(float(parsed), float(len(flags)))


def empty_answer_flag(answer: str) -> float:
    return 1.0 if not answer.strip() else 0.0


def empty_answer_rate(answers: list[str]) -> float:
    if not answers:
        return 0.0
    empty = sum(1 for answer in answers if not answer.strip())
    return _safe_divide(float(empty), float(len(answers)))


def no_answer_flag(confidence: str, answer: str | None = None) -> float:
    if confidence.strip().lower() == CONFIDENCE_NONE:
        return 1.0
    if answer is not None and not answer.strip():
        return 1.0
    return 0.0


def no_answer_rate(confidences: list[str], answers: list[str] | None = None) -> float:
    if not confidences:
        return 0.0
    count = 0
    for index, confidence in enumerate(confidences):
        answer = answers[index] if answers is not None and index < len(answers) else None
        if no_answer_flag(confidence, answer) == 1.0:
            count += 1
    return _safe_divide(float(count), float(len(confidences)))


def answer_length_chars(answer: str) -> int:
    return len(answer.strip())


def answer_length_tokens(answer: str) -> int:
    return len(_tokenize(answer))


def reference_token_f1(prediction: str, reference: str | None) -> float:
    if reference is None or not reference.strip():
        return 0.0
    predicted_tokens = _tokenize(prediction)
    reference_tokens = _tokenize(reference)
    if not predicted_tokens or not reference_tokens:
        return 0.0
    predicted_counts: dict[str, int] = {}
    reference_counts: dict[str, int] = {}
    for token in predicted_tokens:
        predicted_counts[token] = predicted_counts.get(token, 0) + 1
    for token in reference_tokens:
        reference_counts[token] = reference_counts.get(token, 0) + 1
    overlap = sum(
        min(predicted_counts.get(token, 0), reference_counts.get(token, 0))
        for token in reference_counts
    )
    precision = _safe_divide(float(overlap), float(len(predicted_tokens)))
    recall = _safe_divide(float(overlap), float(len(reference_tokens)))
    if precision + recall == 0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def reference_completeness(prediction: str, reference: str | None) -> float:
    if reference is None or not reference.strip():
        return 0.0
    reference_tokens = _tokenize(reference)
    if not reference_tokens:
        return 0.0
    prediction_set = set(_tokenize(prediction))
    hits = sum(1 for token in reference_tokens if token in prediction_set)
    return _safe_divide(float(hits), float(len(reference_tokens)))


def valid_generation_output_flag(*, answer: str, confidence: str, parsed_successfully: bool) -> float:
    confidence_value = confidence.strip().lower()
    if confidence_value not in _VALID_CONFIDENCE:
        return 0.0
    if not parsed_successfully:
        return 0.0
    if confidence_value == CONFIDENCE_NONE:
        return 1.0
    return 0.0 if not answer.strip() else 1.0


def build_generation_metrics(
    *,
    answer: str,
    confidence: str,
    parsed_successfully: bool,
    reference_answer: str | None = None,
) -> dict[str, float]:
    """Build deterministic generation metrics for one case."""
    return {
        "parsed_successfully": 1.0 if parsed_successfully else 0.0,
        "empty_answer": empty_answer_flag(answer),
        "no_answer": no_answer_flag(confidence, answer),
        "answer_length_chars": float(answer_length_chars(answer)),
        "answer_length_tokens": float(answer_length_tokens(answer)),
        "valid_generation_output": valid_generation_output_flag(
            answer=answer,
            confidence=confidence,
            parsed_successfully=parsed_successfully,
        ),
        "reference_f1": reference_token_f1(answer, reference_answer),
        "reference_completeness": reference_completeness(answer, reference_answer),
    }

