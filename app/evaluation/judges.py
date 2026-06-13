"""Judge adapters for eval runner."""

from __future__ import annotations

import json
import re
from typing import Any

from app.evaluation.generation_metrics import reference_token_f1
from app.evaluation.records import JudgeRecord
from app.utils.logging import get_logger

logger = get_logger(__name__)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _jaccard_similarity(left: str, right: str) -> float:
    left_set = set(_tokenize(left))
    right_set = set(_tokenize(right))
    if not left_set or not right_set:
        return 0.0
    overlap = len(left_set & right_set)
    union = len(left_set | right_set)
    return _safe_divide(float(overlap), float(union))


class HeuristicJudge:
    """Deterministic lexical heuristic judge."""

    backend = "heuristic"

    def judge(
        self,
        *,
        sample_id: str,
        question: str,
        answer: str,
        context_texts: list[str],
        sources: list[Any] | None = None,
        reference_answer: str | None = None,
    ) -> JudgeRecord:
        answer_text = answer.strip()
        lowered_answer = answer_text.lower()
        abstain_like = (not answer_text) or ("don't have enough information" in lowered_answer)
        context_joined = " ".join(part for part in context_texts if part and part.strip())

        if abstain_like:
            groundedness = 1.0
            relevance = 0.0
            source_support = 1.0
            completeness = 1.0 if reference_answer is None else 0.0
            hallucination = 0.0
            verdict = "abstain"
            reason = "No-answer response detected."
        else:
            answer_tokens = _tokenize(answer_text)
            context_tokens = set(_tokenize(context_joined))
            if answer_tokens and context_tokens:
                overlap = sum(1 for token in answer_tokens if token in context_tokens)
                groundedness = _safe_divide(float(overlap), float(len(answer_tokens)))
            else:
                groundedness = 0.0
            relevance = _jaccard_similarity(question, answer_text)

            source_support = 1.0 if (sources and context_texts) else 0.0
            if reference_answer is not None and reference_answer.strip():
                completeness = reference_token_f1(answer_text, reference_answer)
            else:
                completeness = min(1.0, _safe_divide(float(len(answer_tokens)), 40.0))

            combined_support = (groundedness + source_support) / 2.0
            hallucination = 1.0 - combined_support

            if groundedness >= 0.7 and source_support >= 0.5:
                verdict = "pass"
                reason = "Strong grounding and source support."
            elif groundedness >= 0.4:
                verdict = "warn"
                reason = "Partial grounding; answer may be incomplete."
            else:
                verdict = "fail"
                reason = "Low grounding relative to provided context."

        return JudgeRecord(
            sample_id=sample_id,
            groundedness_score=_clamp(groundedness),
            relevance_score=_clamp(relevance),
            source_support_score=_clamp(source_support),
            completeness_score=_clamp(completeness),
            hallucination_risk=_clamp(hallucination),
            verdict=verdict,
            reason=reason,
            judge_backend=self.backend,
            raw_output=None,
            metadata={"context_texts_total": len(context_texts)},
        )


class LocalLLMJudge:
    """Adapter placeholder for local Ollama-based judging."""

    backend = "local_llm"

    def __init__(self, *, settings: Any | None = None) -> None:
        self._settings = settings

    def judge(
        self,
        *,
        sample_id: str,
        question: str,
        answer: str,
        context_texts: list[str],
        sources: list[Any] | None = None,
        reference_answer: str | None = None,
    ) -> JudgeRecord:
        raise NotImplementedError(
            "LocalLLMJudge will be implemented in a later task."
        )

    @staticmethod
    def parse_json_output(sample_id: str, raw_output: str) -> JudgeRecord:
        """Parse judge JSON output into a JudgeRecord."""
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError as err:
            raise ValueError("Judge output is not valid JSON.") from err
        if not isinstance(payload, dict):
            raise ValueError("Judge output must be a JSON object.")
        return JudgeRecord(
            sample_id=sample_id,
            groundedness_score=float(payload.get("groundedness_score", 0.0)),
            relevance_score=float(payload.get("relevance_score", 0.0)),
            source_support_score=float(payload.get("source_support_score", 0.0)),
            completeness_score=float(payload.get("completeness_score", 0.0)),
            hallucination_risk=float(payload.get("hallucination_risk", 1.0)),
            verdict=str(payload.get("verdict", "warn")),
            reason=str(payload.get("reason", "")) or None,
            judge_backend="local_llm",
            raw_output=raw_output,
            metadata=dict(payload.get("metadata", {}))
            if isinstance(payload.get("metadata"), dict)
            else {},
        )


def create_judge(*, backend: str = "heuristic", settings: Any | None = None) -> Any:
    """Create a judge adapter from backend label."""
    normalized = backend.strip().lower()
    if normalized == "none":
        return None
    if normalized == "heuristic":
        return HeuristicJudge()
    if normalized == "local_llm":
        return LocalLLMJudge(settings=settings)
    raise ValueError(f"Unsupported judge backend: {backend!r}")

