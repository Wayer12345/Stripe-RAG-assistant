"""Parser for generation raw LLM output -> GeneratedAnswer."""

from __future__ import annotations

import json
import re
from typing import Any

from app.domain.models.answer import Confidence, GeneratedAnswer
from app.domain.models.context import ContextBundle
from app.domain.models.source import Source
from app.utils.constants import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_NONE,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

_JSON_CODE_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_ANSWER_HEADER_PATTERN = re.compile(r"^\s*answer\s*:\s*(.*)$", re.IGNORECASE)
_BEGIN_RESPONSE_MARKER = "BEGIN_RESPONSE"
_END_RESPONSE_MARKER = "END_RESPONSE"
_CLEANUP_PREFIX_PATTERN = re.compile(
    r"^\s*(answer|final answer|the final answer is)\s*:\s*",
    re.IGNORECASE,
)
_CORRECTION_MARKER_PATTERN = re.compile(r"(?i)was incorrect\.\s*here is the correct answer:\s*")


class OutputParser:
    """Extracts answer text and derives confidence/sources deterministically."""

    def __init__(
        self,
        *,
        fallback_answer: str | None = None,
        no_answer_quality_threshold_pct: float = 50.0,
    ) -> None:
        self._fallback_answer = fallback_answer or (
            "I don't have enough information in the indexed Stripe Guides sources to answer this reliably."
        )
        self._no_answer_quality_threshold_pct = no_answer_quality_threshold_pct
        self._last_stats: dict[str, Any] = {}

    def last_stats(self) -> dict[str, Any]:
        """Return parser stats for last parse call."""
        return dict(self._last_stats)

    def parse(
        self,
        *,
        raw_output: str,
        context_bundle: ContextBundle,
    ) -> GeneratedAnswer:
        """Parse answer text only; derive confidence/sources from context."""
        logger.info("Starting output parsing: raw_output_chars=%s", len(raw_output))
        if not isinstance(raw_output, str):
            return self.fallback_no_answer(
                raw_output="",
                reason="raw_output_not_string",
                context_bundle=context_bundle,
            )
        if not isinstance(context_bundle, ContextBundle):
            raise TypeError("context_bundle must be a ContextBundle.")

        answer_raw = self._extract_answer_text(raw_output)
        if answer_raw is None or not answer_raw.strip():
            logger.warning("Raw output does not contain answer text.")
            return self.fallback_no_answer(
                raw_output=raw_output,
                reason="answer_not_found_or_empty",
                context_bundle=context_bundle,
            )

        answer_text = self._normalize_answer_text(answer_raw)
        if not answer_text:
            return self.fallback_no_answer(
                raw_output=raw_output,
                reason="answer_not_found_or_empty",
                context_bundle=context_bundle,
            )
        selected_sources = self._select_sources(context_bundle=context_bundle, limit=3)
        quality_pct = self._estimate_quality_pct(
            context_bundle=context_bundle,
            selected_sources=selected_sources,
        )
        if quality_pct < self._no_answer_quality_threshold_pct:
            return self.fallback_no_answer(
                raw_output=raw_output,
                reason="quality_below_threshold",
                context_bundle=context_bundle,
            )

        if self._contains_fallback_phrase(answer_text):
            sanitized_answer = self._remove_fallback_phrase(answer_text)
            if sanitized_answer:
                answer_text = sanitized_answer
            elif self._is_fallback_only(answer_text):
                best_effort = self._best_effort_answer_from_context(context_bundle)
                if best_effort:
                    answer_text = best_effort
                else:
                    return self.fallback_no_answer(
                        raw_output=raw_output,
                        reason="model_returned_no_answer",
                        context_bundle=context_bundle,
                    )

        confidence = self._derive_confidence(
            context_bundle=context_bundle,
            answer_text=answer_text,
            selected_sources=selected_sources,
        )

        answer = GeneratedAnswer(
            answer=answer_text,
            confidence=Confidence(confidence),
            sources=selected_sources,
            raw_output=raw_output,
            parsed_successfully=True,
            metadata={
                "quality_score_pct": quality_pct,
                "quality_threshold_pct": self._no_answer_quality_threshold_pct,
            },
        )
        self._last_stats = {
            "parsed_successfully": True,
            "warnings_total": 0,
            "sources_total": len(selected_sources),
            "raw_output_chars": len(raw_output),
            "confidence": confidence,
            "quality_score_pct": quality_pct,
            "quality_threshold_pct": self._no_answer_quality_threshold_pct,
        }
        logger.info(
            "Finished output parsing: parsed_successfully=%s confidence=%s sources_total=%s",
            True,
            confidence,
            len(selected_sources),
        )
        return answer

    def fallback_no_answer(
        self,
        *,
        raw_output: str = "",
        reason: str,
        context_bundle: ContextBundle | None = None,
    ) -> GeneratedAnswer:
        """Return deterministic safe fallback when parsing/validation fails."""
        logger.warning("Using no-answer fallback: reason=%s", reason)
        metadata: dict[str, Any] = {"no_answer_reason": reason}
        if context_bundle is not None:
            metadata["context_token_count"] = context_bundle.token_count
            metadata["context_sources_total"] = len(context_bundle.sources)

        answer = GeneratedAnswer(
            answer=self._fallback_answer,
            confidence=Confidence.NONE,
            sources=[],
            raw_output=raw_output,
            parsed_successfully=False,
            metadata=metadata,
        )
        self._last_stats = {
            "parsed_successfully": False,
            "warnings_total": 1,
            "sources_total": 0,
            "raw_output_chars": len(raw_output),
            "confidence": CONFIDENCE_NONE,
        }
        return answer

    def _extract_answer_text(self, raw_output: str) -> str | None:
        block = self._extract_last_marked_response_block(raw_output)
        if block is not None:
            answer_from_block = self._extract_answer_from_key_value(block)
            if answer_from_block:
                return answer_from_block

        json_payload = self._extract_json_payload(raw_output)
        if isinstance(json_payload.get("answer"), str):
            answer_from_json = str(json_payload["answer"]).strip()
            if answer_from_json:
                return answer_from_json

        answer_from_key_value = self._extract_answer_from_key_value(raw_output)
        if answer_from_key_value:
            return answer_from_key_value

        plain_text = raw_output.strip()
        return plain_text if plain_text else None

    @staticmethod
    def _extract_last_marked_response_block(raw_output: str) -> str | None:
        start_positions = [m.start() for m in re.finditer(_BEGIN_RESPONSE_MARKER, raw_output)]
        end_positions = [m.start() for m in re.finditer(_END_RESPONSE_MARKER, raw_output)]
        if not start_positions or not end_positions:
            return None

        best_pair: tuple[int, int] | None = None
        for start_idx in start_positions:
            for end_idx in end_positions:
                if end_idx > start_idx:
                    best_pair = (start_idx, end_idx)
        if best_pair is None:
            return None

        start_idx, end_idx = best_pair
        start_content_idx = start_idx + len(_BEGIN_RESPONSE_MARKER)
        return raw_output[start_content_idx:end_idx].strip()

    @staticmethod
    def _extract_answer_from_key_value(raw_text: str) -> str | None:
        lines = raw_text.splitlines()
        if not lines:
            return None

        collecting_answer = False
        answer_lines: list[str] = []
        for line in lines:
            header_match = _ANSWER_HEADER_PATTERN.match(line)
            if header_match:
                collecting_answer = True
                inline_value = header_match.group(1).strip()
                if inline_value:
                    answer_lines.append(inline_value)
                continue

            if collecting_answer and re.match(r"^\s*(confidence|sources)\s*:", line, re.IGNORECASE):
                break
            if collecting_answer:
                answer_lines.append(line.rstrip())

        if not answer_lines:
            return None
        answer = "\n".join(answer_lines).strip()
        return answer if answer else None

    def _normalize_answer_text(self, answer: str) -> str:
        cleaned = answer.strip()
        cleaned = self._extract_after_answer_marker(cleaned)
        while True:
            new_cleaned = _CLEANUP_PREFIX_PATTERN.sub("", cleaned, count=1).strip()
            if new_cleaned == cleaned:
                break
            cleaned = new_cleaned

        correction_match = _CORRECTION_MARKER_PATTERN.search(cleaned)
        if correction_match:
            cleaned = cleaned[correction_match.end() :].strip()

        for stop_marker in ("\nExplanation:", "\nSources:"):
            if stop_marker in cleaned:
                cleaned = cleaned.split(stop_marker, 1)[0].strip()
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
        if paragraphs:
            cleaned = self._select_best_paragraph(paragraphs)
        cleaned = self._collapse_repeated_sentence_cycles(cleaned)
        cleaned = self._truncate_to_max_sentences(cleaned, max_sentences=4)
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
            cleaned = cleaned[1:-1].strip()
        return cleaned

    def _is_fallback_only(self, answer_text: str) -> bool:
        normalized = self._normalize_text_for_match(answer_text)
        fallback = self._normalize_text_for_match(self._fallback_answer)
        return normalized.lower() == fallback.lower()

    def _contains_fallback_phrase(self, answer_text: str) -> bool:
        normalized_answer = self._normalize_text_for_match(answer_text).lower()
        normalized_fallback = self._normalize_text_for_match(self._fallback_answer).lower()
        return normalized_fallback in normalized_answer

    def _remove_fallback_phrase(self, answer_text: str) -> str:
        fallback_pattern = re.escape(self._fallback_answer.strip())
        cleaned = re.sub(fallback_pattern, "", answer_text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .\n\t")
        return cleaned

    @staticmethod
    def _extract_after_answer_marker(text: str) -> str:
        marker_pattern = re.compile(r"(?i)\banswer\s*:")
        marker_match = marker_pattern.search(text)
        if marker_match:
            text = text[marker_match.end() :].strip()
        second_marker_match = marker_pattern.search(text)
        if second_marker_match:
            text = text[: second_marker_match.start()].strip()
        return text

    @staticmethod
    def _select_best_paragraph(paragraphs: list[str]) -> str:
        if len(paragraphs) == 1:
            return paragraphs[0]

        for idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            if idx == 0 and OutputParser._looks_like_question_only(paragraph):
                continue
            return paragraph
        return paragraphs[0]

    @staticmethod
    def _looks_like_question_only(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized.endswith("?"):
            return False
        if len(normalized) < 15:
            return False
        sentence_parts = [
            part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()
        ]
        return len(sentence_parts) == 1

    @staticmethod
    def _best_effort_answer_from_context(context_bundle: ContextBundle) -> str:
        if not context_bundle.chunks:
            return ""
        top_text = context_bundle.chunks[0].text.strip()
        if not top_text:
            return ""
        first_sentence = re.split(r"(?<=[.!?])\s+", top_text)[0].strip()
        return first_sentence[:260].strip()

    @staticmethod
    def _normalize_text_for_match(value: str) -> str:
        normalized = value.strip().strip('"').strip("'")
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = normalized.rstrip(".").strip()
        return normalized

    @staticmethod
    def _collapse_repeated_sentence_cycles(answer_text: str) -> str:
        normalized = re.sub(r"\s+", " ", answer_text).strip()
        if not normalized:
            return normalized

        sentences = [
            part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()
        ]
        sentence_count = len(sentences)
        if sentence_count < 2:
            return normalized

        max_cycle = min(8, sentence_count // 2)
        for cycle_size in range(1, max_cycle + 1):
            cycle = sentences[:cycle_size]
            matches_cycle = True
            for idx, sentence in enumerate(sentences):
                if sentence != cycle[idx % cycle_size]:
                    matches_cycle = False
                    break
            if matches_cycle and sentence_count >= cycle_size * 2:
                return " ".join(cycle).strip()
        return normalized

    @staticmethod
    def _truncate_to_max_sentences(answer_text: str, *, max_sentences: int) -> str:
        if max_sentences <= 0:
            return answer_text.strip()
        sentences = [
            part.strip() for part in re.split(r"(?<=[.!?])\s+", answer_text.strip()) if part.strip()
        ]
        if not sentences:
            return answer_text.strip()
        return " ".join(sentences[:max_sentences]).strip()

    @staticmethod
    def _extract_json_payload(raw_output: str) -> dict[str, Any]:
        candidate_strings: list[str] = [raw_output.strip()]
        fence_match = _JSON_CODE_FENCE_PATTERN.search(raw_output)
        if fence_match:
            candidate_strings.append(fence_match.group(1).strip())

        first = raw_output.find("{")
        last = raw_output.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidate_strings.append(raw_output[first : last + 1].strip())

        for candidate in candidate_strings:
            try:
                loaded = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, dict):
                return loaded
        return {}

    @staticmethod
    def _select_sources(*, context_bundle: ContextBundle, limit: int) -> list[Source]:
        if limit <= 0:
            return []
        return list(context_bundle.sources[:limit])

    @staticmethod
    def _derive_confidence(
        *,
        context_bundle: ContextBundle,
        answer_text: str,
        selected_sources: list[Source],
    ) -> str:
        if not answer_text.strip() or not selected_sources:
            return CONFIDENCE_NONE

        confidence = CONFIDENCE_LOW
        if len(selected_sources) >= 2:
            confidence = CONFIDENCE_MEDIUM
        if len(selected_sources) >= 3 and not context_bundle.truncated:
            confidence = CONFIDENCE_HIGH

        if context_bundle.truncated and confidence == CONFIDENCE_HIGH:
            confidence = CONFIDENCE_MEDIUM
        if len(answer_text.split()) < 10 and confidence == CONFIDENCE_HIGH:
            confidence = CONFIDENCE_MEDIUM
        return confidence

    @staticmethod
    def _estimate_quality_pct(
        *,
        context_bundle: ContextBundle,
        selected_sources: list[Source],
    ) -> float:
        support_scores = [
            float(source.support_score)
            for source in selected_sources
            if source.support_score is not None
        ]
        if support_scores:
            quality_pct = (sum(support_scores) / len(support_scores)) * 100.0
        else:
            quality_pct = min(100.0, len(selected_sources) * 30.0)

        if context_bundle.truncated:
            quality_pct -= 10.0
        return max(0.0, min(100.0, quality_pct))
