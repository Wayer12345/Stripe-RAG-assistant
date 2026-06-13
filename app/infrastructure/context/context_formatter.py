"""Deterministic rendering of retrieval results into LLM context text."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.domain.models.retrieval_result import RetrievalResult

_SAFE_METADATA_FIELDS = (
    "source_type",
    "category",
    "section",
    "heading_path",
    "token_count",
    "char_start",
    "char_end",
)


class ContextFormatter:
    """Formats retrieval chunks into deterministic context source blocks."""

    def format(
        self,
        *,
        query: str,
        results: list[RetrievalResult],
        include_scores: bool,
        include_metadata: bool,
    ) -> str:
        """Return rendered context containing one block per result."""
        _ = query  # Query is part of public API for future formatter versions.
        blocks = [
            self.format_source_block(
                index=index,
                result=result,
                text=result.text,
                include_scores=include_scores,
                include_metadata=include_metadata,
            )
            for index, result in enumerate(results, start=1)
        ]
        return "\n\n".join(block for block in blocks if block.strip())

    def format_source_block(
        self,
        *,
        index: int,
        result: RetrievalResult,
        text: str,
        include_scores: bool,
        include_metadata: bool,
    ) -> str:
        """Render one result as a deterministic `[Source N]` context block."""
        source = result.source
        lines: list[str] = [f"[Source {index}]"]
        lines.append(f"Title: {source.title}")
        if source.url:
            lines.append(f"URL: {source.url}")
        if source.section:
            lines.append(f"Section: {source.section}")
        lines.append(f"Chunk ID: {result.chunk_id}")
        lines.append(f"Document ID: {result.document_id}")

        if include_scores:
            score = self._score_for_display(result)
            if score is not None:
                lines.append(f"Score: {score:.6f}")

        if include_metadata:
            metadata_lines = self._safe_metadata_lines(result)
            lines.extend(metadata_lines)

        lines.append("")
        lines.append("Content:")
        lines.append(text.strip())
        return "\n".join(lines).strip()

    @staticmethod
    def _score_for_display(result: RetrievalResult) -> float | None:
        for candidate in (
            result.final_score,
            result.reranker_score,
            result.retrieval_score,
            result.dense_score,
            result.lexical_score,
        ):
            if isinstance(candidate, (int, float)):
                return float(candidate)
        return None

    @staticmethod
    def _safe_metadata_lines(result: RetrievalResult) -> list[str]:
        metadata: Mapping[str, Any] = result.metadata
        lines: list[str] = []

        safe_pairs: list[tuple[str, Any]] = []
        for key in _SAFE_METADATA_FIELDS:
            if key == "source_type" and result.source.source_type:
                safe_pairs.append((key, result.source.source_type))
                continue
            if key == "section" and result.source.section:
                safe_pairs.append((key, result.source.section))
                continue
            if key == "heading_path" and result.source.heading_path:
                safe_pairs.append((key, result.source.heading_path))
                continue
            if key in metadata:
                safe_pairs.append((key, metadata[key]))

        for key, value in safe_pairs:
            if isinstance(value, list):
                value_str = " > ".join(str(item) for item in value)
            else:
                value_str = str(value)
            lines.append(f"Metadata {key}: {value_str}")

        return lines
