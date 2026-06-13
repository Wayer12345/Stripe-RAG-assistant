"""Load stage trace JSON artifacts into domain models for online layer chaining."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from app.domain.models.context import ContextBundle
from app.domain.models.retrieval_result import RetrievalResult
from app.domain.models.source import Source


def load_candidates_from_trace(
    input_path: Path | str,
    *,
    empty_text_placeholder: str = "[empty chunk text]",
) -> list[RetrievalResult]:
    """Load retrieval or rerank candidates from a stage trace JSON file.

    Suitable for chaining retrieve → rerank → context layers by reading the
    serialised ``results`` list written by a previous stage.

    Args:
        input_path: Path to a retrieve or rerank trace JSON file.
        empty_text_placeholder: Replacement when the trace contains no chunk text.

    Returns:
        List of RetrievalResult domain models in trace order.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If the file format is invalid or required fields are missing.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"input-path does not exist: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(f"Invalid trace JSON format: {path}") from err

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("Invalid trace format: missing 'results' list.")

    candidates: list[RetrievalResult] = []
    for index, item in enumerate(raw_results, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid trace result at index {index}: expected object.")

        chunk_id = str(item.get("chunk_id", "")).strip()
        document_id = str(item.get("document_id", "")).strip()
        title = str(item.get("title", "")).strip()
        if not chunk_id or not document_id or not title:
            raise ValueError(
                "Invalid trace result entry: chunk_id, document_id, and title are required."
            )

        text = str(item.get("text", "") or item.get("text_preview", "")).strip()
        if not text:
            text = empty_text_placeholder

        raw_metadata = item.get("metadata")
        metadata: dict[str, Any] = (
            cast(dict[str, Any], raw_metadata) if isinstance(raw_metadata, dict) else {}
        )
        source_type = item.get("source_type")
        if source_type is not None and "source_type" not in metadata:
            metadata["source_type"] = source_type
        if isinstance(item.get("token_count"), int):
            metadata.setdefault("token_count", item["token_count"])

        retrieval_score_raw = item.get("retrieval_score")
        dense_score_raw = item.get("dense_score")
        lexical_score_raw = item.get("lexical_score")
        reranker_score_raw = item.get("reranker_score")
        final_score_raw = item.get("final_score")

        retrieval_score = (
            float(retrieval_score_raw)
            if isinstance(retrieval_score_raw, (int, float))
            else (float(dense_score_raw) if isinstance(dense_score_raw, (int, float)) else None)
        )
        dense_score = float(dense_score_raw) if isinstance(dense_score_raw, (int, float)) else None
        lexical_score = (
            float(lexical_score_raw) if isinstance(lexical_score_raw, (int, float)) else None
        )
        reranker_score = (
            float(reranker_score_raw) if isinstance(reranker_score_raw, (int, float)) else None
        )
        final_score = (
            float(final_score_raw)
            if isinstance(final_score_raw, (int, float))
            else (
                reranker_score
                if reranker_score is not None
                else (
                    retrieval_score
                    if retrieval_score is not None
                    else (dense_score if dense_score is not None else 0.0)
                )
            )
        )

        support_score = (
            retrieval_score
            if retrieval_score is not None and 0.0 <= retrieval_score <= 1.0
            else None
        )
        source = Source(
            title=title,
            url=item.get("url"),
            section=item.get("section"),
            chunk_id=chunk_id,
            document_id=document_id,
            support_score=support_score,
            source_type=source_type if isinstance(source_type, str) else None,
            source_name=item.get("source_name"),
            source_path=item.get("source_path"),
            heading_path=(
                [str(part) for part in item["heading_path"]]
                if isinstance(item.get("heading_path"), list)
                else []
            ),
        )
        candidates.append(
            RetrievalResult(
                chunk_id=chunk_id,
                document_id=document_id,
                text=text,
                source=source,
                retrieval_score=retrieval_score,
                lexical_score=lexical_score,
                dense_score=dense_score,
                reranker_score=reranker_score,
                final_score=final_score,
                rank=int(item["rank"]) if isinstance(item.get("rank"), int) else None,
                metadata=metadata,
            )
        )
    return candidates


def load_context_bundle_from_input_path(input_path: Path | str) -> ContextBundle:
    """Load a serialised ContextBundle from a JSON context trace file.

    Supports both direct ContextBundle payloads and context-stage trace files
    that wrap the bundle under a ``context_bundle`` key.

    Args:
        input_path: Path to a context stage trace JSON file.

    Returns:
        A validated ContextBundle domain model.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If the file is not valid JSON, not an object, or does not
            contain sufficient ContextBundle data for generation.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"input-path does not exist: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(f"Invalid JSON input format: {path}") from err
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")

    if isinstance(payload.get("context_bundle"), dict):
        return ContextBundle.model_validate(payload["context_bundle"])

    required_keys = {"query", "chunks", "rendered_context", "token_count"}
    if required_keys.issubset(payload):
        return ContextBundle.model_validate(payload)

    raise ValueError(
        "Input file does not contain a full serialised ContextBundle. "
        "Generation requires full context data (query/chunks/rendered_context/token_count). "
        "Build-context trace files without chunk payloads are insufficient."
    )
