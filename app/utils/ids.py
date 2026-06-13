"""Deterministic ID generation utilities for RAG pipeline artifacts."""

import hashlib
from datetime import UTC, datetime


def make_run_id(stage: str) -> str:
    """Return a compact UTC run ID for a pipeline stage.

    Args:
        stage: Stage name prefix, for example ``"ingestion"``.

    Returns:
        Run identifier like ``ingestion_20260606T152530Z``.

    Raises:
        ValueError: If ``stage`` is empty or whitespace-only.
    """
    if not stage or not stage.strip():
        raise ValueError("stage must not be empty or whitespace-only.")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stage.strip()}_{timestamp}"


def make_document_id(source_identity: str, content_hash: str) -> str:
    """Return a stable deterministic document ID with a ``doc_`` prefix.

    The ID is derived by SHA-256-hashing the combination of normalised
    source identity and content hash, then truncating to 24 hex characters.
    No randomness is involved, so the same inputs always produce the same ID.

    Args:
        source_identity: Stable string identifying the source (e.g. file path
            or URL).  Must not be empty or whitespace-only.
        content_hash: Pre-computed content hash of the document text.
            Must not be empty or whitespace-only.

    Returns:
        A string of the form ``doc_<24-hex-chars>``.

    Raises:
        ValueError: If either argument is empty or whitespace-only.
    """
    if not source_identity or not source_identity.strip():
        raise ValueError("source_identity must not be empty or whitespace-only.")
    if not content_hash or not content_hash.strip():
        raise ValueError("content_hash must not be empty or whitespace-only.")

    combined = f"{source_identity.strip()}:{content_hash.strip()}"
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return f"doc_{digest[:24]}"


def make_chunk_id(
    *,
    document_id: str,
    chunk_index: int,
    start_char: int,
    end_char: int,
    chunk_text_hash: str,
) -> str:
    """Return a stable deterministic chunk ID with a ``chunk_`` prefix."""
    if not document_id or not document_id.strip():
        raise ValueError("document_id must not be empty or whitespace-only.")
    if chunk_index < 0:
        raise ValueError("chunk_index must be >= 0.")
    if start_char < 0 or end_char < start_char:
        raise ValueError("start_char/end_char are invalid.")
    if not chunk_text_hash or not chunk_text_hash.strip():
        raise ValueError("chunk_text_hash must not be empty or whitespace-only.")

    combined = (
        f"{document_id.strip()}:{chunk_index}:{start_char}:{end_char}:"
        f"{chunk_text_hash.strip()}"
    )
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return f"chunk_{digest[:24]}"
