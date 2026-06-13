"""Pipeline stage error record builders and layer status helpers.

These utilities construct structured error payloads for offline stage manifests
and compute canonical layer status strings from run counters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.domain.interfaces.document_loader import RawDocument
    from app.domain.models.chunk import Chunk
    from app.domain.models.document import Document
    from app.domain.models.embedded_chunk import EmbeddedChunk

from app.utils.constants import STATUS_FAILED, STATUS_PARTIAL, STATUS_SUCCESS


def build_raw_document_error_record(
    *,
    raw_document: RawDocument,
    exc: Exception,
    parser_name: str | None = None,
) -> dict[str, Any]:
    """Build a raw-document parsing error payload for manifests."""
    return {
        "source_path": raw_document.source_path,
        "source_name": raw_document.source_name,
        "source_type": raw_document.source_type,
        "mime_type": raw_document.mime_type,
        "parser": parser_name,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def build_document_error_record(
    *,
    document: Document,
    exc: Exception,
) -> dict[str, Any]:
    """Build a document-level stage error payload for manifests."""
    return {
        "document_id": document.id,
        "source_path": document.source_path,
        "source_name": document.source_name,
        "source_type": document.source_type,
        "title": document.title,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def build_chunk_error_record(
    *,
    chunk: Chunk,
    exc: Exception,
) -> dict[str, Any]:
    """Build a chunk-level stage error payload for manifests."""
    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "source_path": metadata.get("source_path"),
        "source_name": metadata.get("source_name"),
        "source_type": metadata.get("source_type"),
        "title": metadata.get("title"),
        "chunk_index": chunk.chunk_index,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def build_embedded_chunk_error_record(
    *,
    embedded_chunk: EmbeddedChunk,
    exc: Exception,
) -> dict[str, Any]:
    """Build an embedded-chunk-level stage error payload for manifests."""
    return {
        "chunk_id": embedded_chunk.chunk.id,
        "document_id": embedded_chunk.chunk.document_id,
        "embedding_model": embedded_chunk.embedding_model,
        "embedding_dim": embedded_chunk.embedding_dim,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def build_stage_error_record(
    *,
    exc: Exception,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a generic stage-level error payload for manifests."""
    record: dict[str, Any] = {}
    if context:
        record.update(context)
    record["error_type"] = type(exc).__name__
    record["error_message"] = str(exc)
    return record


def compute_layer_status(
    *,
    success_count: int,
    error_count: int,
    zero_success_is_failure: bool = True,
) -> str:
    """Return canonical layer status from success and error counters."""
    if error_count > 0 and success_count > 0:
        return STATUS_PARTIAL
    if error_count > 0:
        return STATUS_FAILED
    if success_count == 0 and zero_success_is_failure:
        return STATUS_FAILED
    return STATUS_SUCCESS
