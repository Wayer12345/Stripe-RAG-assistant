"""Build chunk artifacts from cleaned documents for the offline stage."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.domain.models.chunk import Chunk
from app.domain.models.document import Document
from app.infrastructure.chunking.chunker_factory import create_chunker
from app.infrastructure.chunking.semantic_chunker import SemanticChunker
from app.infrastructure.storage.artifact_paths import (
    ChunkingArtifactPaths,
    resolve_chunking_artifact_paths,
)
from app.infrastructure.storage.jsonl_store import count_jsonl, read_jsonl, write_jsonl
from app.infrastructure.storage.manifest_store import build_base_manifest, write_manifest
from app.infrastructure.storage.pipeline_records import (
    build_document_error_record,
    compute_layer_status,
)
from app.utils.config import (
    Settings,
    load_settings,
    resolve_config_dir_and_path,
    to_optional_path,
    validate_positive_limit,
)
from app.utils.constants import (
    ARTIFACT_SCHEMA_VERSION,
    STAGE_CHUNKING,
)
from app.utils.logging import get_logger, setup_logging
from app.utils.timing import finish_timed_run, start_timed_run

_STAGE_NAME = STAGE_CHUNKING
logger = get_logger(__name__)


@dataclass(frozen=True)
class BuildChunksResult:
    """Summary of a chunking layer run."""

    run_id: str
    input_path: Path
    output_path: Path
    manifest_path: Path
    cleaned_documents_total: int
    chunks_total: int
    failed_documents_total: int
    skipped_documents_total: int
    duration_ms: int


class BuildChunksLayer:
    """Offline layer that chunks cleaned documents and writes chunk artifacts."""

    @staticmethod
    def _build_chunking_options(settings: Settings) -> dict[str, Any]:
        return {
            "strategy": settings.chunking.strategy,
            "chunk_size": settings.chunking.chunk_size,
            "chunk_size_min": settings.chunking.chunk_size_min,
            "chunk_size_max": settings.chunking.chunk_size_max,
            "chunk_overlap": settings.chunking.chunk_overlap,
            "min_chunk_chars": settings.chunking.min_chunk_chars,
            "max_chunk_chars": settings.chunking.max_chunk_chars,
            "overlap_chars": settings.chunking.overlap_chars,
            "max_overlap_units": settings.chunking.max_overlap_units,
            "use_semantic_boundaries": settings.chunking.use_semantic_boundaries,
            "similarity_threshold": settings.chunking.similarity_threshold,
            "boundary_embedding_model_name": settings.chunking.boundary_embedding_model_name,
            "unit_embed_batch_size": settings.chunking.unit_embed_batch_size,
        }

    @staticmethod
    def _build_chunk_stats(
        chunks: list[Chunk],
        chunks_per_document: dict[str, int],
        documents_with_zero_chunks: int,
    ) -> dict[str, Any]:
        chunk_chars = [len(chunk.text) for chunk in chunks]
        chunk_tokens = [chunk.token_count for chunk in chunks]

        avg_chunk_chars = sum(chunk_chars) / len(chunk_chars) if chunk_chars else 0.0
        avg_chunk_tokens = sum(chunk_tokens) / len(chunk_tokens) if chunk_tokens else 0.0
        return {
            "avg_chunk_chars": avg_chunk_chars,
            "min_chunk_chars": min(chunk_chars) if chunk_chars else 0,
            "max_chunk_chars": max(chunk_chars) if chunk_chars else 0,
            "avg_chunk_tokens": avg_chunk_tokens,
            "min_chunk_tokens": min(chunk_tokens) if chunk_tokens else 0,
            "max_chunk_tokens": max(chunk_tokens) if chunk_tokens else 0,
            "documents_with_zero_chunks": documents_with_zero_chunks,
            "chunks_per_document": chunks_per_document,
        }

    def __init__(
        self,
        *,
        config_path: Path | str = Path("configs/config.yaml"),
        input_path: Path | str | None = None,
        output_path: Path | str | None = None,
        manifest_path: Path | str | None = None,
        fail_fast: bool | None = None,
        limit: int | None = None,
    ) -> None:
        validate_positive_limit(limit)

        self._config_path = Path(config_path)
        self._input_path_override = to_optional_path(input_path)
        self._output_path_override = to_optional_path(output_path)
        self._manifest_path_override = to_optional_path(manifest_path)
        self._fail_fast_override = fail_fast
        self._limit = limit

    def run(self) -> BuildChunksResult:
        """Chunk cleaned documents, write chunks JSONL + manifest, and return a result."""

        config_dir, resolved_config_path = resolve_config_dir_and_path(self._config_path)
        settings = load_settings(config_dir)

        setup_logging(settings)

        artifact_paths: ChunkingArtifactPaths = resolve_chunking_artifact_paths(
            settings,
            input_path_override=self._input_path_override,
            output_path_override=self._output_path_override,
            manifest_path_override=self._manifest_path_override,
        )

        fail_fast = self._fail_fast_override if self._fail_fast_override is not None else False

        timed_run = start_timed_run(_STAGE_NAME)

        logger.info(
            "Starting chunking layer: run_id=%s stage=%s config_path=%s fail_fast=%s limit=%s",
            timed_run.run_id,
            _STAGE_NAME,
            resolved_config_path,
            fail_fast,
            self._limit,
        )
        logger.info(
            "Resolved chunking paths: input_path=%s output_path=%s manifest_path=%s",
            artifact_paths.input_path,
            artifact_paths.output_path,
            artifact_paths.manifest_path,
        )
        logger.info(
            "Chunker config: strategy=%s chunk_size_min=%s chunk_size_max=%s chunk_overlap=%s max_overlap_units=%s semantic_boundaries=%s",
            settings.chunking.strategy,
            settings.chunking.chunk_size_min,
            settings.chunking.chunk_size_max,
            settings.chunking.chunk_overlap,
            settings.chunking.max_overlap_units,
            settings.chunking.use_semantic_boundaries,
        )

        total_available = (
            count_jsonl(artifact_paths.input_path) if self._limit is not None else None
        )
        cleaned_documents = read_jsonl(artifact_paths.input_path, Document, limit=self._limit)

        logger.info("Loaded cleaned documents: count=%s limit=%s", len(cleaned_documents), self._limit)
        if not cleaned_documents:
            logger.warning(
                "Zero cleaned documents loaded: input_path=%s", artifact_paths.input_path
            )
        logger.info(
            "Cleaned documents by source_type: counts=%s",
            dict(Counter(doc.source_type for doc in cleaned_documents)),
        )

        logger.info("Documents to process: count=%s", len(cleaned_documents))

        chunker = create_chunker(settings)

        chunker_name = (
            chunker.CHUNKER_NAME if isinstance(chunker, SemanticChunker) else type(chunker).__name__
        )

        logger.info("Chunker resolved: chunker_name=%s", chunker_name)

        chunks: list[Chunk] = []
        errors: list[dict[str, Any]] = []
        chunks_per_document: dict[str, int] = {}
        documents_with_zero_chunks = 0
        processed_documents_total = 0

        for document in cleaned_documents:
            try:
                document_chunks = chunker.chunk(document)

                chunks.extend(document_chunks)
                chunks_per_document[document.id] = len(document_chunks)

                if not document_chunks:
                    documents_with_zero_chunks += 1
                    if document.text.strip():
                        logger.warning(
                            "Non-empty document produced zero chunks: document_id=%s source_path=%s source_type=%s",
                            document.id,
                            document.source_path,
                            document.source_type,
                        )

            except Exception as exc:
                logger.error(
                    "Chunking failure: document_id=%s source_path=%s source_type=%s title=%s error_type=%s error_message=%s",
                    document.id,
                    document.source_path,
                    document.source_type,
                    document.title,
                    type(exc).__name__,
                    str(exc),
                )

                errors.append(build_document_error_record(document=document, exc=exc))
                chunks_per_document[document.id] = 0
                processed_documents_total += 1

                if fail_fast:
                    break
                continue

            processed_documents_total += 1

        actual_total = total_available if total_available is not None else len(cleaned_documents)
        skipped_documents_total = actual_total - processed_documents_total

        write_jsonl(artifact_paths.output_path, chunks)

        logger.info("Wrote chunks artifact: path=%s", artifact_paths.output_path)

        finished_at, duration_ms = finish_timed_run(timed_run)

        stats = self._build_chunk_stats(
            chunks,
            chunks_per_document=chunks_per_document,
            documents_with_zero_chunks=documents_with_zero_chunks,
        )
        avg_chunks_per_doc = (
            len(chunks) / len(cleaned_documents) if cleaned_documents else 0.0
        )
        oversized_chunks = [
            chunk for chunk in chunks if len(chunk.text) > settings.chunking.max_chunk_chars
        ]

        if oversized_chunks:
            logger.warning(
                "Chunks exceed configured max size: oversized=%s max_chunk_chars=%s",
                len(oversized_chunks),
                settings.chunking.max_chunk_chars,
            )

        small_chunk_threshold = settings.chunking.chunk_size_min
        small_chunks = [chunk for chunk in chunks if len(chunk.text) < small_chunk_threshold]
        small_chunk_rate = (len(small_chunks) / len(chunks)) if chunks else 0.0

        if chunks and small_chunk_rate > 0.3:
            logger.warning(
                "High small-chunk rate detected: small_chunks=%s total_chunks=%s rate=%.3f threshold_chars=%s",
                len(small_chunks),
                len(chunks),
                small_chunk_rate,
                small_chunk_threshold,
            )

        if len(chunks) == 0:
            logger.warning("Zero total chunks produced for chunking stage.")

        top_chunked_documents = sorted(
            chunks_per_document.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]

        logger.info(
            "Top documents by produced chunks: top=%s",
            top_chunked_documents,
        )

        docs_per_second = (
            processed_documents_total / (duration_ms / 1000)
            if duration_ms > 0
            else float(processed_documents_total)
        )

        chunks_per_second = (
            len(chunks) / (duration_ms / 1000) if duration_ms > 0 else float(len(chunks))
        )

        logger.info(
            "Chunking stats: chunks_total=%s documents_with_zero_chunks=%s avg_chunks_per_document=%.3f min_chunk_chars=%s avg_chunk_chars=%.3f max_chunk_chars=%s min_chunk_tokens=%s avg_chunk_tokens=%.3f max_chunk_tokens=%s docs_per_second=%.2f chunks_per_second=%.2f",
            len(chunks),
            documents_with_zero_chunks,
            avg_chunks_per_doc,
            stats["min_chunk_chars"],
            stats["avg_chunk_chars"],
            stats["max_chunk_chars"],
            stats["min_chunk_tokens"],
            stats["avg_chunk_tokens"],
            stats["max_chunk_tokens"],
            docs_per_second,
            chunks_per_second,
        )

        manifest: dict[str, Any] = {
            **build_base_manifest(
                run_id=timed_run.run_id,
                stage=_STAGE_NAME,
                started_at=timed_run.started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
                config_path=resolved_config_path,
                errors=errors,
                schema_version=ARTIFACT_SCHEMA_VERSION,
            ),
            "input_path": str(artifact_paths.input_path),
            "output_path": str(artifact_paths.output_path),
            "cleaned_documents_total": actual_total,
            "chunks_total": len(chunks),
            "failed_documents_total": len(errors),
            "skipped_documents_total": skipped_documents_total,
            "chunker_name": chunker_name,
            "chunking_strategy": settings.chunking.strategy,
            "chunking_options": self._build_chunking_options(settings),
            **stats,
        }

        write_manifest(artifact_paths.manifest_path, manifest)

        logger.info("Wrote chunking manifest: path=%s", artifact_paths.manifest_path)

        status = compute_layer_status(
            success_count=len(chunks),
            error_count=len(errors),
        )

        logger.info(
            "Finished chunking layer: status=%s duration_ms=%s cleaned_documents_total=%s chunks_total=%s failed_documents_total=%s skipped_documents_total=%s",
            status,
            duration_ms,
            actual_total,
            len(chunks),
            len(errors),
            skipped_documents_total,
        )

        return BuildChunksResult(
            run_id=timed_run.run_id,
            input_path=artifact_paths.input_path,
            output_path=artifact_paths.output_path,
            manifest_path=artifact_paths.manifest_path,
            cleaned_documents_total=actual_total,
            chunks_total=len(chunks),
            failed_documents_total=len(errors),
            skipped_documents_total=skipped_documents_total,
            duration_ms=duration_ms,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build chunks from cleaned document artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to config.yaml or a config directory.",
    )
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--fail-fast", action="store_true", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    setup_logging()
    args = _build_arg_parser().parse_args()
    layer = BuildChunksLayer(
        config_path=args.config,
        input_path=args.input_path,
        output_path=args.output_path,
        manifest_path=args.manifest_path,
        fail_fast=args.fail_fast,
        limit=args.limit,
    )
    try:
        result = layer.run()
    except Exception:
        logger.exception("BuildChunksLayer failed")
        sys.exit(1)

    if args.fail_fast and result.failed_documents_total > 0:
        sys.exit(1)
    if result.chunks_total == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
