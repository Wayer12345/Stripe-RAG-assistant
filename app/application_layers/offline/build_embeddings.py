"""Build embedded chunk artifacts from chunk artifacts for the offline stage."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from app.domain.interfaces.embedder import Embedder
from app.domain.models.chunk import Chunk
from app.domain.models.embedded_chunk import EmbeddedChunk
from app.infrastructure.embeddings.embedder_factory import create_embedder
from app.infrastructure.embeddings.embedding_cache import (
    EmbeddingCache,
    build_embedding_cache_key,
)
from app.infrastructure.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from app.infrastructure.storage.artifact_paths import (
    EmbeddingArtifactPaths,
    resolve_embedding_artifact_paths,
)
from app.infrastructure.storage.jsonl_store import count_jsonl, read_jsonl, write_jsonl
from app.infrastructure.storage.manifest_store import build_base_manifest, write_manifest
from app.infrastructure.storage.pipeline_records import (
    build_chunk_error_record,
    build_stage_error_record,
    compute_layer_status,
)
from app.utils.config import (
    load_settings,
    resolve_config_dir_and_path,
    to_optional_path,
    validate_positive_limit,
)
from app.utils.constants import (
    ARTIFACT_SCHEMA_VERSION,
    STAGE_EMBEDDINGS,
)
from app.utils.logging import get_logger, setup_logging
from app.utils.timing import finish_timed_run, start_timed_run

_STAGE_NAME = STAGE_EMBEDDINGS
_ALLOWED_PREFIX_MODES = {"none", "bge", "e5"}
_QUERY_PREFIX_BY_MODE: dict[str, str] = {
    "none": "",
    "bge": "Represent this sentence for searching relevant passages: ",
    "e5": "query: ",
}
_DOCUMENT_PREFIX_BY_MODE: dict[str, str] = {
    "none": "",
    "bge": "",
    "e5": "passage: ",
}
logger = get_logger(__name__)


@dataclass(frozen=True)
class BuildEmbeddingsResult:
    """Summary of an embeddings layer run."""

    run_id: str
    input_path: Path
    output_path: Path
    manifest_path: Path
    cache_path: Path
    chunks_total: int
    embedded_chunks_total: int
    failed_chunks_total: int
    skipped_chunks_total: int
    embedding_model: str
    embedding_dim: int
    duration_ms: int


class BuildEmbeddingsLayer:
    """Offline layer that embeds chunks and writes embedding artifacts."""

    @staticmethod
    def _normalize_prefix_mode(prefix_mode: str) -> str:
        normalized = prefix_mode.strip().lower()
        if normalized not in _ALLOWED_PREFIX_MODES:
            allowed = ", ".join(sorted(_ALLOWED_PREFIX_MODES))
            raise ValueError(f"Unsupported prefix mode {prefix_mode!r}. Allowed: {allowed}.")
        return normalized

    @staticmethod
    def _document_prefix(prefix_mode: str) -> str:
        return _DOCUMENT_PREFIX_BY_MODE[prefix_mode]

    @staticmethod
    def _query_prefix(prefix_mode: str) -> str:
        return _QUERY_PREFIX_BY_MODE[prefix_mode]

    @staticmethod
    def _embed_documents(embedder: Embedder, texts: list[str]) -> list[list[float]]:
        embed_texts = getattr(embedder, "embed_texts", None)
        if callable(embed_texts):
            return cast(list[list[float]], embed_texts(texts, input_type="document"))
        return embedder.embed_documents(texts)

    @staticmethod
    def _build_vector_stats(embedded_chunks: list[EmbeddedChunk]) -> dict[str, Any]:
        if not embedded_chunks:
            return {
                "vector_count": 0,
                "vector_dim": 0,
                "min_vector_norm": 0.0,
                "max_vector_norm": 0.0,
                "avg_vector_norm": 0.0,
                "zero_vector_count": 0,
            }

        norms: list[float] = []
        zero_vector_count = 0
        for embedded in embedded_chunks:
            norm = math.sqrt(sum(value * value for value in embedded.vector))
            norms.append(norm)
            if norm == 0.0:
                zero_vector_count += 1
        return {
            "vector_count": len(embedded_chunks),
            "vector_dim": embedded_chunks[0].embedding_dim,
            "min_vector_norm": min(norms),
            "max_vector_norm": max(norms),
            "avg_vector_norm": sum(norms) / len(norms),
            "zero_vector_count": zero_vector_count,
        }

    def __init__(
        self,
        *,
        config_path: Path | str = Path("configs/config.yaml"),
        input_path: Path | str | None = None,
        output_path: Path | str | None = None,
        manifest_path: Path | str | None = None,
        cache_path: Path | str | None = None,
        fail_fast: bool | None = None,
        limit: int | None = None,
        embedder: Embedder | None = None,
        cache_enabled: bool | None = None,
    ) -> None:
        validate_positive_limit(limit)

        self._config_path = Path(config_path)
        self._input_path_override = to_optional_path(input_path)
        self._output_path_override = to_optional_path(output_path)
        self._manifest_path_override = to_optional_path(manifest_path)
        self._cache_path_override = to_optional_path(cache_path)
        self._fail_fast_override = fail_fast
        self._limit = limit
        self._embedder = embedder
        self._cache_enabled_override = cache_enabled

    def run(self) -> BuildEmbeddingsResult:
        """Embed chunks, write embedded chunks JSONL + manifest, and return a result."""

        config_dir, resolved_config_path = resolve_config_dir_and_path(self._config_path)
        settings = load_settings(config_dir)

        setup_logging(settings)

        artifact_paths: EmbeddingArtifactPaths = resolve_embedding_artifact_paths(
            settings,
            input_path_override=self._input_path_override,
            output_path_override=self._output_path_override,
            manifest_path_override=self._manifest_path_override,
            cache_path_override=self._cache_path_override,
        )

        fail_fast = self._fail_fast_override if self._fail_fast_override is not None else False

        cache_enabled = (
            self._cache_enabled_override
            if self._cache_enabled_override is not None
            else settings.embeddings.cache_enabled
        )

        prefix_mode = self._normalize_prefix_mode(settings.embeddings.prefix_mode)
        normalized = settings.embeddings.normalize_embeddings
        batch_size = settings.embeddings.batch_size

        timed_run = start_timed_run(_STAGE_NAME)

        logger.info(
            "Starting embeddings layer: run_id=%s stage=%s config_path=%s fail_fast=%s limit=%s",
            timed_run.run_id,
            _STAGE_NAME,
            resolved_config_path,
            fail_fast,
            self._limit,
        )
        logger.info(
            "Resolved embedding paths: input_path=%s output_path=%s manifest_path=%s cache_path=%s",
            artifact_paths.input_path,
            artifact_paths.output_path,
            artifact_paths.manifest_path,
            artifact_paths.cache_path,
        )
        logger.info(
            "Embedding config: provider=%s model=%s batch_size=%s normalized=%s prefix_mode=%s cache_enabled=%s",
            settings.embeddings.provider,
            settings.embeddings.model_name,
            batch_size,
            normalized,
            prefix_mode,
            cache_enabled,
        )

        if not cache_enabled:
            logger.warning("Embedding cache is disabled for this run.")
        elif not artifact_paths.cache_path.exists():
            logger.warning(
                "Embedding cache path does not exist and may be created: cache_path=%s",
                artifact_paths.cache_path,
            )

        total_available = (
            count_jsonl(artifact_paths.input_path) if self._limit is not None else None
        )
        chunks_to_process = read_jsonl(artifact_paths.input_path, Chunk, limit=self._limit)
        chunks_total = total_available if total_available is not None else len(chunks_to_process)
        skipped_chunks_total = chunks_total - len(chunks_to_process)

        logger.info("Loaded chunks: count=%s limit=%s", len(chunks_to_process), self._limit)

        if not chunks_to_process:
            logger.warning("Zero chunks loaded: input_path=%s", artifact_paths.input_path)
        logger.info(
            "Chunks by source_type: counts=%s",
            dict(
                Counter(
                    (
                        chunk.metadata.get("source_type")
                        if isinstance(chunk.metadata, dict)
                        else "unknown"
                    )
                    or "unknown"
                    for chunk in chunks_to_process
                )
            ),
        )

        logger.info("Chunks to process: count=%s", len(chunks_to_process))

        embedder = self._embedder or create_embedder(settings)
        embedding_model = embedder.model_name()

        logger.info("Embedder resolved: embedding_model=%s", embedding_model)

        cache = EmbeddingCache(artifact_paths.cache_path) if cache_enabled else None

        query_prefix = (
            embedder.query_prefix_for_mode(prefix_mode)
            if isinstance(embedder, SentenceTransformerEmbedder)
            else self._query_prefix(prefix_mode)
        )

        document_prefix = (
            embedder.document_prefix_for_mode(prefix_mode)
            if isinstance(embedder, SentenceTransformerEmbedder)
            else self._document_prefix(prefix_mode)
        )

        embedded_by_index: dict[int, EmbeddedChunk] = {}
        missing_items: list[tuple[int, Chunk, str]] = []
        errors: list[dict[str, Any]] = []
        cache_hits = 0
        cache_misses = 0
        cache_writes = 0
        single_item_fallback_attempts = 0
        fail_fast_error: Exception | None = None

        for index, chunk in enumerate(chunks_to_process):
            cache_key = build_embedding_cache_key(
                text=chunk.text,
                model_name=embedding_model,
                normalize_embeddings=normalized,
                prefix_mode=prefix_mode,
                input_type="document",
                prefix=document_prefix,
            )

            if cache is not None:
                vector = cache.get(cache_key)
                if vector is not None:
                    cache_hits += 1
                    embedded_by_index[index] = EmbeddedChunk(
                        chunk=chunk,
                        vector=vector,
                        embedding_model=embedding_model,
                        embedding_dim=len(vector),
                        normalized=normalized,
                    )
                    continue

            missing_items.append((index, chunk, cache_key))

        cache_misses = len(missing_items) if cache_enabled else 0

        if missing_items:
            batch_chunks = [item[1] for item in missing_items]
            batch_texts = [chunk.text for chunk in batch_chunks]

            try:
                vectors = self._embed_documents(embedder, batch_texts)

                if len(vectors) != len(batch_chunks):
                    raise ValueError(
                        "Embedder returned vector count different from input chunk count."
                    )

                for (index, chunk, cache_key), vector in zip(missing_items, vectors, strict=True):
                    embedded_by_index[index] = EmbeddedChunk(
                        chunk=chunk,
                        vector=[float(value) for value in vector],
                        embedding_model=embedding_model,
                        embedding_dim=len(vector),
                        normalized=normalized,
                    )

                    if cache is not None:
                        cache.set(
                            cache_key,
                            embedded_by_index[index].vector,
                            metadata={
                                "embedding_model": embedding_model,
                                "embedding_dim": embedded_by_index[index].embedding_dim,
                                "normalized": normalized,
                                "prefix_mode": prefix_mode,
                                "input_type": "document",
                                "prefix": document_prefix,
                            },
                        )
                        cache_writes += 1

            except Exception as exc:
                if fail_fast:
                    if missing_items:
                        errors.append(build_chunk_error_record(chunk=missing_items[0][1], exc=exc))
                        failed_chunk = missing_items[0][1]
                        logger.error(
                            "Embedding failure: chunk_id=%s document_id=%s chunk_index=%s error_type=%s error_message=%s",
                            failed_chunk.id,
                            failed_chunk.document_id,
                            failed_chunk.chunk_index,
                            type(exc).__name__,
                            str(exc),
                        )
                    fail_fast_error = exc
                else:
                    for index, chunk, cache_key in missing_items:
                        single_item_fallback_attempts += 1

                        try:
                            vector = self._embed_documents(embedder, [chunk.text])[0]
                            embedded_by_index[index] = EmbeddedChunk(
                                chunk=chunk,
                                vector=[float(value) for value in vector],
                                embedding_model=embedding_model,
                                embedding_dim=len(vector),
                                normalized=normalized,
                            )
                            if cache is not None:
                                cache.set(
                                    cache_key,
                                    embedded_by_index[index].vector,
                                    metadata={
                                        "embedding_model": embedding_model,
                                        "embedding_dim": embedded_by_index[index].embedding_dim,
                                        "normalized": normalized,
                                        "prefix_mode": prefix_mode,
                                        "input_type": "document",
                                        "prefix": document_prefix,
                                    },
                                )
                                cache_writes += 1

                        except Exception as item_exc:
                            logger.error(
                                "Embedding failure: chunk_id=%s document_id=%s chunk_index=%s source_path=%s title=%s error_type=%s error_message=%s",
                                chunk.id,
                                chunk.document_id,
                                chunk.chunk_index,
                                chunk.metadata.get("source_path")
                                if isinstance(chunk.metadata, dict)
                                else None,
                                chunk.metadata.get("title")
                                if isinstance(chunk.metadata, dict)
                                else None,
                                type(item_exc).__name__,
                                str(item_exc),
                            )

                            errors.append(build_chunk_error_record(chunk=chunk, exc=item_exc))

        embedded_chunks = [
            embedded_by_index[index]
            for index in range(len(chunks_to_process))
            if index in embedded_by_index
        ]

        if not cache_enabled:
            cache_hits = 0
            cache_misses = len(embedded_chunks)

        write_jsonl(artifact_paths.output_path, embedded_chunks)

        logger.info("Wrote embedded chunks artifact: path=%s", artifact_paths.output_path)

        finished_at, duration_ms = finish_timed_run(timed_run)
        embedding_dim = embedded_chunks[0].embedding_dim if embedded_chunks else 0
        embedding_dims = {item.embedding_dim for item in embedded_chunks}

        if len(embedding_dims) > 1:
            mismatch = ValueError(
                f"Embedding dimension mismatch detected: dims={sorted(embedding_dims)}"
            )
            logger.error("Embedding dimension mismatch: dims=%s", sorted(embedding_dims))

            errors.append(
                build_stage_error_record(
                    exc=mismatch,
                    context={
                        "chunk_id": None,
                        "document_id": None,
                        "source_path": None,
                        "source_name": None,
                        "source_type": None,
                        "title": None,
                        "chunk_index": None,
                    },
                )
            )

            if fail_fast:
                fail_fast_error = mismatch

        vector_stats = self._build_vector_stats(embedded_chunks)

        hit_rate = (cache_hits / len(chunks_to_process)) if chunks_to_process else 0.0
        success_rate = len(embedded_chunks) / len(chunks_to_process) if chunks_to_process else 0.0
        chunks_per_second = (
            len(chunks_to_process) / (duration_ms / 1000)
            if duration_ms > 0
            else float(len(chunks_to_process))
        )

        if vector_stats["zero_vector_count"] > 0:
            logger.warning(
                "Zero vectors detected: zero_vector_count=%s vector_count=%s",
                vector_stats["zero_vector_count"],
                vector_stats["vector_count"],
            )

        logger.info(
            "Embedding complete: embedded=%s failed=%s skipped=%s success_rate=%.3f cache_hits=%s cache_misses=%s cache_writes=%s cache_hit_rate=%.3f single_item_fallback_attempts=%s embedding_dim=%s chunks_per_second=%.2f",
            len(embedded_chunks),
            len(errors),
            skipped_chunks_total,
            success_rate,
            cache_hits,
            cache_misses,
            cache_writes,
            hit_rate,
            single_item_fallback_attempts,
            embedding_dim,
            chunks_per_second,
        )

        if len(embedded_chunks) == 0:
            logger.warning("Zero embedded chunks produced for embeddings stage.")

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
            "cache_path": str(artifact_paths.cache_path),
            "chunks_total": chunks_total,
            "embedded_chunks_total": len(embedded_chunks),
            "failed_chunks_total": len(errors),
            "skipped_chunks_total": skipped_chunks_total,
            "provider": settings.embeddings.provider,
            "embedding_model": embedding_model,
            "embedding_dim": embedding_dim,
            "normalized": normalized,
            "prefix_mode": prefix_mode,
            "batch_size": batch_size,
            "cache_enabled": cache_enabled,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_writes": cache_writes,
            "query_prefix": query_prefix,
            "document_prefix": document_prefix,
            **vector_stats,
        }

        write_manifest(artifact_paths.manifest_path, manifest)
        logger.info("Wrote embedding manifest: path=%s", artifact_paths.manifest_path)
        status = compute_layer_status(
            success_count=len(embedded_chunks),
            error_count=len(errors),
        )

        logger.info(
            "Finished embeddings layer: status=%s duration_ms=%s chunks_total=%s embedded_chunks_total=%s failed_chunks_total=%s skipped_chunks_total=%s",
            status,
            duration_ms,
            chunks_total,
            len(embedded_chunks),
            len(errors),
            skipped_chunks_total,
        )

        result = BuildEmbeddingsResult(
            run_id=timed_run.run_id,
            input_path=artifact_paths.input_path,
            output_path=artifact_paths.output_path,
            manifest_path=artifact_paths.manifest_path,
            cache_path=artifact_paths.cache_path,
            chunks_total=chunks_total,
            embedded_chunks_total=len(embedded_chunks),
            failed_chunks_total=len(errors),
            skipped_chunks_total=skipped_chunks_total,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            duration_ms=duration_ms,
        )
        if fail_fast_error is not None:
            raise RuntimeError(str(fail_fast_error)) from fail_fast_error
        return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build embedded chunks from chunk artifacts.",
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
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--fail-fast", action="store_true", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    return parser


def main() -> None:
    setup_logging()
    args = _build_arg_parser().parse_args()
    layer = BuildEmbeddingsLayer(
        config_path=args.config,
        input_path=args.input_path,
        output_path=args.output_path,
        manifest_path=args.manifest_path,
        cache_path=args.cache_path,
        fail_fast=args.fail_fast,
        limit=args.limit,
        cache_enabled=False if args.no_cache else None,
    )

    try:
        result = layer.run()
    except Exception:
        logger.exception("BuildEmbeddingsLayer failed")
        sys.exit(1)

    if args.fail_fast and result.failed_chunks_total > 0:
        sys.exit(1)
    if result.embedded_chunks_total == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
