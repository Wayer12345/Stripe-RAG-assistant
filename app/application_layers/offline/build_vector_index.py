"""Build Qdrant dense vector index from EmbeddedChunk artifacts."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from app.domain.interfaces.vector_store import VectorStore
from app.domain.models.embedded_chunk import EmbeddedChunk
from app.infrastructure.storage.artifact_paths import (
    VectorIndexArtifactPaths,
    resolve_vector_index_artifact_paths,
)
from app.infrastructure.storage.jsonl_store import count_jsonl, read_jsonl
from app.infrastructure.storage.manifest_store import build_base_manifest, write_manifest
from app.infrastructure.storage.pipeline_records import (
    build_embedded_chunk_error_record,
    build_stage_error_record,
    compute_layer_status,
)
from app.infrastructure.vector_stores.vector_store_factory import create_vector_store
from app.utils.config import (
    load_settings,
    resolve_config_dir_and_path,
    to_optional_path,
    validate_positive_limit,
)
from app.utils.constants import (
    ARTIFACT_SCHEMA_VERSION,
    STAGE_VECTOR_INDEXING,
)
from app.utils.logging import get_logger, setup_logging
from app.utils.timing import finish_timed_run, start_timed_run

_STAGE_NAME = STAGE_VECTOR_INDEXING
logger = get_logger(__name__)


@dataclass(frozen=True)
class BuildVectorIndexResult:
    """Summary of a vector indexing run."""

    run_id: str
    input_path: Path
    manifest_path: Path
    collection_name: str
    embedded_chunks_total: int
    upserted_points_total: int
    failed_chunks_total: int
    skipped_chunks_total: int
    embedding_model: str
    embedding_dim: int
    distance: str
    recreate_collection: bool
    duration_ms: int


class BuildVectorIndexLayer:
    """Offline layer that indexes embedded chunks into local Qdrant."""

    def __init__(
        self,
        *,
        config_path: Path | str = Path("configs/config.yaml"),
        input_path: Path | str | None = None,
        manifest_path: Path | str | None = None,
        collection_name: str | None = None,
        recreate_collection: bool | None = None,
        create_payload_indexes: bool | None = None,
        fail_fast: bool | None = None,
        limit: int | None = None,
        validate_only: bool | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        validate_positive_limit(limit)

        self._config_path = Path(config_path)
        self._input_path_override = to_optional_path(input_path)
        self._manifest_path_override = to_optional_path(manifest_path)
        self._collection_name_override = collection_name
        self._recreate_collection_override = recreate_collection
        self._create_payload_indexes_override = create_payload_indexes
        self._fail_fast_override = fail_fast
        self._limit = limit
        self._validate_only_override = validate_only
        self._vector_store = vector_store

    @staticmethod
    def _create_collection_with_dim(
        vector_store: VectorStore,
        *,
        recreate: bool,
        vector_dim: int,
    ) -> None:
        vector_store_any = cast(Any, vector_store)
        try:
            vector_store_any.create_collection(recreate=recreate, vector_dim=vector_dim)
        except TypeError:
            vector_store_any.create_collection(recreate=recreate)

    def run(self) -> BuildVectorIndexResult:
        """Run vector indexing stage and write index manifest."""

        config_dir, resolved_config_path = resolve_config_dir_and_path(self._config_path)
        settings = load_settings(config_dir)

        setup_logging(settings)

        artifact_paths: VectorIndexArtifactPaths = resolve_vector_index_artifact_paths(
            settings,
            input_path_override=self._input_path_override,
            manifest_path_override=self._manifest_path_override,
        )

        collection_name = (
            self._collection_name_override or settings.vector_store.collection_name
        ).strip()
        distance = settings.vector_store.distance
        recreate_collection = (
            self._recreate_collection_override
            if self._recreate_collection_override is not None
            else (
                settings.indexing.recreate_collection or settings.vector_store.recreate_collection
            )
        )
        create_payload_indexes = (
            self._create_payload_indexes_override
            if self._create_payload_indexes_override is not None
            else (
                settings.indexing.create_payload_indexes
                and settings.vector_store.create_payload_indexes
            )
        )
        validate_only = (
            self._validate_only_override
            if self._validate_only_override is not None
            else settings.indexing.validate_only
        )
        fail_fast = self._fail_fast_override if self._fail_fast_override is not None else True
        upsert_batch_size = settings.indexing.upsert_batch_size
        payload_indexes = settings.vector_store.payload_indexes
        vector_store_mode = settings.vector_store.mode
        vector_store_local_path = settings.vector_store.local_path

        timed_run = start_timed_run(_STAGE_NAME)

        logger.info(
            "Starting vector indexing layer: run_id=%s stage=%s config_path=%s fail_fast=%s limit=%s",
            timed_run.run_id,
            _STAGE_NAME,
            resolved_config_path,
            fail_fast,
            self._limit,
        )
        logger.info(
            "Resolved index paths: input_path=%s manifest_path=%s",
            artifact_paths.input_path,
            artifact_paths.manifest_path,
        )
        logger.info(
            "Qdrant config: mode=%s host=%s port=%s url=%s collection=%s distance=%s recreate=%s create_payload_indexes=%s upsert_batch_size=%s wait=%s validate_only=%s",
            settings.vector_store.mode,
            settings.vector_store.host,
            settings.vector_store.port,
            settings.vector_store.url,
            collection_name,
            distance,
            recreate_collection,
            create_payload_indexes,
            upsert_batch_size,
            settings.vector_store.wait,
            validate_only,
        )
        if recreate_collection:
            logger.warning(
                "Collection recreation is enabled and may drop existing vectors: collection=%s",
                collection_name,
            )

        if not create_payload_indexes:
            logger.warning("Payload index creation is disabled for this run.")

        errors: list[dict[str, Any]] = []
        collection_count_after_upsert: int | None = None
        collection_count_before_upsert: int | None = None
        embedded_chunks_total = 0
        upserted_points_total = 0
        skipped_chunks_total = 0
        embedding_model = ""
        embedding_dim = 0
        normalized = False
        vector_store_provider = settings.vector_store.provider

        total_available = (
            count_jsonl(artifact_paths.input_path) if self._limit is not None else None
        )

        try:
            chunks_to_process = read_jsonl(artifact_paths.input_path, EmbeddedChunk, limit=self._limit)

            embedded_chunks_total = len(chunks_to_process)
            skipped_chunks_total = (
                (total_available - len(chunks_to_process)) if total_available is not None else 0
            )
            logger.info("Loaded embedded chunks: total=%s limit=%s", embedded_chunks_total, self._limit)
            if not chunks_to_process:
                logger.warning(
                    "Zero embedded chunks loaded: input_path=%s", artifact_paths.input_path
                )
            logger.info("Chunks to process: count=%s", embedded_chunks_total)

            (
                embedding_dim,
                embedding_model,
                normalized,
                consistency_failures,
            ) = EmbeddedChunk.validate_batch_consistency(chunks_to_process)
            consistency_errors = [
                build_embedded_chunk_error_record(embedded_chunk=chunk, exc=exc)
                for chunk, exc in consistency_failures
            ]
            errors.extend(consistency_errors)

            if consistency_errors:
                logger.warning(
                    "Embedding consistency issues detected: count=%s",
                    len(consistency_errors),
                )
            logger.info(
                "Inferred embeddings: model=%s dim=%s normalized=%s",
                embedding_model,
                embedding_dim,
                normalized,
            )

            if consistency_errors and fail_fast:
                logger.error(
                    "Inconsistent embedding metadata detected: errors=%s",
                    len(consistency_errors),
                )
                raise ValueError("Embedded chunks contain inconsistent embedding metadata.")

            vector_store = self._vector_store or create_vector_store(
                settings,
                collection_name=collection_name,
                distance=distance,
                upsert_batch_size=upsert_batch_size,
                payload_indexes=payload_indexes,
            )

            if not vector_store.healthcheck():
                healthcheck_error = None
                healthcheck_error_getter = getattr(vector_store, "healthcheck_error", None)

                if callable(healthcheck_error_getter):
                    healthcheck_error = healthcheck_error_getter()

                if healthcheck_error:
                    logger.error("Qdrant healthcheck failure: error=%s", healthcheck_error)
                    raise RuntimeError(f"Qdrant healthcheck failed: {healthcheck_error}")

                logger.error("Qdrant healthcheck failure without details.")
                raise RuntimeError("Qdrant healthcheck failed.")

            logger.info("Qdrant healthcheck succeeded.")

            if embedded_chunks_total > 0:
                if validate_only:
                    if not vector_store.collection_exists():
                        logger.error(
                            "Collection validation failed: collection does not exist: collection=%s",
                            collection_name,
                        )
                        raise ValueError(
                            "Collection validation requested but collection does not exist."
                        )

                    vector_store.validate_collection(embedding_dim)

                    logger.info(
                        "Collection validation succeeded: collection=%s embedding_dim=%s",
                        collection_name,
                        embedding_dim,
                    )

                else:
                    collection_exists = vector_store.collection_exists()
                    if collection_exists and not recreate_collection:
                        logger.warning(
                            "Collection already exists and will be reused: collection=%s",
                            collection_name,
                        )
                    elif not collection_exists:
                        logger.info(
                            "Collection does not exist and will be created: collection=%s",
                            collection_name,
                        )
                    if recreate_collection:
                        self._create_collection_with_dim(
                            vector_store,
                            recreate=True,
                            vector_dim=embedding_dim,
                        )
                        logger.info("Collection recreated: collection=%s", collection_name)

                    elif not collection_exists:
                        self._create_collection_with_dim(
                            vector_store,
                            recreate=False,
                            vector_dim=embedding_dim,
                        )
                        logger.info("Collection created: collection=%s", collection_name)

                    vector_store.validate_collection(embedding_dim)

                    logger.info(
                        "Collection config validation succeeded: collection=%s embedding_dim=%s",
                        collection_name,
                        embedding_dim,
                    )

                    payload_index_creator = getattr(vector_store, "create_payload_indexes", None)
                    if create_payload_indexes and callable(payload_index_creator):
                        try:
                            payload_index_creator()
                            logger.info(
                                "Payload indexes created: fields=%s",
                                sorted(payload_indexes.keys()),
                            )
                        except Exception as exc:
                            logger.error(
                                "Payload index creation failure: collection=%s error_type=%s error_message=%s",
                                collection_name,
                                type(exc).__name__,
                                str(exc),
                            )
                            raise RuntimeError(f"Failed to create payload indexes: {exc}") from exc

                    elif not create_payload_indexes:
                        logger.warning("Payload indexes skipped by configuration.")

                    if payload_indexes and chunks_to_process:
                        total_chunks = len(chunks_to_process)
                        for field_name in payload_indexes:
                            missing_count = sum(
                                1
                                for item in chunks_to_process
                                if not isinstance(item.chunk.metadata, dict)
                                or item.chunk.metadata.get(field_name) in (None, "")
                            )
                            if missing_count / total_chunks > 0.5:
                                logger.warning(
                                    "Payload field missing on many chunks: field=%s missing=%s total=%s",
                                    field_name,
                                    missing_count,
                                    total_chunks,
                                )

                    if settings.indexing.validate_after_upsert:
                        collection_count_before_upsert = vector_store.count()
                        logger.info(
                            "Collection count before upsert: count=%s",
                            collection_count_before_upsert,
                        )

                    upserted_points_total = vector_store.upsert(chunks_to_process)
                    logger.info("Upsert complete: points=%s", upserted_points_total)
                    if settings.indexing.validate_after_upsert:
                        collection_count_after_upsert = vector_store.count()
                        count_delta = (
                            collection_count_after_upsert - collection_count_before_upsert
                            if collection_count_before_upsert is not None
                            else None
                        )
                        logger.info(
                            "Collection count after upsert: count=%s delta=%s",
                            collection_count_after_upsert,
                            count_delta,
                        )
                    else:
                        logger.warning(
                            "Collection count verification unavailable; validate_after_upsert is disabled."
                        )

        except Exception as exc:
            logger.error(
                "Vector indexing stage error: error_type=%s error_message=%s",
                type(exc).__name__,
                str(exc),
            )
            if fail_fast:
                if not errors:
                    errors.append(
                        build_stage_error_record(
                            exc=exc,
                            context={
                                "chunk_id": None,
                                "document_id": None,
                                "embedding_model": embedding_model or None,
                                "embedding_dim": embedding_dim or None,
                            },
                        )
                    )
                raise

            errors.append(
                build_stage_error_record(
                    exc=exc,
                    context={
                        "chunk_id": None,
                        "document_id": None,
                        "embedding_model": embedding_model or None,
                        "embedding_dim": embedding_dim or None,
                    },
                )
            )

        finally:
            finished_at, duration_ms = finish_timed_run(timed_run)
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
                "collection_name": collection_name,
                "qdrant_host": getattr(settings.vector_store, "host", None),
                "qdrant_port": getattr(settings.vector_store, "port", None),
                "qdrant_url": getattr(settings.vector_store, "url", None),
                "qdrant_mode": vector_store_mode,
                "qdrant_local_path": str(vector_store_local_path),
                "distance": distance,
                "embedded_chunks_total": embedded_chunks_total,
                "upserted_points_total": upserted_points_total,
                "failed_chunks_total": len(errors),
                "skipped_chunks_total": skipped_chunks_total,
                "embedding_model": embedding_model or None,
                "embedding_dim": embedding_dim,
                "normalized": normalized,
                "recreate_collection": recreate_collection,
                "create_payload_indexes": create_payload_indexes,
                "upsert_batch_size": upsert_batch_size,
                "collection_count_after_upsert": collection_count_after_upsert,
                "embedded_chunks_artifact": str(artifact_paths.input_path),
                "qdrant_collection": collection_name,
                "vector_store_provider": vector_store_provider,
                "payload_indexes": payload_indexes if create_payload_indexes else {},
                "validate_only": validate_only,
            }

            write_manifest(artifact_paths.manifest_path, manifest)

            logger.info("Wrote vector index manifest: path=%s", artifact_paths.manifest_path)

            upsert_success_rate = (
                upserted_points_total / embedded_chunks_total if embedded_chunks_total > 0 else 0.0
            )
            points_per_second = (
                upserted_points_total / (duration_ms / 1000)
                if duration_ms > 0
                else float(upserted_points_total)
            )

            status = compute_layer_status(
                success_count=upserted_points_total,
                error_count=len(errors),
                zero_success_is_failure=(not validate_only) or embedded_chunks_total == 0,
            )

            logger.info(
                "Finished vector indexing layer: status=%s duration_ms=%s embedded_chunks_total=%s upserted_points_total=%s failed_chunks_total=%s skipped_chunks_total=%s upsert_success_rate=%.3f points_per_second=%.2f",
                status,
                duration_ms,
                embedded_chunks_total,
                upserted_points_total,
                len(errors),
                skipped_chunks_total,
                upsert_success_rate,
                points_per_second,
            )

        return BuildVectorIndexResult(
            run_id=timed_run.run_id,
            input_path=artifact_paths.input_path,
            manifest_path=artifact_paths.manifest_path,
            collection_name=collection_name,
            embedded_chunks_total=embedded_chunks_total,
            upserted_points_total=upserted_points_total,
            failed_chunks_total=len(errors),
            skipped_chunks_total=skipped_chunks_total,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            distance=distance,
            recreate_collection=recreate_collection,
            duration_ms=duration_ms,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Qdrant vector index from embedded chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--collection-name", type=str, default=None)
    parser.add_argument("--recreate-collection", action="store_true", default=None)
    parser.add_argument("--no-payload-indexes", action="store_true")
    parser.add_argument("--fail-fast", action="store_true", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true", default=None)
    return parser


def main() -> None:
    setup_logging()

    args = _build_arg_parser().parse_args()

    layer = BuildVectorIndexLayer(
        config_path=args.config,
        input_path=args.input_path,
        manifest_path=args.manifest_path,
        collection_name=args.collection_name,
        recreate_collection=args.recreate_collection,
        create_payload_indexes=False if args.no_payload_indexes else None,
        fail_fast=args.fail_fast,
        limit=args.limit,
        validate_only=args.validate_only,
    )

    try:
        result = layer.run()
    except Exception:
        logger.exception("BuildVectorIndexLayer failed")
        sys.exit(1)

    if result.embedded_chunks_total == 0:
        sys.exit(1)
    if result.failed_chunks_total > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
