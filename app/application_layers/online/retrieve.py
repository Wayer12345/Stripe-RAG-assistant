"""Online retrieval layer: question -> dense vector search results."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from app.domain.interfaces.retriever import Retriever
from app.domain.models.retrieval_result import RetrievalResult
from app.infrastructure.retrieval.retriever_factory import (
    create_cached_retriever,
    shutdown_retriever_cache,
)
from app.infrastructure.storage.manifest_store import write_manifest
from app.utils.config import load_settings, resolve_config_dir_and_path, to_optional_path
from app.utils.constants import DEFAULT_RETRIEVE_TEXT_PREVIEW_CHARS, STAGE_RETRIEVE
from app.utils.logging import get_logger, setup_logging
from app.utils.timing import finish_timed_run, start_timed_run

logger = get_logger(__name__)
_QUESTION_LOG_PREVIEW_CHARS = 120


@dataclass(frozen=True)
class RetrieveResult:
    """Runtime result for one online retrieval execution."""

    run_id: str
    question: str
    results: list[RetrievalResult]
    results_total: int
    top_k: int
    strategy: str
    embedding_model: str
    embedding_dim: int | None
    filters: dict[str, Any] | None
    trace_path: Path | None
    duration_ms: int


class RetrieveLayer:
    """Orchestrates dense retrieval for a single user question."""

    def __init__(
        self,
        *,
        question: str,
        config_path: Path | str = Path("configs/config.yaml"),
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        trace_path: Path | str | None = None,
        write_trace: bool | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        if not question.strip():
            raise ValueError("question must not be empty.")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be > 0 when provided.")

        self._question = question.strip()
        self._config_path = Path(config_path)
        self._top_k_override = top_k
        self._filters = filters
        self._trace_path_override = to_optional_path(trace_path)
        self._write_trace_override = write_trace
        self._retriever = retriever

    @property
    def retriever(self) -> Retriever | None:
        """Return retriever used by this layer execution."""
        return self._retriever

    @classmethod
    def warmup(
        cls,
        *,
        config_path: Path | str = Path("configs/config.yaml"),
        retriever: Retriever | None = None,
    ) -> tuple[Retriever, dict[str, Any]]:
        """Warm retrieval component and return component with warmup payload."""
        config_dir, _ = resolve_config_dir_and_path(Path(config_path))
        settings = load_settings(config_dir)
        setup_logging(settings)

        resolved_retriever = retriever or create_cached_retriever(settings)
        warmup = getattr(resolved_retriever, "warmup", None)
        warmup_payload: dict[str, Any] = {}
        if callable(warmup):
            warmup_payload = cast(dict[str, Any], warmup())
        return resolved_retriever, warmup_payload

    def run(self) -> RetrieveResult:
        """Run dense retrieval and optionally write query trace."""

        config_dir, resolved_config_path = resolve_config_dir_and_path(self._config_path)
        settings = load_settings(config_dir)

        setup_logging(settings)

        strategy = getattr(settings.retrieval, "strategy", "dense")
        if strategy != "dense":
            logger.warning("Unsupported retrieval strategy attempted: strategy=%s", strategy)
            raise ValueError(f"Unsupported retrieval strategy for this implementation: {strategy}")

        top_k = (
            self._top_k_override
            if self._top_k_override is not None
            else getattr(settings.retrieval, "dense_top_k", 30)
        )

        if top_k <= 0:
            raise ValueError("Resolved retrieval top_k must be > 0.")

        write_trace = (
            self._write_trace_override
            if self._write_trace_override is not None
            else getattr(settings.retrieval, "write_trace", True)
        )
        trace_dir = Path(getattr(settings.retrieval, "trace_dir", Path("data/traces/queries")))

        timed_run = start_timed_run(STAGE_RETRIEVE)

        trace_path = (
            self._trace_path_override
            if self._trace_path_override is not None
            else trace_dir / f"{timed_run.run_id}_retrieve.json"
        )

        question_preview = (
            self._question[:_QUESTION_LOG_PREVIEW_CHARS]
            if len(self._question) <= _QUESTION_LOG_PREVIEW_CHARS
            else f"{self._question[:_QUESTION_LOG_PREVIEW_CHARS]}..."
        )

        logger.info(
            "Starting retrieval layer: run_id=%s stage=%s strategy=%s top_k=%s question_len=%s question_preview=%s filters=%s config_path=%s",
            timed_run.run_id,
            STAGE_RETRIEVE,
            strategy,
            top_k,
            len(self._question),
            question_preview,
            self._filters,
            resolved_config_path,
        )
        logger.info(
            "Retrieval infra config: qdrant_collection=%s embedding_model=%s",
            settings.vector_store.collection_name,
            settings.embeddings.model_name,
        )

        if top_k > settings.retrieval.dense_top_k:
            logger.warning(
                "top_k override larger than configured dense_top_k: top_k=%s configured_dense_top_k=%s",
                top_k,
                settings.retrieval.dense_top_k,
            )

        retriever = self._retriever or create_cached_retriever(settings)
        self._retriever = retriever

        collection_exists_fn = getattr(retriever, "collection_exists", None)

        if callable(collection_exists_fn):
            try:
                if not collection_exists_fn():
                    logger.warning(
                        "Qdrant collection missing before search: collection=%s",
                        settings.vector_store.collection_name,
                    )
            except Exception:
                logger.warning("Unable to pre-check Qdrant collection existence.")

        results: list[RetrievalResult] = []

        try:
            results = retriever.retrieve(
                self._question,
                top_k=top_k,
                filters=self._filters,
            )
        except Exception:
            logger.exception("Retrieval run failed.")
            raise

        if not results:
            logger.warning("Retrieval returned empty result set.")

        finished_at, duration_ms = finish_timed_run(timed_run)

        results_total = len(results)
        top_score = results[0].dense_score if results else None
        logger.info(
            "Finished retrieval layer: run_id=%s results_total=%s top_score=%s duration_ms=%s",
            timed_run.run_id,
            results_total,
            top_score,
            duration_ms,
        )

        embedding_dim = self._resolve_embedding_dim(results)

        if write_trace:
            trace_payload = self._build_trace_payload(
                run_id=timed_run.run_id,
                started_at=timed_run.started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
                config_path=resolved_config_path,
                question=self._question,
                strategy=strategy,
                top_k=top_k,
                filters=self._filters,
                results=results,
                embedding_model=settings.embeddings.model_name,
                embedding_dim=embedding_dim,
            )

            try:
                write_manifest(trace_path, trace_payload)
                logger.info("Wrote retrieval trace: path=%s", trace_path)
            except Exception:
                logger.exception("Failed to write retrieval trace: path=%s", trace_path)
                raise

        else:
            logger.warning("Trace writing disabled for retrieval run.")
            trace_path = None

        return RetrieveResult(
            run_id=timed_run.run_id,
            question=self._question,
            results=results,
            results_total=results_total,
            top_k=top_k,
            strategy=strategy,
            embedding_model=settings.embeddings.model_name,
            embedding_dim=embedding_dim,
            filters=self._filters,
            trace_path=trace_path,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _resolve_embedding_dim(results: list[RetrievalResult]) -> int | None:
        if not results:
            return None
        value = results[0].metadata.get("embedding_dim")
        if isinstance(value, int) and value > 0:
            return value
        return None

    @staticmethod
    def _build_trace_payload(
        *,
        run_id: str,
        started_at: str,
        finished_at: str,
        duration_ms: int,
        config_path: Path,
        question: str,
        strategy: str,
        top_k: int,
        filters: dict[str, Any] | None,
        results: list[RetrievalResult],
        embedding_model: str,
        embedding_dim: int | None,
    ) -> dict[str, Any]:
        serialized_results: list[dict[str, Any]] = []
        for rank, result in enumerate(results, start=1):
            serialized_results.append(
                {
                    "rank": rank,
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "title": result.source.title,
                    "url": result.source.url,
                    "source_type": result.source.source_type or result.metadata.get("source_type"),
                    "dense_score": result.dense_score,
                    "final_score": result.final_score,
                    "token_count": result.metadata.get("token_count"),
                    "text_preview": result.text[:DEFAULT_RETRIEVE_TEXT_PREVIEW_CHARS],
                }
            )
        return {
            "run_id": run_id,
            "stage": STAGE_RETRIEVE,
            "question": question,
            "strategy": strategy,
            "top_k": top_k,
            "filters": filters,
            "results_total": len(results),
            "results": serialized_results,
            "embedding_model": embedding_model,
            "embedding_dim": embedding_dim,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": duration_ms,
            "config_path": str(config_path),
        }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run online dense retrieval for one user question.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--filter-document-id", type=str, default=None)
    parser.add_argument("--filter-source-type", type=str, default=None)
    parser.add_argument("--filter-url", type=str, default=None)
    parser.add_argument("--trace-path", type=Path, default=None)
    parser.add_argument("--no-trace", action="store_true")
    return parser


def _build_filters(args: argparse.Namespace) -> dict[str, Any] | None:
    filters: dict[str, Any] = {}
    if args.filter_document_id:
        filters["document_id"] = args.filter_document_id
    if args.filter_source_type:
        filters["source_type"] = args.filter_source_type
    if args.filter_url:
        filters["url"] = args.filter_url
    return filters or None


def main() -> None:
    setup_logging()

    args = _build_arg_parser().parse_args()

    filters = _build_filters(args)

    try:
        layer = RetrieveLayer(
            question=args.question,
            config_path=args.config,
            top_k=args.top_k,
            filters=filters,
            trace_path=args.trace_path,
            write_trace=False if args.no_trace else None,
        )

        result = layer.run()

    except Exception:
        logger.exception("RetrieveLayer failed")
        sys.exit(1)

    finally:
        # CLI entrypoint is process-scoped; explicitly close cached resources.
        shutdown_retriever_cache()
    logger.info(
        "Retrieval result: question=%r strategy=%s top_k=%s results_total=%s duration_ms=%s",
        result.question,
        result.strategy,
        result.top_k,
        result.results_total,
        result.duration_ms,
    )
    for rank, item in enumerate(result.results[:5], start=1):
        preview = item.text[:160].replace("\n", " ").strip()
        logger.info(
            "rank=%s score=%.4f title=%r url=%r chunk_id=%s preview=%s",
            rank,
            item.final_score,
            item.source.title,
            item.source.url,
            item.chunk_id,
            preview,
        )


if __name__ == "__main__":
    main()
