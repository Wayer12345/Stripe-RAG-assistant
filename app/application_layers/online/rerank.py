"""Online reranking layer: query + retrieval candidates -> reranked results."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.domain.models.retrieval_result import RetrievalResult
from app.infrastructure.reranking import create_reranker
from app.infrastructure.storage.manifest_store import write_manifest
from app.infrastructure.storage.trace_loader import load_candidates_from_trace
from app.utils.config import load_settings, resolve_config_dir_and_path, to_optional_path
from app.utils.constants import STAGE_RERANK
from app.utils.logging import get_logger, setup_logging
from app.utils.timing import finish_timed_run, start_timed_run

logger = get_logger(__name__)


@dataclass(frozen=True)
class RerankResult:
    """Runtime result for one online reranking execution."""

    run_id: str
    question: str
    input_results_total: int
    reranked_results_total: int
    results: list[RetrievalResult]
    enabled: bool
    model_name: str
    top_k_before: int
    top_k_after: int
    latency_budget_ms: int
    latency_budget_exceeded: bool
    cache_hits: int
    cache_misses: int
    trace_path: Path | None
    duration_ms: int


class RerankLayer:
    """Orchestrates candidate reranking for a single user question."""

    def __init__(
        self,
        *,
        question: str,
        candidates: list[RetrievalResult],
        config_path: Path | str = Path("configs/config.yaml"),
        top_k_before: int | None = None,
        top_k_after: int | None = None,
        trace_path: Path | str | None = None,
        write_trace: bool | None = None,
        reranker: Any | None = None,
    ) -> None:
        if not question.strip():
            raise ValueError("question must not be empty.")
        if top_k_before is not None and top_k_before <= 0:
            raise ValueError("top_k_before must be > 0 when provided.")
        if top_k_after is not None and top_k_after <= 0:
            raise ValueError("top_k_after must be > 0 when provided.")
        if top_k_before is not None and top_k_after is not None and top_k_after > top_k_before:
            raise ValueError("top_k_after must be <= top_k_before.")

        self._question = question.strip()
        self._candidates = list(candidates)
        self._config_path = Path(config_path)
        self._top_k_before_override = top_k_before
        self._top_k_after_override = top_k_after
        self._trace_path_override = to_optional_path(trace_path)
        self._write_trace_override = write_trace
        self._reranker = reranker

    @property
    def reranker(self) -> Any | None:
        """Return reranker used by this layer execution."""
        return self._reranker

    @classmethod
    def warmup(
        cls,
        *,
        config_path: Path | str = Path("configs/config.yaml"),
        reranker: Any | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Warm reranker component and return component with warmup payload."""
        config_dir, _ = resolve_config_dir_and_path(Path(config_path))
        settings = load_settings(config_dir)
        setup_logging(settings)

        resolved_reranker = reranker if reranker is not None else create_reranker(settings)
        warmup = getattr(resolved_reranker, "warmup", None)
        warmup_payload: dict[str, Any] = {}
        if callable(warmup) and settings.reranking.warmup_enabled:
            warmup_payload = cast(dict[str, Any], warmup())
        return resolved_reranker, warmup_payload

    def run(self) -> RerankResult:
        """Run reranking and optionally write stage trace."""
        config_dir, resolved_config_path = resolve_config_dir_and_path(self._config_path)
        settings = load_settings(config_dir)

        setup_logging(settings)

        reranking_settings = settings.reranking
        enabled = reranking_settings.enabled
        top_k_before = (
            self._top_k_before_override
            if self._top_k_before_override is not None
            else reranking_settings.top_k_before
        )
        top_k_after = (
            self._top_k_after_override
            if self._top_k_after_override is not None
            else reranking_settings.top_k_after
        )
        if top_k_before <= 0:
            raise ValueError("Resolved top_k_before must be > 0.")
        if top_k_after <= 0:
            raise ValueError("Resolved top_k_after must be > 0.")
        if top_k_after > top_k_before:
            raise ValueError("Resolved top_k_after must be <= top_k_before.")

        write_trace = (
            self._write_trace_override
            if self._write_trace_override is not None
            else reranking_settings.write_trace
        )

        trace_dir = reranking_settings.trace_dir
        latency_budget_ms = reranking_settings.latency_budget_ms
        model_name = reranking_settings.model_name
        text_preview_chars = reranking_settings.text_preview_chars

        timed_run = start_timed_run(STAGE_RERANK)

        trace_path = (
            self._trace_path_override
            if self._trace_path_override is not None
            else trace_dir / f"{timed_run.run_id}_rerank.json"
        )

        logger.info(
            "Starting rerank layer: run_id=%s stage=%s candidates=%s enabled=%s model_name=%s top_k_before=%s top_k_after=%s trace_path=%s",
            timed_run.run_id,
            STAGE_RERANK,
            len(self._candidates),
            enabled,
            model_name,
            top_k_before,
            top_k_after,
            trace_path,
        )

        if not self._candidates:
            logger.warning("Rerank layer received zero candidates.")

        reranked_results: list[RetrievalResult]
        cache_hits = 0
        cache_misses = 0
        latency_budget_exceeded = False

        if not enabled:
            logger.warning(
                "Reranking disabled by configuration; returning input candidates truncated."
            )
            reranked_results = self._candidates[:top_k_after]
            stats: dict[str, Any] = {}
        else:
            reranker = self._reranker if self._reranker is not None else create_reranker(settings)
            self._reranker = reranker

            if reranking_settings.warmup_enabled and hasattr(reranker, "warmup"):
                reranker.warmup()
            try:
                reranked_results = reranker.rerank(
                    self._question,
                    self._candidates,
                    top_k_before=top_k_before,
                    top_k_after=top_k_after,
                )
            except Exception:
                logger.exception("Reranker failure in RerankLayer.")
                raise

            stats = reranker.last_stats() if hasattr(reranker, "last_stats") else {}
            if hasattr(reranker, "model_name"):
                model_name = str(reranker.model_name())
            cache_hits = int(stats.get("cache_hits", 0))
            cache_misses = int(stats.get("cache_misses", 0))
            latency_budget_exceeded = bool(stats.get("latency_budget_exceeded", False))

        finished_at, duration_ms = finish_timed_run(timed_run)

        if duration_ms > latency_budget_ms:
            latency_budget_exceeded = True

        if latency_budget_exceeded:
            logger.warning(
                "Rerank latency budget exceeded: duration_ms=%s latency_budget_ms=%s",
                duration_ms,
                latency_budget_ms,
            )

        logger.info(
            "Finished rerank layer: run_id=%s input_results_total=%s reranked_results_total=%s duration_ms=%s cache_hits=%s cache_misses=%s",
            timed_run.run_id,
            len(self._candidates),
            len(reranked_results),
            duration_ms,
            cache_hits,
            cache_misses,
        )

        trace_path_value: Path | None = trace_path
        if write_trace:
            payload = self._build_trace_payload(
                run_id=timed_run.run_id,
                question=self._question,
                enabled=enabled,
                model_name=model_name,
                input_results_total=len(self._candidates),
                reranked_results=reranked_results,
                top_k_before=top_k_before,
                top_k_after=top_k_after,
                latency_budget_ms=latency_budget_ms,
                latency_budget_exceeded=latency_budget_exceeded,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                started_at=timed_run.started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
                config_path=resolved_config_path,
                text_preview_chars=text_preview_chars,
            )

            try:
                write_manifest(trace_path, payload)
                logger.info("Wrote rerank trace: path=%s", trace_path)
            except Exception:
                logger.exception("Failed to write rerank trace: path=%s", trace_path)
                raise
        else:
            trace_path_value = None

        return RerankResult(
            run_id=timed_run.run_id,
            question=self._question,
            input_results_total=len(self._candidates),
            reranked_results_total=len(reranked_results),
            results=reranked_results,
            enabled=enabled,
            model_name=model_name,
            top_k_before=top_k_before,
            top_k_after=top_k_after,
            latency_budget_ms=latency_budget_ms,
            latency_budget_exceeded=latency_budget_exceeded,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            trace_path=trace_path_value,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _build_trace_payload(
        *,
        run_id: str,
        question: str,
        enabled: bool,
        model_name: str,
        input_results_total: int,
        reranked_results: list[RetrievalResult],
        top_k_before: int,
        top_k_after: int,
        latency_budget_ms: int,
        latency_budget_exceeded: bool,
        cache_hits: int,
        cache_misses: int,
        started_at: str,
        finished_at: str,
        duration_ms: int,
        config_path: Path,
        text_preview_chars: int,
    ) -> dict[str, Any]:
        serialized_results: list[dict[str, Any]] = []
        for rank, result in enumerate(reranked_results, start=1):
            serialized_results.append(
                {
                    "rank": rank,
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "title": result.source.title,
                    "url": result.source.url,
                    "dense_score": result.dense_score,
                    "retrieval_score": result.retrieval_score,
                    "lexical_score": result.lexical_score,
                    "reranker_score": result.reranker_score,
                    "final_score": result.final_score,
                    "text_preview": result.text[:text_preview_chars],
                }
            )

        return {
            "run_id": run_id,
            "stage": STAGE_RERANK,
            "question": question,
            "enabled": enabled,
            "model_name": model_name,
            "input_results_total": input_results_total,
            "reranked_results_total": len(reranked_results),
            "top_k_before": top_k_before,
            "top_k_after": top_k_after,
            "latency_budget_ms": latency_budget_ms,
            "latency_budget_exceeded": latency_budget_exceeded,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": duration_ms,
            "config_path": str(config_path),
            "results": serialized_results,
        }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run online reranking for retrieval candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--top-k-before", type=int, default=None)
    parser.add_argument("--top-k-after", type=int, default=None)
    parser.add_argument("--trace-path", type=Path, default=None)
    parser.add_argument("--no-trace", action="store_true")
    return parser


def main() -> None:
    setup_logging()

    args = _build_arg_parser().parse_args()

    try:
        candidates = load_candidates_from_trace(args.input_path)
        result = RerankLayer(
            question=args.question,
            candidates=candidates,
            config_path=args.config,
            top_k_before=args.top_k_before,
            top_k_after=args.top_k_after,
            trace_path=args.trace_path,
            write_trace=False if args.no_trace else None,
        ).run()

    except Exception:
        logger.exception("RerankLayer failed")
        sys.exit(1)
    logger.info(
        "Rerank result: question=%r enabled=%s model_name=%s top_k_before=%s top_k_after=%s "
        "input_results_total=%s reranked_results_total=%s cache_hits=%s cache_misses=%s "
        "latency_budget_ms=%s latency_budget_exceeded=%s duration_ms=%s",
        result.question,
        result.enabled,
        result.model_name,
        result.top_k_before,
        result.top_k_after,
        result.input_results_total,
        result.reranked_results_total,
        result.cache_hits,
        result.cache_misses,
        result.latency_budget_ms,
        result.latency_budget_exceeded,
        result.duration_ms,
    )
    for rank, item in enumerate(result.results[:5], start=1):
        preview = item.text[:160].replace("\n", " ").strip()
        logger.info(
            "rank=%s final=%.4f reranker=%s dense=%s title=%r url=%r chunk_id=%s preview=%s",
            rank,
            item.final_score,
            item.reranker_score,
            item.dense_score,
            item.source.title,
            item.source.url,
            item.chunk_id,
            preview,
        )


if __name__ == "__main__":
    main()
