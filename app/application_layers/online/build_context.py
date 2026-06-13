"""Online context-building layer: reranked results -> context bundle."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.domain.models.context import ContextBundle
from app.domain.models.retrieval_result import RetrievalResult
from app.infrastructure.context import create_context_builder
from app.infrastructure.storage.manifest_store import write_manifest
from app.infrastructure.storage.trace_loader import load_candidates_from_trace
from app.utils.config import load_settings, resolve_config_dir_and_path, to_optional_path
from app.utils.constants import STAGE_BUILD_CONTEXT
from app.utils.logging import get_logger, setup_logging
from app.utils.timing import finish_timed_run, start_timed_run

logger = get_logger(__name__)


@dataclass(frozen=True)
class BuildContextResult:
    """Runtime result for one online context-building execution."""

    run_id: str
    question: str
    context_bundle: ContextBundle
    input_results_total: int
    included_chunks_total: int
    dropped_chunks_total: int
    token_budget: int
    token_count: int
    truncated: bool
    sources_total: int
    trace_path: Path | None
    duration_ms: int


class BuildContextLayer:
    """Orchestrates context building for a single user question."""

    def __init__(
        self,
        *,
        question: str,
        results: list[RetrievalResult],
        config_path: Path | str = Path("configs/config.yaml"),
        token_budget: int | None = None,
        max_chunks: int | None = None,
        trace_path: Path | str | None = None,
        write_trace: bool | None = None,
        context_builder: Any | None = None,
    ) -> None:
        if not question.strip():
            raise ValueError("question must not be empty.")
        if token_budget is not None and token_budget <= 0:
            raise ValueError("token_budget must be > 0 when provided.")
        if max_chunks is not None and max_chunks <= 0:
            raise ValueError("max_chunks must be > 0 when provided.")

        self._question = question.strip()
        self._results = list(results)
        self._config_path = Path(config_path)
        self._token_budget_override = token_budget
        self._max_chunks_override = max_chunks
        self._trace_path_override = to_optional_path(trace_path)
        self._write_trace_override = write_trace
        self._context_builder = context_builder

    @property
    def context_builder(self) -> Any | None:
        """Return context builder used by this layer execution."""
        return self._context_builder

    @classmethod
    def warmup(
        cls,
        *,
        config_path: Path | str = Path("configs/config.yaml"),
        context_builder: Any | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Warm context builder and return component with warmup payload."""
        config_dir, _ = resolve_config_dir_and_path(Path(config_path))
        settings = load_settings(config_dir)
        setup_logging(settings)

        resolved_context_builder = (
            context_builder if context_builder is not None else create_context_builder(settings)
        )
        warmup = getattr(resolved_context_builder, "warmup", None)
        warmup_payload: dict[str, Any] = {}
        if callable(warmup):
            warmup_payload = cast(dict[str, Any], warmup())
        return resolved_context_builder, warmup_payload

    def run(self) -> BuildContextResult:
        """Run context building and optionally write stage trace."""

        config_dir, resolved_config_path = resolve_config_dir_and_path(self._config_path)
        settings = load_settings(config_dir)

        setup_logging(settings)

        context_settings = settings.context
        token_budget = (
            self._token_budget_override
            if self._token_budget_override is not None
            else context_settings.token_budget
        )
        max_chunks = (
            self._max_chunks_override
            if self._max_chunks_override is not None
            else context_settings.max_chunks
        )
        if token_budget <= 0:
            logger.error("Invalid token budget resolved in BuildContextLayer.")
            raise ValueError("Resolved token_budget must be > 0.")
        if max_chunks <= 0:
            logger.error("Invalid max chunks resolved in BuildContextLayer.")
            raise ValueError("Resolved max_chunks must be > 0.")

        write_trace = (
            self._write_trace_override
            if self._write_trace_override is not None
            else context_settings.write_trace
        )
        trace_dir = context_settings.trace_dir
        text_preview_chars = context_settings.text_preview_chars
        include_full_context_in_trace = context_settings.include_full_context_in_trace

        timed_run = start_timed_run(STAGE_BUILD_CONTEXT)
        trace_path = (
            self._trace_path_override
            if self._trace_path_override is not None
            else trace_dir / f"{timed_run.run_id}_context.json"
        )

        logger.info(
            "Starting context layer: run_id=%s stage=%s input_results_total=%s token_budget=%s max_chunks=%s",
            timed_run.run_id,
            STAGE_BUILD_CONTEXT,
            len(self._results),
            token_budget,
            max_chunks,
        )

        builder = (
            self._context_builder
            if self._context_builder is not None
            else create_context_builder(settings)
        )
        self._context_builder = builder
        try:
            context_bundle = builder.build(
                query=self._question,
                results=self._results,
                token_budget=token_budget,
                max_chunks=max_chunks,
            )
        except Exception:
            logger.exception("Context builder failure in BuildContextLayer.")
            raise

        finished_at, duration_ms = finish_timed_run(timed_run)
        included_chunks_total = len(context_bundle.chunks)
        dropped_chunks_total = len(context_bundle.dropped_chunk_ids)
        sources_total = len(context_bundle.sources)

        if context_bundle.truncated:
            logger.warning(
                "Context is truncated: dropped_chunks_total=%s token_count=%s token_budget=%s",
                dropped_chunks_total,
                context_bundle.token_count,
                token_budget,
            )
        if included_chunks_total == 0:
            logger.warning("Context layer produced zero included chunks.")

        logger.info(
            "Finished context layer: run_id=%s token_count=%s sources_total=%s duration_ms=%s",
            timed_run.run_id,
            context_bundle.token_count,
            sources_total,
            duration_ms,
        )

        trace_path_value: Path | None = trace_path
        if write_trace:
            payload = self._build_trace_payload(
                run_id=timed_run.run_id,
                started_at=timed_run.started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
                config_path=resolved_config_path,
                context_bundle=context_bundle,
                input_results_total=len(self._results),
                include_full_context_in_trace=include_full_context_in_trace,
                text_preview_chars=text_preview_chars,
            )
            try:
                write_manifest(trace_path, payload)
                logger.info("Wrote context trace: path=%s", trace_path)
            except Exception:
                logger.exception("Failed to write context trace: path=%s", trace_path)
                raise
        else:
            logger.warning("Trace writing disabled for context run.")
            trace_path_value = None

        return BuildContextResult(
            run_id=timed_run.run_id,
            question=self._question,
            context_bundle=context_bundle,
            input_results_total=len(self._results),
            included_chunks_total=included_chunks_total,
            dropped_chunks_total=dropped_chunks_total,
            token_budget=token_budget,
            token_count=context_bundle.token_count,
            truncated=context_bundle.truncated,
            sources_total=sources_total,
            trace_path=trace_path_value,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _build_trace_payload(
        *,
        run_id: str,
        started_at: str,
        finished_at: str,
        duration_ms: int,
        config_path: Path,
        context_bundle: ContextBundle,
        input_results_total: int,
        include_full_context_in_trace: bool,
        text_preview_chars: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": run_id,
            "stage": STAGE_BUILD_CONTEXT,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": duration_ms,
            "config_path": str(config_path),
            "question": context_bundle.query,
            "input_results_total": input_results_total,
            "included_chunks_total": len(context_bundle.chunks),
            "dropped_chunks_total": len(context_bundle.dropped_chunk_ids),
            "context_bundle": context_bundle.model_dump(mode="json"),
            "rendered_context_preview": context_bundle.rendered_context[:text_preview_chars],
        }
        if include_full_context_in_trace:
            payload["context_bundle"]["rendered_context"] = context_bundle.rendered_context
        return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run online context building from rerank trace candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--trace-path", type=Path, default=None)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--print-context", action="store_true")
    return parser


def main() -> None:
    setup_logging()

    args = _build_arg_parser().parse_args()

    try:
        results = load_candidates_from_trace(args.input_path)
        layer = BuildContextLayer(
            question=args.question,
            results=results,
            config_path=args.config,
            token_budget=args.token_budget,
            max_chunks=args.max_chunks,
            trace_path=args.trace_path,
            write_trace=False if args.no_trace else None,
        )

        result = layer.run()

    except Exception:
        logger.exception("BuildContextLayer failed")
        sys.exit(1)

    logger.info(
        "Context result: question=%r input_results_total=%s included_chunks_total=%s "
        "dropped_chunks_total=%s token_count=%s token_budget=%s truncated=%s "
        "sources_total=%s duration_ms=%s trace_path=%s",
        result.question,
        result.input_results_total,
        result.included_chunks_total,
        result.dropped_chunks_total,
        result.token_count,
        result.token_budget,
        result.truncated,
        result.sources_total,
        result.duration_ms,
        result.trace_path,
    )
    if args.print_context:
        print(result.context_bundle.rendered_context)


if __name__ == "__main__":
    main()
