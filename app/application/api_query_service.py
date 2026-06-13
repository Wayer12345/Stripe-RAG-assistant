"""API-specific online query service for long-lived FastAPI runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from app.application_layers.online.build_context import BuildContextLayer
from app.application_layers.online.generate_answer import GenerateAnswerLayer
from app.application_layers.online.rerank import RerankLayer
from app.application_layers.online.retrieve import RetrieveLayer
from app.domain.models.source import Source
from app.infrastructure.storage.manifest_store import write_json_payload
from app.utils.config import Settings, load_settings, resolve_config_dir_and_path
from app.utils.constants import STAGE_ONLINE_QUERY, STATUS_FAILED, STATUS_SUCCESS
from app.utils.ids import make_run_id
from app.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@dataclass(frozen=True)
class ApiQueryWarmupResult:
    """Warmup summary for service startup and health endpoints."""

    status: str
    started_at: str
    finished_at: str
    duration_ms: int
    components: dict[str, dict[str, Any]]
    error: dict[str, Any] | None = None


@dataclass(frozen=True)
class ApiQueryServiceResult:
    """Rich service result returned to API routes."""

    run_id: str
    question: str
    answer: str
    confidence: str
    sources: list[Source]
    sources_total: int
    retrieve_results_total: int
    reranked_results_total: int
    context_token_count: int
    context_sources_total: int
    context_truncated: bool
    latency_ms: dict[str, int]
    trace_paths: dict[str, str | None]
    debug: dict[str, Any] | None
    status: str
    error: dict[str, Any] | None


class ApiQueryService:
    """Long-lived in-memory online query service for FastAPI."""

    def __init__(
        self,
        *,
        config_path: Path | str = Path("configs/config.yaml"),
        settings: Settings | None = None,
        retriever: Any | None = None,
        reranker: Any | None = None,
        context_builder: Any | None = None,
        answer_generator: Any | None = None,
        retrieve_layer_cls: Any | None = None,
        rerank_layer_cls: Any | None = None,
        build_context_layer_cls: Any | None = None,
        generate_answer_layer_cls: Any | None = None,
    ) -> None:
        self._config_path = Path(config_path)

        if settings is None:
            config_dir, _ = resolve_config_dir_and_path(self._config_path)
            settings = load_settings(config_dir)

        self._settings = settings
        setup_logging(self._settings)

        self._retriever = retriever
        self._reranker = reranker
        self._context_builder = context_builder
        self._answer_generator = answer_generator
        self._owns_retriever = retriever is None
        self._owns_reranker = reranker is None
        self._owns_context_builder = context_builder is None
        self._owns_answer_generator = answer_generator is None

        self._retrieve_layer_cls = retrieve_layer_cls or RetrieveLayer
        self._rerank_layer_cls = rerank_layer_cls or RerankLayer
        self._build_context_layer_cls = build_context_layer_cls or BuildContextLayer
        self._generate_answer_layer_cls = generate_answer_layer_cls or GenerateAnswerLayer

        self._last_warmup_result: ApiQueryWarmupResult | None = None

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def last_warmup_result(self) -> ApiQueryWarmupResult | None:
        return self._last_warmup_result

    def warmup(self) -> ApiQueryWarmupResult:
        """Warm service components according to config-driven warmup policy."""
        started_perf = perf_counter()
        started_at = datetime.now(UTC).isoformat()

        warmup_cfg = self._settings.api.warmup
        components: dict[str, dict[str, Any]] = {
            "retrieval": {"status": "skipped"},
            "reranker": {"status": "skipped"},
            "context": {"status": "skipped"},
            "generation": {"status": "skipped"},
        }
        overall_error: dict[str, Any] | None = None

        if warmup_cfg.retrieval_enabled:
            started = perf_counter()
            try:
                self._retriever, payload = self._retrieve_layer_cls.warmup(
                    config_path=self._config_path,
                    retriever=self._retriever,
                )
                components["retrieval"] = {
                    "status": STATUS_SUCCESS
                    if payload.get("status") != STATUS_FAILED
                    else STATUS_FAILED,
                    "duration_ms": max(0, int((perf_counter() - started) * 1000)),
                    "retrieval_warmup_ok": payload.get("status") != STATUS_FAILED,
                    **payload,
                }
            except Exception as err:
                components["retrieval"] = {
                    "status": STATUS_FAILED,
                    "duration_ms": max(0, int((perf_counter() - started) * 1000)),
                    "error": str(err),
                }

        if warmup_cfg.reranker_enabled:
            started = perf_counter()
            try:
                self._reranker, payload = self._rerank_layer_cls.warmup(
                    config_path=self._config_path,
                    reranker=self._reranker,
                )
                components["reranker"] = {
                    "status": STATUS_SUCCESS
                    if payload.get("status") != STATUS_FAILED
                    else STATUS_FAILED,
                    "duration_ms": max(0, int((perf_counter() - started) * 1000)),
                    "reranker_warmup_ok": payload.get("status") != STATUS_FAILED,
                    **payload,
                }
            except Exception as err:
                components["reranker"] = {
                    "status": STATUS_FAILED,
                    "duration_ms": max(0, int((perf_counter() - started) * 1000)),
                    "error": str(err),
                }

        if warmup_cfg.context_enabled:
            started = perf_counter()
            try:
                self._context_builder, payload = self._build_context_layer_cls.warmup(
                    config_path=self._config_path,
                    context_builder=self._context_builder,
                )
                components["context"] = {
                    "status": STATUS_SUCCESS
                    if payload.get("status") != STATUS_FAILED
                    else STATUS_FAILED,
                    "duration_ms": max(0, int((perf_counter() - started) * 1000)),
                    "context_warmup_ok": payload.get("status") != STATUS_FAILED,
                    **payload,
                }
            except Exception as err:
                components["context"] = {
                    "status": STATUS_FAILED,
                    "duration_ms": max(0, int((perf_counter() - started) * 1000)),
                    "error": str(err),
                }

        if warmup_cfg.generation_enabled:
            started = perf_counter()
            try:
                self._answer_generator, payload = self._generate_answer_layer_cls.warmup(
                    config_path=self._config_path,
                    answer_generator=self._answer_generator,
                )
                components["generation"] = {
                    "status": STATUS_SUCCESS
                    if payload.get("status") != STATUS_FAILED
                    else STATUS_FAILED,
                    "duration_ms": max(0, int((perf_counter() - started) * 1000)),
                    "generation_warmup_ok": payload.get("status") != STATUS_FAILED,
                    **payload,
                }
            except Exception as err:
                components["generation"] = {
                    "status": STATUS_FAILED,
                    "duration_ms": max(0, int((perf_counter() - started) * 1000)),
                    "error": str(err),
                }

        if any(item.get("status") == STATUS_FAILED for item in components.values()):
            status = STATUS_FAILED
            overall_error = {"code": "warmup_failed", "message": "One or more warmup steps failed."}
        else:
            status = STATUS_SUCCESS

        result = ApiQueryWarmupResult(
            status=status,
            started_at=started_at,
            finished_at=datetime.now(UTC).isoformat(),
            duration_ms=max(0, int((perf_counter() - started_perf) * 1000)),
            components=components,
            error=overall_error,
        )

        self._last_warmup_result = result

        return result

    def query(
        self,
        *,
        question: str,
        filters: dict[str, Any] | None = None,
        retrieve_top_k: int | None = None,
        rerank_top_k_before: int | None = None,
        rerank_top_k_after: int | None = None,
        context_token_budget: int | None = None,
        context_max_chunks: int | None = None,
        write_trace: bool | None = None,
        include_debug: bool = False,
    ) -> ApiQueryServiceResult:
        """Run query flow over online layers and return API-ready payload."""

        if not question.strip():
            raise ValueError("question must not be empty.")

        self._validate_positive_int(retrieve_top_k, "retrieve_top_k")
        self._validate_positive_int(rerank_top_k_before, "rerank_top_k_before")
        self._validate_positive_int(rerank_top_k_after, "rerank_top_k_after")
        self._validate_positive_int(context_token_budget, "context_token_budget")
        self._validate_positive_int(context_max_chunks, "context_max_chunks")

        if (
            rerank_top_k_before is not None
            and rerank_top_k_after is not None
            and rerank_top_k_after > rerank_top_k_before
        ):
            raise ValueError("rerank_top_k_after must be <= rerank_top_k_before.")

        run_id = make_run_id(STAGE_ONLINE_QUERY)
        started_perf = perf_counter()
        normalized_filters = self._normalize_filters(filters)
        traces_enabled = (
            self._settings.online_query.write_trace if write_trace is None else bool(write_trace)
        )
        trace_dir = self._settings.online_query.trace_dir
        trace_path = trace_dir / f"{run_id}_api_query.json"

        retrieve_trace_path: Path | None = None
        rerank_trace_path: Path | None = None
        context_trace_path: Path | None = None
        generation_trace_path: Path | None = None

        try:
            retrieve_layer = self._retrieve_layer_cls(
                question=question,
                config_path=self._config_path,
                top_k=retrieve_top_k,
                filters=normalized_filters,
                write_trace=traces_enabled,
                retriever=self._retriever,
            )
            retrieve_result = retrieve_layer.run()
            self._retriever = getattr(retrieve_layer, "retriever", self._retriever)
            retrieve_trace_path = retrieve_result.trace_path

            rerank_layer = self._rerank_layer_cls(
                question=question,
                candidates=retrieve_result.results,
                config_path=self._config_path,
                top_k_before=rerank_top_k_before,
                top_k_after=rerank_top_k_after,
                write_trace=traces_enabled,
                reranker=self._reranker,
            )
            rerank_result = rerank_layer.run()
            self._reranker = getattr(rerank_layer, "reranker", self._reranker)
            rerank_trace_path = rerank_result.trace_path

            context_layer = self._build_context_layer_cls(
                question=question,
                results=rerank_result.results,
                config_path=self._config_path,
                token_budget=context_token_budget,
                max_chunks=context_max_chunks,
                write_trace=traces_enabled,
                context_builder=self._context_builder,
            )
            context_result = context_layer.run()
            self._context_builder = getattr(context_layer, "context_builder", self._context_builder)
            context_trace_path = context_result.trace_path

            generation_layer = self._generate_answer_layer_cls(
                question=question,
                context_bundle=context_result.context_bundle,
                config_path=self._config_path,
                write_trace=traces_enabled,
                answer_generator=self._answer_generator,
            )
            generation_result = generation_layer.run()
            self._answer_generator = getattr(
                generation_layer, "answer_generator", self._answer_generator
            )
            generation_trace_path = generation_result.trace_path

        except Exception as err:
            logger.exception("API query pipeline failed: run_id=%s", run_id)

            failed_result = ApiQueryServiceResult(
                run_id=run_id,
                question=question.strip(),
                answer="",
                confidence="none",
                sources=[],
                sources_total=0,
                retrieve_results_total=0,
                reranked_results_total=0,
                context_token_count=0,
                context_sources_total=0,
                context_truncated=False,
                latency_ms={
                    "total": max(0, int((perf_counter() - started_perf) * 1000)),
                    "retrieve": 0,
                    "rerank": 0,
                    "build_context": 0,
                    "generate_answer": 0,
                },
                trace_paths={
                    "retrieve": str(retrieve_trace_path) if retrieve_trace_path else None,
                    "rerank": str(rerank_trace_path) if rerank_trace_path else None,
                    "build_context": str(context_trace_path) if context_trace_path else None,
                    "generate_answer": str(generation_trace_path)
                    if generation_trace_path
                    else None,
                    "api_query": str(trace_path) if traces_enabled else None,
                },
                debug=None,
                status=STATUS_FAILED,
                error={"code": "query_failed", "message": str(err)},
            )
            if traces_enabled:
                self._write_top_level_trace(
                    path=trace_path, result=failed_result, filters=normalized_filters
                )
            raise RuntimeError("ApiQueryService query failed.") from err

        latency_ms = {
            "retrieve": retrieve_result.duration_ms,
            "rerank": rerank_result.duration_ms,
            "build_context": context_result.duration_ms,
            "generate_answer": generation_result.duration_ms,
            "total": max(0, int((perf_counter() - started_perf) * 1000)),
        }
        trace_paths = {
            "retrieve": str(retrieve_trace_path) if retrieve_trace_path else None,
            "rerank": str(rerank_trace_path) if rerank_trace_path else None,
            "build_context": str(context_trace_path) if context_trace_path else None,
            "generate_answer": str(generation_trace_path) if generation_trace_path else None,
            "api_query": str(trace_path) if traces_enabled else None,
        }
        stage_counts: dict[str, int | bool] = {
            "retrieve_results_total": retrieve_result.results_total,
            "reranked_results_total": rerank_result.reranked_results_total,
            "context_sources_total": context_result.sources_total,
            "context_token_count": context_result.token_count,
            "context_truncated": context_result.truncated,
        }
        debug_payload = (
            {
                "stage_counts": stage_counts,
                "latency_ms": latency_ms,
                "trace_paths": trace_paths,
            }
            if include_debug
            else None
        )

        result = ApiQueryServiceResult(
            run_id=run_id,
            question=question.strip(),
            answer=generation_result.generated_answer.answer,
            confidence=generation_result.confidence,
            sources=generation_result.generated_answer.sources,
            sources_total=generation_result.sources_total,
            retrieve_results_total=retrieve_result.results_total,
            reranked_results_total=rerank_result.reranked_results_total,
            context_token_count=context_result.token_count,
            context_sources_total=context_result.sources_total,
            context_truncated=context_result.truncated,
            latency_ms=latency_ms,
            trace_paths=trace_paths,
            debug=debug_payload,
            status=STATUS_SUCCESS,
            error=None,
        )

        if traces_enabled:
            self._write_top_level_trace(path=trace_path, result=result, filters=normalized_filters)

        return result

    def shutdown(self) -> None:
        """Release long-lived resources for process shutdown."""

        logger.info("Shutting down ApiQueryService resources.")

        if self._owns_answer_generator:
            self._close_object(getattr(self._answer_generator, "_llm_client", None))
        self._close_object(getattr(self._retriever, "_vector_store", None))

        self._close_object(self._retriever)
        self._close_object(self._answer_generator)

    @staticmethod
    def _validate_positive_int(value: int | None, field_name: str) -> None:
        if value is not None and value <= 0:
            raise ValueError(f"{field_name} must be > 0 when provided.")

    @staticmethod
    def _normalize_filters(filters: dict[str, Any] | None) -> dict[str, Any] | None:
        if not filters:
            return None

        normalized: dict[str, Any] = {}

        for key, value in filters.items():
            if isinstance(value, str):
                trimmed = value.strip()
                if trimmed:
                    normalized[key] = trimmed
            elif value is not None:
                normalized[key] = value

        return normalized or None

    @staticmethod
    def _close_object(obj: Any) -> None:
        close_method = getattr(obj, "close", None)
        if callable(close_method):
            close_method()

    def _write_top_level_trace(
        self,
        *,
        path: Path,
        result: ApiQueryServiceResult,
        filters: dict[str, Any] | None,
    ) -> None:
        payload = {
            "run_id": result.run_id,
            "question": result.question,
            "filters": filters,
            "status": result.status,
            "confidence": result.confidence,
            "sources_total": result.sources_total,
            "retrieve_results_total": result.retrieve_results_total,
            "reranked_results_total": result.reranked_results_total,
            "context_token_count": result.context_token_count,
            "context_sources_total": result.context_sources_total,
            "context_truncated": result.context_truncated,
            "latency_ms": result.latency_ms,
            "trace_paths": result.trace_paths,
            "error": result.error,
            "created_at": datetime.now(UTC).isoformat(),
        }

        write_json_payload(path, payload)
