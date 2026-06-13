"""Core eval runner for deterministic case execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.application_layers.online.build_context import BuildContextLayer
from app.application_layers.online.generate_answer import GenerateAnswerLayer
from app.application_layers.online.rerank import RerankLayer
from app.application_layers.online.retrieve import RetrieveLayer
import app.evaluation.citation_metrics as citation_metrics
import app.evaluation.confidence_metrics as confidence_metrics
import app.evaluation.context_metrics as context_metrics
import app.evaluation.generation_metrics as generation_metrics
import app.evaluation.rerank_metrics as rerank_metrics
import app.evaluation.retrieval_metrics as retrieval_metrics
import app.evaluation.robustness_metrics as robustness_metrics
from app.evaluation.judges import create_judge
from app.evaluation.records import (
    CitationEvalRecord,
    ContextEvalRecord,
    EvalCaseResult,
    EvalRunnerOptions,
    EvalSample,
    GenerationEvalRecord,
    JudgeRecord,
    RerankEvalRecord,
    RetrievalEvalRecord,
)
from app.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

_SUPPORTED_MODES = {
    "retrieval",
    "rerank",
    "context",
    "generation",
    "full",
    "citation",
    "robustness",
}


def _options_or_default(options: EvalRunnerOptions | None) -> EvalRunnerOptions:
    return options or EvalRunnerOptions()


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}.{key}": float(value) for key, value in metrics.items()}


def _safe_trace_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return str(value)


def _source_url(source: Any) -> str | None:
    url = getattr(source, "url", None)
    if isinstance(url, str) and url.strip():
        return url.strip()
    return None


def _extract_result_ids(results: list[Any]) -> tuple[list[str], list[str], list[str], list[float]]:
    chunk_ids: list[str] = []
    document_ids: list[str] = []
    urls: list[str] = []
    scores: list[float] = []
    for item in results:
        chunk_id = getattr(item, "chunk_id", None)
        if isinstance(chunk_id, str) and chunk_id.strip():
            chunk_ids.append(chunk_id.strip())
        document_id = getattr(item, "document_id", None)
        if isinstance(document_id, str) and document_id.strip():
            document_ids.append(document_id.strip())
        source = getattr(item, "source", None)
        url = _source_url(source)
        if url is not None:
            urls.append(url)
        score = getattr(item, "final_score", None)
        if isinstance(score, (int, float)):
            scores.append(float(score))
    return chunk_ids, document_ids, urls, scores


def _extract_generation_source_ids(sources: list[Any]) -> tuple[list[str], list[str], list[str]]:
    chunk_ids: list[str] = []
    document_ids: list[str] = []
    urls: list[str] = []
    for source in sources:
        chunk_id = getattr(source, "chunk_id", None)
        if isinstance(chunk_id, str) and chunk_id.strip():
            chunk_ids.append(chunk_id.strip())
        document_id = getattr(source, "document_id", None)
        if isinstance(document_id, str) and document_id.strip():
            document_ids.append(document_id.strip())
        url = _source_url(source)
        if url is not None:
            urls.append(url)
    return chunk_ids, document_ids, urls


def _context_texts_from_bundle(context_bundle: Any) -> list[str]:
    chunks = getattr(context_bundle, "chunks", [])
    texts: list[str] = []
    if isinstance(chunks, list):
        for chunk in chunks:
            text = getattr(chunk, "text", None)
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    rendered_context = getattr(context_bundle, "rendered_context", None)
    if not texts and isinstance(rendered_context, str) and rendered_context.strip():
        texts.append(rendered_context.strip())
    return texts


def _base_case_result(sample: EvalSample) -> EvalCaseResult:
    return EvalCaseResult(
        sample_id=sample.id,
        question=sample.question,
        subset=sample.subset.value,
        type=sample.type.value,
        difficulty=sample.difficulty.value,
        expected_behavior=sample.expected_behavior.value,
        expected_chunk_ids=list(sample.expected_chunk_ids),
        expected_document_ids=list(sample.expected_document_ids),
        expected_urls=list(sample.expected_urls),
        reference_answer=sample.reference_answer,
    )


def _error_result(sample: EvalSample, err: Exception) -> EvalCaseResult:
    result = _base_case_result(sample)
    result.error = f"{type(err).__name__}: {err}"
    result.passed = False
    return result


def _run_retrieve_stage(
    sample: EvalSample,
    *,
    options: EvalRunnerOptions,
    retrieve_layer_cls: Any,
) -> tuple[RetrievalEvalRecord, list[Any]]:
    layer = retrieve_layer_cls(
        question=sample.question,
        config_path=options.config_path,
        top_k=options.retrieve_top_k,
        write_trace=options.write_trace,
    )
    output = layer.run()
    results = list(getattr(output, "results", []))
    chunk_ids, document_ids, urls, scores = _extract_result_ids(results)
    record = RetrievalEvalRecord(
        sample_id=sample.id,
        query=sample.question,
        retrieved_chunk_ids=chunk_ids,
        retrieved_document_ids=document_ids,
        retrieved_urls=urls,
        retrieved_scores=scores,
        results_total=int(getattr(output, "results_total", len(results))),
        top_k=getattr(output, "top_k", options.retrieve_top_k),
        strategy=getattr(output, "strategy", None),
        duration_ms=float(getattr(output, "duration_ms", 0.0)),
        trace_path=_safe_trace_path(getattr(output, "trace_path", None)),
    )
    return record, results


def _run_rerank_stage(
    sample: EvalSample,
    *,
    options: EvalRunnerOptions,
    retrieval_record: RetrievalEvalRecord,
    rerank_layer_cls: Any,
    retrieve_results: list[Any],
) -> tuple[RerankEvalRecord, list[Any]]:
    layer = rerank_layer_cls(
        question=sample.question,
        candidates=retrieve_results,
        config_path=options.config_path,
        top_k_before=options.rerank_top_k_before,
        top_k_after=options.rerank_top_k_after,
        write_trace=options.write_trace,
    )
    output = layer.run()
    results = list(getattr(output, "results", []))
    chunk_ids, document_ids, urls, _scores = _extract_result_ids(results)
    record = RerankEvalRecord(
        sample_id=sample.id,
        reranked_chunk_ids=chunk_ids,
        reranked_document_ids=document_ids,
        reranked_urls=urls,
        input_results_total=int(
            getattr(output, "input_results_total", retrieval_record.results_total)
        ),
        reranked_results_total=int(getattr(output, "reranked_results_total", len(results))),
        model_name=getattr(output, "model_name", None),
        top_k_before=getattr(output, "top_k_before", options.rerank_top_k_before),
        top_k_after=getattr(output, "top_k_after", options.rerank_top_k_after),
        latency_budget_exceeded=bool(getattr(output, "latency_budget_exceeded", False)),
        cache_hits=int(getattr(output, "cache_hits", 0)),
        cache_misses=int(getattr(output, "cache_misses", 0)),
        duration_ms=float(getattr(output, "duration_ms", 0.0)),
        trace_path=_safe_trace_path(getattr(output, "trace_path", None)),
    )
    return record, results


def _run_context_stage(
    sample: EvalSample,
    *,
    options: EvalRunnerOptions,
    rerank_results: list[Any],
    build_context_layer_cls: Any,
) -> tuple[ContextEvalRecord, Any]:
    layer = build_context_layer_cls(
        question=sample.question,
        results=rerank_results,
        config_path=options.config_path,
        token_budget=options.context_token_budget,
        max_chunks=options.context_max_chunks,
        write_trace=options.write_trace,
    )
    output = layer.run()
    context_bundle = getattr(output, "context_bundle")
    context_chunks = list(getattr(context_bundle, "chunks", []))
    context_sources = list(getattr(context_bundle, "sources", []))
    chunk_ids, document_ids, urls, _scores = _extract_result_ids(context_chunks)
    if not urls:
        _, _, urls = _extract_generation_source_ids(context_sources)
    record = ContextEvalRecord(
        sample_id=sample.id,
        context_chunk_ids=chunk_ids,
        context_document_ids=document_ids,
        context_urls=urls,
        token_count=int(getattr(context_bundle, "token_count", getattr(output, "token_count", 0))),
        token_budget=getattr(context_bundle, "token_budget", getattr(output, "token_budget", None)),
        sources_total=int(getattr(output, "sources_total", len(context_sources))),
        truncated=bool(getattr(context_bundle, "truncated", getattr(output, "truncated", False))),
        included_chunks_total=int(
            getattr(output, "included_chunks_total", len(context_chunks))
        ),
        dropped_chunks_total=int(getattr(output, "dropped_chunks_total", 0)),
        duration_ms=float(getattr(output, "duration_ms", 0.0)),
        trace_path=_safe_trace_path(getattr(output, "trace_path", None)),
    )
    return record, context_bundle


def _run_generation_stage(
    sample: EvalSample,
    *,
    options: EvalRunnerOptions,
    context_bundle: Any,
    generate_answer_layer_cls: Any,
) -> GenerationEvalRecord:
    layer = generate_answer_layer_cls(
        question=sample.question,
        context_bundle=context_bundle,
        config_path=options.config_path,
        write_trace=options.write_trace,
    )
    output = layer.run()
    generated_answer = getattr(output, "generated_answer")
    sources = list(getattr(generated_answer, "sources", []))
    cited_chunk_ids, cited_document_ids, cited_urls = _extract_generation_source_ids(sources)
    return GenerationEvalRecord(
        sample_id=sample.id,
        answer=str(getattr(generated_answer, "answer", "")),
        confidence=str(getattr(output, "confidence", getattr(generated_answer, "confidence", "none"))),
        sources_total=int(getattr(output, "sources_total", len(sources))),
        cited_chunk_ids=cited_chunk_ids,
        cited_document_ids=cited_document_ids,
        cited_urls=cited_urls,
        parsed_successfully=bool(
            getattr(output, "parsed_successfully", getattr(generated_answer, "parsed_successfully", False))
        ),
        provider=getattr(output, "provider", None),
        model_name=getattr(output, "model_name", None),
        duration_ms=float(getattr(output, "duration_ms", 0.0)),
        trace_path=_safe_trace_path(getattr(output, "trace_path", None)),
    )


def _build_citation_record(
    sample: EvalSample,
    generation_record: GenerationEvalRecord,
    context_record: ContextEvalRecord,
) -> CitationEvalRecord:
    context_set = set(context_record.context_chunk_ids)
    valid_ids = [value for value in generation_record.cited_chunk_ids if value in context_set]
    invented_ids = [value for value in generation_record.cited_chunk_ids if value not in context_set]
    return CitationEvalRecord(
        sample_id=sample.id,
        cited_chunk_ids=list(generation_record.cited_chunk_ids),
        context_chunk_ids=list(context_record.context_chunk_ids),
        expected_chunk_ids=list(sample.expected_chunk_ids),
        valid_cited_chunk_ids=valid_ids,
        invented_cited_chunk_ids=invented_ids,
        cited_document_ids=list(generation_record.cited_document_ids),
        cited_urls=list(generation_record.cited_urls),
    )


def _build_judge_record(
    *,
    sample: EvalSample,
    context_bundle: Any,
    generation_record: GenerationEvalRecord,
    judge: Any,
) -> JudgeRecord:
    context_texts = _context_texts_from_bundle(context_bundle)
    sources = getattr(context_bundle, "sources", None)
    if not isinstance(sources, list):
        sources = []
    return judge.judge(
        sample_id=sample.id,
        question=sample.question,
        answer=generation_record.answer,
        context_texts=context_texts,
        sources=sources,
        reference_answer=sample.reference_answer,
    )


def _append_metrics_for_retrieval(result: EvalCaseResult) -> None:
    if result.retrieval is None:
        return
    values = retrieval_metrics.build_retrieval_metrics(
        retrieved_chunk_ids=result.retrieval.retrieved_chunk_ids,
        retrieved_document_ids=result.retrieval.retrieved_document_ids,
        retrieved_urls=result.retrieval.retrieved_urls,
        expected_chunk_ids=result.expected_chunk_ids,
        expected_document_ids=result.expected_document_ids,
        expected_urls=result.expected_urls,
    )
    result.metrics.update(_prefix_metrics("retrieval", values))


def _append_metrics_for_rerank(result: EvalCaseResult) -> None:
    if result.retrieval is None or result.rerank is None:
        return
    values = rerank_metrics.build_rerank_metrics(
        retrieved_chunk_ids_before=result.retrieval.retrieved_chunk_ids,
        reranked_chunk_ids_after=result.rerank.reranked_chunk_ids,
        expected_chunk_ids=result.expected_chunk_ids,
        latency_budget_exceeded=result.rerank.latency_budget_exceeded,
        cache_hits=result.rerank.cache_hits,
        cache_misses=result.rerank.cache_misses,
    )
    result.metrics.update(_prefix_metrics("rerank", values))


def _append_metrics_for_context(result: EvalCaseResult) -> None:
    if result.context is None:
        return
    reranked_ids: list[str] | None = None
    if result.rerank is not None:
        reranked_ids = result.rerank.reranked_chunk_ids
    values = context_metrics.build_context_metrics(
        context_chunk_ids=result.context.context_chunk_ids,
        context_document_ids=result.context.context_document_ids,
        context_urls=result.context.context_urls,
        expected_chunk_ids=result.expected_chunk_ids,
        expected_document_ids=result.expected_document_ids,
        expected_urls=result.expected_urls,
        reranked_chunk_ids=reranked_ids,
        token_count=result.context.token_count,
        token_budget=result.context.token_budget,
        truncated=result.context.truncated,
    )
    result.metrics.update(_prefix_metrics("context", values))


def _append_metrics_for_generation(result: EvalCaseResult) -> None:
    if result.generation is None:
        return
    values = generation_metrics.build_generation_metrics(
        answer=result.generation.answer,
        confidence=result.generation.confidence,
        parsed_successfully=result.generation.parsed_successfully,
        reference_answer=result.reference_answer,
    )
    result.metrics.update(_prefix_metrics("generation", values))


def _append_metrics_for_citation(result: EvalCaseResult) -> None:
    if result.generation is None or result.context is None:
        return
    values = citation_metrics.build_citation_metrics(
        cited_chunk_ids=result.generation.cited_chunk_ids,
        context_chunk_ids=result.context.context_chunk_ids,
        expected_chunk_ids=result.expected_chunk_ids,
        answer=result.generation.answer,
        confidence=result.generation.confidence,
    )
    result.metrics.update(_prefix_metrics("citation", values))


def _append_metrics_for_confidence(result: EvalCaseResult) -> None:
    if result.generation is None:
        return
    context_tokens = result.context.token_count if result.context is not None else 0
    values = confidence_metrics.build_confidence_metrics(
        confidence=result.generation.confidence,
        sources_total=result.generation.sources_total,
        context_token_count=context_tokens,
        expected_behavior=result.expected_behavior,
    )
    result.metrics.update(_prefix_metrics("confidence", values))


def _append_metrics_for_robustness(result: EvalCaseResult) -> None:
    if result.generation is None:
        return
    invented_rate = result.metrics.get("citation.invented_source_rate", 0.0)
    valid_sources = (invented_rate == 0.0) and (result.generation.sources_total > 0)
    if result.generation.confidence.strip().lower() == "none":
        valid_sources = True
    values = robustness_metrics.build_robustness_metrics(
        sample_type=result.type,
        confidence=result.generation.confidence,
        valid_sources=valid_sources,
    )
    result.metrics.update(_prefix_metrics("robustness", values))


def _append_judge_metrics(result: EvalCaseResult) -> None:
    if result.judge is None:
        return
    result.metrics.update(
        {
            "judge.groundedness_score": result.judge.groundedness_score,
            "judge.relevance_score": result.judge.relevance_score,
            "judge.source_support_score": result.judge.source_support_score,
            "judge.completeness_score": result.judge.completeness_score,
            "judge.hallucination_risk": result.judge.hallucination_risk,
        }
    )


def _mark_passed(result: EvalCaseResult) -> None:
    if result.error is not None:
        result.passed = False
        return
    if result.judge is not None:
        result.passed = result.judge.verdict in {"pass", "abstain", "warn"}
        return
    result.passed = True


def _run_case_internal(
    sample: EvalSample,
    *,
    mode: str,
    options: EvalRunnerOptions | None = None,
    retrieve_layer_cls: Any | None = None,
    rerank_layer_cls: Any | None = None,
    build_context_layer_cls: Any | None = None,
    generate_answer_layer_cls: Any | None = None,
    judge: Any | None = None,
) -> EvalCaseResult:
    config = _options_or_default(options)
    if mode not in _SUPPORTED_MODES:
        raise ValueError(f"Unsupported eval mode: {mode!r}")

    retrieve_cls = retrieve_layer_cls or RetrieveLayer
    rerank_cls = rerank_layer_cls or RerankLayer
    context_cls = build_context_layer_cls or BuildContextLayer
    generation_cls = generate_answer_layer_cls or GenerateAnswerLayer

    result = _base_case_result(sample)
    setup_logging()
    try:
        retrieval_stage, retrieve_results = _run_retrieve_stage(
            sample, options=config, retrieve_layer_cls=retrieve_cls
        )
        result.retrieval = retrieval_stage
        result.trace_paths["retrieve"] = retrieval_stage.trace_path
        result.latency_ms["retrieve"] = retrieval_stage.duration_ms
        _append_metrics_for_retrieval(result)

        if mode == "retrieval":
            _mark_passed(result)
            return result

        rerank_stage, reranked_results = _run_rerank_stage(
            sample,
            options=config,
            retrieval_record=retrieval_stage,
            rerank_layer_cls=rerank_cls,
            retrieve_results=retrieve_results,
        )
        result.rerank = rerank_stage
        result.trace_paths["rerank"] = rerank_stage.trace_path
        result.latency_ms["rerank"] = rerank_stage.duration_ms
        _append_metrics_for_rerank(result)

        if mode == "rerank":
            _mark_passed(result)
            return result

        context_stage, context_bundle = _run_context_stage(
            sample,
            options=config,
            rerank_results=reranked_results,
            build_context_layer_cls=context_cls,
        )
        result.context = context_stage
        result.trace_paths["context"] = context_stage.trace_path
        result.latency_ms["context"] = context_stage.duration_ms
        _append_metrics_for_context(result)

        if mode == "context":
            _mark_passed(result)
            return result

        generation_stage = _run_generation_stage(
            sample,
            options=config,
            context_bundle=context_bundle,
            generate_answer_layer_cls=generation_cls,
        )
        result.generation = generation_stage
        result.trace_paths["generation"] = generation_stage.trace_path
        result.latency_ms["generation"] = generation_stage.duration_ms
        result.latency_ms["total"] = (
            result.latency_ms.get("retrieve", 0.0)
            + result.latency_ms.get("rerank", 0.0)
            + result.latency_ms.get("context", 0.0)
            + result.latency_ms.get("generation", 0.0)
        )

        result.citation = _build_citation_record(sample, generation_stage, context_stage)
        _append_metrics_for_generation(result)
        _append_metrics_for_citation(result)
        _append_metrics_for_confidence(result)
        _append_metrics_for_robustness(result)

        if config.judge_enabled:
            resolved_judge = judge or create_judge(backend=config.judge_backend)
            if resolved_judge is not None:
                result.judge = _build_judge_record(
                    sample=sample,
                    context_bundle=context_bundle,
                    generation_record=generation_stage,
                    judge=resolved_judge,
                )
                _append_judge_metrics(result)

        _mark_passed(result)
        return result
    except Exception as err:
        logger.warning(
            "Eval case failed: sample_id=%s mode=%s error_type=%s",
            sample.id,
            mode,
            type(err).__name__,
        )
        if config.fail_fast:
            raise
        return _error_result(sample, err)


def run_retrieval_case(
    sample: EvalSample,
    *,
    options: EvalRunnerOptions | None = None,
    retrieve_layer_cls: Any | None = None,
) -> EvalCaseResult:
    return _run_case_internal(
        sample,
        mode="retrieval",
        options=options,
        retrieve_layer_cls=retrieve_layer_cls,
    )


def run_rerank_case(
    sample: EvalSample,
    *,
    options: EvalRunnerOptions | None = None,
    retrieve_layer_cls: Any | None = None,
    rerank_layer_cls: Any | None = None,
) -> EvalCaseResult:
    return _run_case_internal(
        sample,
        mode="rerank",
        options=options,
        retrieve_layer_cls=retrieve_layer_cls,
        rerank_layer_cls=rerank_layer_cls,
    )


def run_context_case(
    sample: EvalSample,
    *,
    options: EvalRunnerOptions | None = None,
    retrieve_layer_cls: Any | None = None,
    rerank_layer_cls: Any | None = None,
    build_context_layer_cls: Any | None = None,
) -> EvalCaseResult:
    return _run_case_internal(
        sample,
        mode="context",
        options=options,
        retrieve_layer_cls=retrieve_layer_cls,
        rerank_layer_cls=rerank_layer_cls,
        build_context_layer_cls=build_context_layer_cls,
    )


def run_generation_case(
    sample: EvalSample,
    *,
    options: EvalRunnerOptions | None = None,
    retrieve_layer_cls: Any | None = None,
    rerank_layer_cls: Any | None = None,
    build_context_layer_cls: Any | None = None,
    generate_answer_layer_cls: Any | None = None,
    judge: Any | None = None,
) -> EvalCaseResult:
    return _run_case_internal(
        sample,
        mode="generation",
        options=options,
        retrieve_layer_cls=retrieve_layer_cls,
        rerank_layer_cls=rerank_layer_cls,
        build_context_layer_cls=build_context_layer_cls,
        generate_answer_layer_cls=generate_answer_layer_cls,
        judge=judge,
    )


def run_full_case(
    sample: EvalSample,
    *,
    options: EvalRunnerOptions | None = None,
    retrieve_layer_cls: Any | None = None,
    rerank_layer_cls: Any | None = None,
    build_context_layer_cls: Any | None = None,
    generate_answer_layer_cls: Any | None = None,
    judge: Any | None = None,
) -> EvalCaseResult:
    return _run_case_internal(
        sample,
        mode="full",
        options=options,
        retrieve_layer_cls=retrieve_layer_cls,
        rerank_layer_cls=rerank_layer_cls,
        build_context_layer_cls=build_context_layer_cls,
        generate_answer_layer_cls=generate_answer_layer_cls,
        judge=judge,
    )


def run_eval_batch(
    samples: list[EvalSample],
    *,
    mode: str = "full",
    options: EvalRunnerOptions | None = None,
    retrieve_layer_cls: Any | None = None,
    rerank_layer_cls: Any | None = None,
    build_context_layer_cls: Any | None = None,
    generate_answer_layer_cls: Any | None = None,
    judge: Any | None = None,
    progress_log_interval: int = 10,
) -> tuple[list[EvalCaseResult], list[dict[str, Any]]]:
    """Run eval cases sequentially and return results with error rows."""
    if mode not in _SUPPORTED_MODES:
        raise ValueError(f"Unsupported eval mode: {mode!r}")
    if progress_log_interval <= 0:
        raise ValueError("progress_log_interval must be > 0.")

    config = _options_or_default(options)
    setup_logging()
    logger.info("Starting eval batch: mode=%s samples_total=%s", mode, len(samples))
    results: list[EvalCaseResult] = []
    errors: list[dict[str, Any]] = []

    for index, sample in enumerate(samples, start=1):
        if mode == "retrieval":
            result = run_retrieval_case(
                sample,
                options=config,
                retrieve_layer_cls=retrieve_layer_cls,
            )
        elif mode == "rerank":
            result = run_rerank_case(
                sample,
                options=config,
                retrieve_layer_cls=retrieve_layer_cls,
                rerank_layer_cls=rerank_layer_cls,
            )
        elif mode == "context":
            result = run_context_case(
                sample,
                options=config,
                retrieve_layer_cls=retrieve_layer_cls,
                rerank_layer_cls=rerank_layer_cls,
                build_context_layer_cls=build_context_layer_cls,
            )
        elif mode == "generation":
            result = run_generation_case(
                sample,
                options=config,
                retrieve_layer_cls=retrieve_layer_cls,
                rerank_layer_cls=rerank_layer_cls,
                build_context_layer_cls=build_context_layer_cls,
                generate_answer_layer_cls=generate_answer_layer_cls,
                judge=judge,
            )
        else:
            # full/citation/robustness share full stage execution.
            result = run_full_case(
                sample,
                options=config,
                retrieve_layer_cls=retrieve_layer_cls,
                rerank_layer_cls=rerank_layer_cls,
                build_context_layer_cls=build_context_layer_cls,
                generate_answer_layer_cls=generate_answer_layer_cls,
                judge=judge,
            )

        results.append(result)
        if result.error is not None:
            errors.append({"sample_id": sample.id, "error": result.error})
            logger.warning(
                "Eval batch case error: mode=%s sample_id=%s error=%s",
                mode,
                sample.id,
                result.error,
            )
            if config.fail_fast:
                raise RuntimeError(result.error)

        if (index % progress_log_interval) == 0 or index == len(samples):
            logger.info(
                "Eval batch progress: mode=%s processed=%s/%s errors=%s",
                mode,
                index,
                len(samples),
                len(errors),
            )

    logger.info(
        "Finished eval batch: mode=%s samples_total=%s errors_total=%s",
        mode,
        len(samples),
        len(errors),
    )
    return results, errors
