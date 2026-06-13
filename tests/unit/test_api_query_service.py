"""Unit tests for ApiQueryService behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from app.application.api_query_service import ApiQueryService
from app.domain.models.source import Source
from app.utils.config import Settings, load_settings


def _settings_with_trace_dir(tmp_path: Path) -> Settings:
    settings = load_settings()
    return settings.model_copy(
        deep=True,
        update={
            "online_query": settings.online_query.model_copy(
                update={"trace_dir": tmp_path, "write_trace": True}
            )
        },
    )


@dataclass
class _FakeRetrieveResult:
    results: list[Any]
    results_total: int
    duration_ms: int
    trace_path: Path | None


@dataclass
class _FakeRerankResult:
    results: list[Any]
    reranked_results_total: int
    duration_ms: int
    trace_path: Path | None


@dataclass
class _FakeContextResult:
    context_bundle: Any
    token_count: int
    sources_total: int
    truncated: bool
    duration_ms: int
    trace_path: Path | None


@dataclass
class _FakeGenerationResult:
    generated_answer: Any
    confidence: str
    sources_total: int
    duration_ms: int
    trace_path: Path | None


class _RetrieveLayer:
    called = False

    def __init__(self, **kwargs: Any) -> None:
        _RetrieveLayer.called = True
        self.kwargs = kwargs

    def run(self) -> _FakeRetrieveResult:
        return _FakeRetrieveResult(
            results=[SimpleNamespace(chunk_id="chunk_1")],
            results_total=1,
            duration_ms=7,
            trace_path=Path("retrieve.json"),
        )


class _RerankLayer:
    got_candidates: list[Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        _RerankLayer.got_candidates = kwargs["candidates"]

    def run(self) -> _FakeRerankResult:
        return _FakeRerankResult(
            results=[SimpleNamespace(chunk_id="chunk_1")],
            reranked_results_total=1,
            duration_ms=5,
            trace_path=Path("rerank.json"),
        )


class _ContextLayer:
    got_results: list[Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        _ContextLayer.got_results = kwargs["results"]

    def run(self) -> _FakeContextResult:
        context_bundle = SimpleNamespace(
            token_count=42,
            sources=[Source(title="Stripe", chunk_id="chunk_1", document_id="doc_1")],
            truncated=False,
        )
        return _FakeContextResult(
            context_bundle=context_bundle,
            token_count=42,
            sources_total=1,
            truncated=False,
            duration_ms=3,
            trace_path=Path("context.json"),
        )


class _GenerationLayer:
    got_context_bundle: Any = None

    def __init__(self, **kwargs: Any) -> None:
        _GenerationLayer.got_context_bundle = kwargs["context_bundle"]

    def run(self) -> _FakeGenerationResult:
        generated_answer = SimpleNamespace(
            answer="Answer text",
            sources=[Source(title="Stripe", chunk_id="chunk_1", document_id="doc_1")],
        )
        return _FakeGenerationResult(
            generated_answer=generated_answer,
            confidence="high",
            sources_total=1,
            duration_ms=9,
            trace_path=Path("generation.json"),
        )


@pytest.mark.unit
def test_query_runs_layers_and_passes_data_in_memory(tmp_path: Path) -> None:
    service = ApiQueryService(
        settings=_settings_with_trace_dir(tmp_path),
        retriever=object(),
        reranker=object(),
        context_builder=object(),
        answer_generator=object(),
        retrieve_layer_cls=_RetrieveLayer,
        rerank_layer_cls=_RerankLayer,
        build_context_layer_cls=_ContextLayer,
        generate_answer_layer_cls=_GenerationLayer,
    )
    result = service.query(question="hello", include_debug=True)

    assert _RetrieveLayer.called is True
    assert _RerankLayer.got_candidates is not None
    assert _ContextLayer.got_results is not None
    assert _GenerationLayer.got_context_bundle is not None
    assert result.answer == "Answer text"
    assert result.confidence == "high"
    assert result.sources_total == 1
    assert result.latency_ms["retrieve"] == 7
    assert result.trace_paths["retrieve"] == "retrieve.json"


class _EmptyRetrieveLayer:
    def __init__(self, **kwargs: Any) -> None:
        _ = kwargs

    def run(self) -> _FakeRetrieveResult:
        return _FakeRetrieveResult(results=[], results_total=0, duration_ms=1, trace_path=None)


class _PassThroughRerankLayer:
    def __init__(self, **kwargs: Any) -> None:
        assert kwargs["candidates"] == []

    def run(self) -> _FakeRerankResult:
        return _FakeRerankResult(results=[], reranked_results_total=0, duration_ms=1, trace_path=None)


class _EmptyContextLayer:
    def __init__(self, **kwargs: Any) -> None:
        assert kwargs["results"] == []

    def run(self) -> _FakeContextResult:
        context_bundle = SimpleNamespace(token_count=0, sources=[], truncated=False)
        return _FakeContextResult(
            context_bundle=context_bundle,
            token_count=0,
            sources_total=0,
            truncated=False,
            duration_ms=1,
            trace_path=None,
        )


class _NoAnswerGenerationLayer:
    def __init__(self, **kwargs: Any) -> None:
        _ = kwargs

    def run(self) -> _FakeGenerationResult:
        generated_answer = SimpleNamespace(answer="No answer", sources=[])
        return _FakeGenerationResult(
            generated_answer=generated_answer,
            confidence="none",
            sources_total=0,
            duration_ms=1,
            trace_path=None,
        )


@pytest.mark.unit
def test_empty_retrieval_still_returns_no_answer(tmp_path: Path) -> None:
    service = ApiQueryService(
        settings=_settings_with_trace_dir(tmp_path),
        retriever=object(),
        reranker=object(),
        context_builder=object(),
        answer_generator=object(),
        retrieve_layer_cls=_EmptyRetrieveLayer,
        rerank_layer_cls=_PassThroughRerankLayer,
        build_context_layer_cls=_EmptyContextLayer,
        generate_answer_layer_cls=_NoAnswerGenerationLayer,
    )
    result = service.query(question="missing")
    assert result.retrieve_results_total == 0
    assert result.reranked_results_total == 0
    assert result.confidence == "none"


class _FailingRetrieveLayer:
    def __init__(self, **kwargs: Any) -> None:
        _ = kwargs

    def run(self) -> _FakeRetrieveResult:
        raise RuntimeError("boom")


@pytest.mark.unit
def test_stage_failure_raises_clear_error(tmp_path: Path) -> None:
    service = ApiQueryService(
        settings=_settings_with_trace_dir(tmp_path),
        retriever=object(),
        reranker=object(),
        context_builder=object(),
        answer_generator=object(),
        retrieve_layer_cls=_FailingRetrieveLayer,
        rerank_layer_cls=_RerankLayer,
        build_context_layer_cls=_ContextLayer,
        generate_answer_layer_cls=_GenerationLayer,
    )
    with pytest.raises(RuntimeError, match="ApiQueryService query failed"):
        service.query(question="hello")


class _WarmRetriever:
    def __init__(self) -> None:
        self.embed_query_called = False

        def _embed_query(query: str) -> list[float]:
            _ = query
            self.embed_query_called = True
            return [0.0, 0.1]

        self._embedder = SimpleNamespace(embed_query=_embed_query)
        self._vector_store = SimpleNamespace(healthcheck=lambda: True, close=lambda: None)

    def warmup(self) -> dict[str, Any]:
        qdrant_ok = bool(self._vector_store.healthcheck())
        embed_ok: bool | None = None
        _ = self._embedder.embed_query("warmup query")
        embed_ok = True
        return {
            "status": "success",
            "qdrant_healthcheck_ok": qdrant_ok,
            "embed_query_warmup_ok": embed_ok,
            "tiny_search_warmup_ok": None,
        }


class _WarmReranker:
    def warmup(self) -> dict[str, Any]:
        return {"status": "success", "reranker_warmup_ok": True}


class _WarmAnswerGenerator:
    def __init__(self) -> None:
        self.warmup_generate_called = False

        def _warmup_generate(**kwargs: Any) -> bool:
            _ = kwargs
            self.warmup_generate_called = True
            return True

        self._llm_client = SimpleNamespace(
            healthcheck=lambda: True,
            warmup_generate=_warmup_generate,
            close=lambda: None,
        )

    def warmup(self) -> dict[str, Any]:
        health_ok = bool(self._llm_client.healthcheck())
        generate_ok: bool | None = None
        if health_ok is not False:
            generate_ok = bool(self._llm_client.warmup_generate())
        return {
            "status": "success",
            "ollama_healthcheck_ok": health_ok,
            "ollama_generate_warmup_ok": generate_ok,
        }


@pytest.mark.unit
def test_warmup_calls_component_warmups(tmp_path: Path) -> None:
    settings = _settings_with_trace_dir(tmp_path).model_copy(
        deep=True,
        update={
            "api": _settings_with_trace_dir(tmp_path).api.model_copy(
                update={
                    "warmup": _settings_with_trace_dir(tmp_path).api.warmup.model_copy(
                        update={"ollama_generate_enabled": True}
                    )
                }
            )
        },
    )
    warm_generator = _WarmAnswerGenerator()
    warm_retriever = _WarmRetriever()
    service = ApiQueryService(
        settings=settings,
        retriever=warm_retriever,
        reranker=_WarmReranker(),
        context_builder=object(),
        answer_generator=warm_generator,
    )
    warmup = service.warmup()
    assert warmup.status == "success"
    assert warmup.components["retrieval"]["status"] == "success"
    assert warmup.components["retrieval"]["embed_query_warmup_ok"] is True
    assert warmup.components["generation"]["status"] == "success"
    assert warmup.components["generation"]["ollama_generate_warmup_ok"] is True
    assert warm_retriever.embed_query_called is True
    assert warm_generator.warmup_generate_called is True


@pytest.mark.unit
def test_shutdown_calls_cleanup_functions(tmp_path: Path) -> None:
    closed = {"value": False}

    class _InjectedRetriever:
        def __init__(self) -> None:
            self._vector_store = SimpleNamespace(close=lambda: closed.__setitem__("value", True))

    service = ApiQueryService(
        settings=_settings_with_trace_dir(tmp_path),
        retriever=_InjectedRetriever(),
        reranker=object(),
        context_builder=object(),
        answer_generator=object(),
    )
    service.shutdown()
    assert closed["value"] is True


@pytest.mark.unit
def test_write_trace_false_disables_top_level_trace(tmp_path: Path) -> None:
    service = ApiQueryService(
        settings=_settings_with_trace_dir(tmp_path),
        retriever=object(),
        reranker=object(),
        context_builder=object(),
        answer_generator=object(),
        retrieve_layer_cls=_RetrieveLayer,
        rerank_layer_cls=_RerankLayer,
        build_context_layer_cls=_ContextLayer,
        generate_answer_layer_cls=_GenerationLayer,
    )
    result = service.query(question="hello", write_trace=False)
    assert result.trace_paths["api_query"] is None
