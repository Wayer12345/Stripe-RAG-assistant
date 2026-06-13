"""Integration tests for online rerank layer orchestration."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import app.application_layers.online.rerank as rerank_layer_module
import pytest
from app.application_layers.online.rerank import RerankLayer
from app.domain.models.retrieval_result import RetrievalResult
from app.domain.models.source import Source


def _write_config(config_path: Path, *, enabled: bool = True) -> None:
    config_path.write_text(
        f"""
app:
  name: "test-app"
  environment: "local"
  log_level: "INFO"
paths:
  raw_dir: "data/raw"
  interim_dir: "data/interim"
  processed_dir: "data/processed"
  indexes_dir: "data/indexes"
  manifests_dir: "data/manifests"
  eval_dir: "eval"
ingestion:
  input_dir: "data/raw"
  recursive: true
  supported_extensions: [".md"]
  outputs:
    parsed_documents_path: "data/interim/parsed_documents.jsonl"
    cleaned_documents_path: "data/interim/cleaned_documents.jsonl"
    manifest_path: "data/manifests/ingestion_manifest.json"
cleaning:
  mode: "conservative"
  boilerplate:
    phrases: ["Sign in"]
chunking: {{}}
embeddings: {{}}
vector_store: {{}}
indexing: {{}}
retrieval: {{}}
reranking:
  enabled: {str(enabled).lower()}
  provider: "cross_encoder"
  model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k_before: 4
  top_k_after: 2
  batch_size: 8
  max_query_chars: 512
  max_pair_chars: 1200
  warmup_enabled: true
  cache_enabled: false
  cache_path: "data/indexes/reranker_cache"
  latency_budget_ms: 50
  on_latency_budget_exceeded: "warn"
  write_trace: true
  trace_dir: "data/traces/queries"
  text_preview_chars: 12
generation: {{}}
eval: {{}}
""",
        encoding="utf-8",
    )


def _candidate(*, idx: int, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"chunk-{idx}",
        document_id=f"doc-{idx}",
        text=text,
        source=Source(
            title=f"title-{idx}",
            url=f"https://docs.example/{idx}",
            section="section",
            chunk_id=f"chunk-{idx}",
            document_id=f"doc-{idx}",
            support_score=0.9,
            source_type="markdown",
        ),
        retrieval_score=0.8,
        lexical_score=None,
        dense_score=0.8,
        reranker_score=None,
        final_score=0.8,
        metadata={"source_type": "markdown"},
    )


class _FakeReranker:
    def __init__(self) -> None:
        self.warmup_called = 0
        self.rerank_calls: list[dict[str, Any]] = []
        self._stats: dict[str, Any] = {
            "cache_hits": 1,
            "cache_misses": 2,
            "latency_budget_exceeded": False,
            "duration_ms": 10,
            "top_k_before": 4,
            "top_k_after": 2,
        }

    def warmup(self) -> None:
        self.warmup_called += 1

    def model_name(self) -> str:
        return "fake-model"

    def last_stats(self) -> dict[str, Any]:
        return self._stats

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        *,
        top_k_before: int | None = None,
        top_k_after: int | None = None,
    ) -> list[RetrievalResult]:
        self.rerank_calls.append(
            {
                "query": query,
                "candidates_total": len(candidates),
                "top_k_before": top_k_before,
                "top_k_after": top_k_after,
            }
        )
        selected = list(candidates)[: top_k_before or len(candidates)]
        reranked = sorted(
            selected,
            key=lambda item: len(item.text),
            reverse=True,
        )
        reranked = reranked[: top_k_after or len(reranked)]
        return [
            item.model_copy(update={"reranker_score": float(len(item.text)), "final_score": float(len(item.text))})
            for item in reranked
        ]


@pytest.mark.integration
def test_run_returns_rerank_result_and_reranks_candidates(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    fake_reranker = _FakeReranker()
    trace_path = tmp_path / "rerank_trace.json"
    layer = RerankLayer(
        question="What is 3D Secure 2?",
        candidates=[_candidate(idx=1, text="short"), _candidate(idx=2, text="longer chunk text")],
        config_path=config_path,
        reranker=fake_reranker,
        trace_path=trace_path,
    )

    with caplog.at_level("INFO"):
        result = layer.run()

    assert result.question == "What is 3D Secure 2?"
    assert result.reranked_results_total == 2
    assert result.results[0].chunk_id == "chunk-2"
    assert result.duration_ms >= 0
    assert fake_reranker.rerank_calls
    assert "Starting rerank layer" in caplog.text
    assert "Finished rerank layer" in caplog.text


@pytest.mark.integration
def test_empty_question_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        RerankLayer(question=" ", candidates=[])


@pytest.mark.integration
def test_invalid_top_k_values_raise() -> None:
    with pytest.raises(ValueError, match="top_k_before must be > 0"):
        RerankLayer(question="q", candidates=[], top_k_before=0)
    with pytest.raises(ValueError, match="top_k_after must be > 0"):
        RerankLayer(question="q", candidates=[], top_k_after=0)
    with pytest.raises(ValueError, match="top_k_after must be <="):
        RerankLayer(question="q", candidates=[], top_k_before=1, top_k_after=2)


@pytest.mark.integration
def test_disabled_reranking_returns_candidates_truncated_and_skips_reranker(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, enabled=False)
    layer = RerankLayer(
        question="What is 3D Secure 2?",
        candidates=[_candidate(idx=1, text="a"), _candidate(idx=2, text="bb"), _candidate(idx=3, text="ccc")],
        config_path=config_path,
        top_k_after=2,
        write_trace=False,
    )
    result = layer.run()
    assert result.enabled is False
    assert len(result.results) == 2
    assert [item.chunk_id for item in result.results] == ["chunk-1", "chunk-2"]
    assert result.cache_hits == 0
    assert result.cache_misses == 0


@pytest.mark.integration
def test_disabled_reranking_does_not_call_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, enabled=False)

    def _boom(_: Any) -> Any:
        raise AssertionError("Factory should not be called when reranking is disabled.")

    monkeypatch.setattr(rerank_layer_module, "create_reranker", _boom)
    layer = RerankLayer(
        question="What is 3D Secure 2?",
        candidates=[_candidate(idx=1, text="a")],
        config_path=config_path,
        write_trace=False,
    )
    _ = layer.run()


@pytest.mark.integration
def test_enabled_reranking_calls_injected_reranker_and_warmup(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, enabled=True)
    fake_reranker = _FakeReranker()
    layer = RerankLayer(
        question="What is 3D Secure 2?",
        candidates=[_candidate(idx=1, text="a"), _candidate(idx=2, text="bb")],
        config_path=config_path,
        reranker=fake_reranker,
        top_k_after=1,
        write_trace=False,
    )
    result = layer.run()
    assert fake_reranker.warmup_called == 1
    assert len(fake_reranker.rerank_calls) == 1
    assert result.reranked_results_total == 1


@pytest.mark.integration
def test_trace_written_and_contains_latency_and_truncated_preview(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, enabled=True)
    trace_path = tmp_path / "trace.json"
    fake_reranker = _FakeReranker()
    fake_reranker._stats["latency_budget_exceeded"] = True
    layer = RerankLayer(
        question="What is 3D Secure 2?",
        candidates=[_candidate(idx=1, text="this is a long chunk for preview testing")],
        config_path=config_path,
        reranker=fake_reranker,
        trace_path=trace_path,
        write_trace=True,
    )
    result = layer.run()
    assert result.trace_path == trace_path
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["latency_budget_ms"] == 50
    assert trace["latency_budget_exceeded"] is True
    assert len(trace["results"][0]["text_preview"]) == 12


@pytest.mark.integration
def test_trace_not_written_when_disabled_by_flag(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, enabled=False)
    trace_path = tmp_path / "trace.json"
    layer = RerankLayer(
        question="What is 3D Secure 2?",
        candidates=[_candidate(idx=1, text="a")],
        config_path=config_path,
        trace_path=trace_path,
        write_trace=False,
    )
    result = layer.run()
    assert result.trace_path is None
    assert not trace_path.exists()


class _FakeParser:
    def __init__(self, input_path: Path, config_path: Path) -> None:
        self._input_path = input_path
        self._config_path = config_path

    def parse_args(self) -> Namespace:
        return Namespace(
            question="What is 3D Secure 2?",
            input_path=self._input_path,
            config=self._config_path,
            top_k_before=3,
            top_k_after=2,
            trace_path=None,
            no_trace=True,
        )


@pytest.mark.integration
def test_main_invocation_with_monkeypatched_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    input_path = tmp_path / "retrieve_trace.json"
    input_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "rank": 1,
                        "chunk_id": "chunk-1",
                        "document_id": "doc-1",
                        "title": "Title",
                        "url": "https://docs.example/1",
                        "dense_score": 0.9,
                        "final_score": 0.9,
                        "text_preview": "text",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    called: dict[str, Any] = {}

    class _Layer:
        def __init__(self, **kwargs: Any) -> None:
            called["init"] = kwargs

        def run(self) -> Any:
            called["ran"] = True
            return SimpleNamespace(
                question="What is 3D Secure 2?",
                enabled=True,
                model_name="fake",
                top_k_before=3,
                top_k_after=2,
                input_results_total=1,
                reranked_results_total=1,
                cache_hits=0,
                cache_misses=1,
                latency_budget_ms=50,
                latency_budget_exceeded=False,
                duration_ms=3,
                results=[],
            )

    monkeypatch.setattr(rerank_layer_module, "_build_arg_parser", lambda: _FakeParser(input_path, config_path))
    monkeypatch.setattr(rerank_layer_module, "RerankLayer", _Layer)
    rerank_layer_module.main()
    assert called["ran"] is True
    assert called["init"]["question"] == "What is 3D Secure 2?"


@pytest.mark.integration
def test_no_real_model_loaded_when_fake_reranker_injected(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, enabled=True)
    fake_reranker = _FakeReranker()
    layer = RerankLayer(
        question="What is 3D Secure 2?",
        candidates=[_candidate(idx=1, text="a")],
        config_path=config_path,
        reranker=fake_reranker,
        write_trace=False,
    )
    _ = layer.run()
    assert fake_reranker.rerank_calls
