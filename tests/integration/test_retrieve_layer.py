"""Integration tests for the online retrieve layer orchestration."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import app.application_layers.online.retrieve as retrieve_layer_module
import pytest
from app.application_layers.online.retrieve import RetrieveLayer
from app.domain.models.retrieval_result import RetrievalMethod, RetrievalResult
from app.domain.models.source import Source


def _write_config(config_path: Path) -> None:
    config_path.write_text(
        """
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
chunking: {}
embeddings:
  model_name: "BAAI/bge-small-en-v1.5"
vector_store:
  collection_name: "stripe_guides_v1"
indexing: {}
retrieval:
  strategy: "dense"
  dense_top_k: 5
  write_trace: true
  trace_dir: "data/traces/queries"
reranking: {}
generation: {}
eval: {}
""",
        encoding="utf-8",
    )


def _fake_result(*, text: str) -> RetrievalResult:
    source = Source(
        title="3D Secure",
        url="https://docs.example/3ds",
        section="Authentication",
        chunk_id="chunk-1",
        document_id="doc-1",
        support_score=0.9,
        source_type="markdown",
    )
    return RetrievalResult(
        chunk_id="chunk-1",
        document_id="doc-1",
        text=text,
        source=source,
        retrieval_score=0.9,
        dense_score=0.9,
        lexical_score=None,
        reranker_score=None,
        final_score=0.9,
        retrieval_method=RetrievalMethod.DENSE,
        rank=1,
        metadata={"token_count": 33, "source_type": "markdown", "embedding_dim": 384},
    )


class _FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return self._results


@pytest.mark.integration
def test_run_returns_retrieve_result_and_calls_injected_retriever(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    fake = _FakeRetriever([_fake_result(text="a" * 500)])
    layer = RetrieveLayer(
        question="What is 3D Secure 2?",
        config_path=config_path,
        top_k=3,
        filters={"document_id": "doc-1"},
        trace_path=tmp_path / "trace.json",
        retriever=fake,
    )

    with caplog.at_level("INFO"):
        result = layer.run()

    assert result.question == "What is 3D Secure 2?"
    assert result.results_total == 1
    assert result.top_k == 3
    assert result.strategy == "dense"
    assert result.duration_ms >= 0
    assert fake.calls == [
        {"query": "What is 3D Secure 2?", "top_k": 3, "filters": {"document_id": "doc-1"}}
    ]
    assert "Starting retrieval layer" in caplog.text
    assert "Finished retrieval layer" in caplog.text


@pytest.mark.integration
def test_empty_question_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        RetrieveLayer(question="   ")


@pytest.mark.integration
def test_top_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="top_k must be > 0"):
        RetrieveLayer(question="valid", top_k=0)


@pytest.mark.integration
def test_trace_written_when_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    trace_path = tmp_path / "trace.json"
    layer = RetrieveLayer(
        question="What is 3D Secure 2?",
        config_path=config_path,
        retriever=_FakeRetriever([_fake_result(text="x" * 1000)]),
        trace_path=trace_path,
        write_trace=True,
    )

    result = layer.run()

    assert result.trace_path == trace_path
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["run_id"] == result.run_id
    assert trace["question"] == "What is 3D Secure 2?"
    assert trace["strategy"] == "dense"
    assert trace["results_total"] == 1
    assert trace["duration_ms"] >= 0
    assert len(trace["results"][0]["text_preview"]) == 300


@pytest.mark.integration
def test_trace_not_written_when_disabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    trace_path = tmp_path / "trace.json"
    layer = RetrieveLayer(
        question="What is 3D Secure 2?",
        config_path=config_path,
        retriever=_FakeRetriever([_fake_result(text="small text")]),
        trace_path=trace_path,
        write_trace=False,
    )

    result = layer.run()

    assert result.trace_path is None
    assert not trace_path.exists()


class _FakeParser:
    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path

    def parse_args(self) -> Namespace:
        return Namespace(
            question="What is 3D Secure 2?",
            config=self._config_path,
            top_k=2,
            filter_document_id=None,
            filter_source_type=None,
            filter_url=None,
            trace_path=None,
            no_trace=True,
        )


@pytest.mark.integration
def test_main_invocation_with_monkeypatched_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    called: dict[str, Any] = {}

    class _Layer:
        def __init__(self, **kwargs: Any) -> None:
            called["init"] = kwargs

        def run(self) -> Any:
            called["ran"] = True
            return SimpleNamespace(
                question="What is 3D Secure 2?",
                strategy="dense",
                top_k=2,
                results_total=0,
                duration_ms=1,
                results=[],
            )

    monkeypatch.setattr(retrieve_layer_module, "_build_arg_parser", lambda: _FakeParser(config_path))
    monkeypatch.setattr(retrieve_layer_module, "RetrieveLayer", _Layer)
    retrieve_layer_module.main()

    assert called["ran"] is True
    assert called["init"]["question"] == "What is 3D Secure 2?"
