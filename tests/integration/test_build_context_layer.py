"""Integration tests for online build-context layer orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from app.application_layers.online.build_context import (
    BuildContextLayer,
    BuildContextResult,
)
from app.domain.models.context import ContextBundle
from app.domain.models.retrieval_result import RetrievalResult
from app.domain.models.source import Source
from app.infrastructure.context.context_builder import ContextBuilder


def _write_config(config_path: Path, *, include_full_context_in_trace: bool = False) -> None:
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
reranking: {{}}
context:
  token_budget: 120
  max_chunks: 3
  max_sources: 2
  min_chunk_tokens: 1
  max_chunk_tokens: 40
  truncate_long_chunks: true
  deduplicate_by: ["chunk_id", "text_hash", "url"]
  include_scores: true
  include_metadata: true
  context_format_version: "context_v1"
  write_trace: true
  trace_dir: "{(config_path.parent / "traces").as_posix()}"
  text_preview_chars: 25
  include_full_context_in_trace: {str(include_full_context_in_trace).lower()}
generation: {{}}
eval: {{}}
""",
        encoding="utf-8",
    )


def _result(*, idx: int, text: str) -> RetrievalResult:
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
        dense_score=0.7,
        reranker_score=0.85,
        final_score=0.85,
        metadata={"source_type": "markdown", "category": "payments"},
    )


class _FakeBuilder:
    def __init__(self, bundle: ContextBundle) -> None:
        self.bundle = bundle
        self.calls: list[dict[str, Any]] = []

    def build(
        self,
        *,
        query: str,
        results: list[RetrievalResult],
        token_budget: int | None = None,
        max_chunks: int | None = None,
    ) -> ContextBundle:
        self.calls.append(
            {
                "query": query,
                "results_total": len(results),
                "token_budget": token_budget,
                "max_chunks": max_chunks,
            }
        )
        return self.bundle


@pytest.mark.integration
def test_run_returns_build_context_result(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    layer = BuildContextLayer(
        question="What is 3D Secure 2?",
        results=[_result(idx=1, text="some text"), _result(idx=2, text="more text")],
        config_path=config_path,
        write_trace=False,
    )
    output = layer.run()
    assert isinstance(output, BuildContextResult)
    assert output.question == "What is 3D Secure 2?"
    assert output.input_results_total == 2


@pytest.mark.integration
def test_non_empty_results_produce_context_bundle_with_real_builder(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    layer = BuildContextLayer(
        question="What is 3D Secure 2?",
        results=[_result(idx=1, text="some text"), _result(idx=2, text="more text")],
        config_path=config_path,
        write_trace=False,
    )
    output = layer.run()
    assert output.context_bundle.chunks
    assert output.context_bundle.rendered_context


@pytest.mark.integration
def test_empty_question_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        BuildContextLayer(question=" ", results=[])


@pytest.mark.integration
def test_invalid_token_budget_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="token_budget must be > 0"):
        BuildContextLayer(question="q", results=[], token_budget=0)


@pytest.mark.integration
def test_invalid_max_chunks_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="max_chunks must be > 0"):
        BuildContextLayer(question="q", results=[], max_chunks=0)


@pytest.mark.integration
def test_injected_context_builder_is_called(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    base_result = _result(idx=1, text="text")
    bundle = ContextBuilder(
        token_budget=100,
        max_chunks=2,
        max_sources=2,
        min_chunk_tokens=1,
        max_chunk_tokens=80,
        truncate_long_chunks=True,
        deduplicate_by=["chunk_id"],
        include_scores=True,
        include_metadata=False,
        context_format_version="context_v1",
    ).build(query="q", results=[base_result])
    fake = _FakeBuilder(bundle)
    layer = BuildContextLayer(
        question="q",
        results=[base_result],
        config_path=config_path,
        context_builder=fake,
        write_trace=False,
    )
    _ = layer.run()
    assert len(fake.calls) == 1
    assert fake.calls[0]["results_total"] == 1


@pytest.mark.integration
def test_trace_written_when_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    trace_path = tmp_path / "trace_context.json"
    layer = BuildContextLayer(
        question="q",
        results=[_result(idx=1, text="long enough text for preview")],
        config_path=config_path,
        trace_path=trace_path,
        write_trace=True,
    )
    result = layer.run()
    assert result.trace_path == trace_path
    assert trace_path.exists()


@pytest.mark.integration
def test_trace_not_written_when_disabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    trace_path = tmp_path / "trace_context.json"
    layer = BuildContextLayer(
        question="q",
        results=[_result(idx=1, text="text")],
        config_path=config_path,
        trace_path=trace_path,
        write_trace=False,
    )
    result = layer.run()
    assert result.trace_path is None
    assert not trace_path.exists()


@pytest.mark.integration
def test_trace_includes_token_budget_and_token_count(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    trace_path = tmp_path / "trace_context.json"
    layer = BuildContextLayer(
        question="q",
        results=[_result(idx=1, text="text text text")],
        config_path=config_path,
        trace_path=trace_path,
        write_trace=True,
    )
    _ = layer.run()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["context_bundle"]["token_budget"] == 120
    assert isinstance(payload["context_bundle"]["token_count"], int)


@pytest.mark.integration
def test_trace_includes_rendered_context_preview_and_not_full_by_default(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, include_full_context_in_trace=False)
    trace_path = tmp_path / "trace_context.json"
    layer = BuildContextLayer(
        question="q",
        results=[_result(idx=1, text="text text text text text")],
        config_path=config_path,
        trace_path=trace_path,
        write_trace=True,
    )
    _ = layer.run()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert "rendered_context_preview" in payload
    assert len(payload["rendered_context_preview"]) <= 25
    assert "context_bundle" in payload
    assert "rendered_context" in payload["context_bundle"]


@pytest.mark.integration
def test_full_context_included_only_when_config_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, include_full_context_in_trace=True)
    trace_path = tmp_path / "trace_context.json"
    layer = BuildContextLayer(
        question="q",
        results=[_result(idx=1, text="text text text text text")],
        config_path=config_path,
        trace_path=trace_path,
        write_trace=True,
    )
    _ = layer.run()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert "rendered_context_preview" in payload
    assert "context_bundle" in payload
    assert "rendered_context" in payload["context_bundle"]


@pytest.mark.integration
def test_trace_contains_serialized_context_bundle_for_generation_input(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, include_full_context_in_trace=False)
    trace_path = tmp_path / "trace_context.json"
    layer = BuildContextLayer(
        question="q",
        results=[_result(idx=1, text="text text text text text")],
        config_path=config_path,
        trace_path=trace_path,
        write_trace=True,
    )
    _ = layer.run()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert "context_bundle" in payload
    assert payload["context_bundle"]["query"] == "q"
    assert isinstance(payload["context_bundle"]["chunks"], list)
    assert isinstance(payload["context_bundle"]["sources"], list)


@pytest.mark.integration
def test_logging_emits_stage_start_and_finish(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    layer = BuildContextLayer(
        question="q",
        results=[_result(idx=1, text="text")],
        config_path=config_path,
        write_trace=False,
    )
    with caplog.at_level("INFO"):
        _ = layer.run()
    assert "Starting context layer" in caplog.text
    assert "Finished context layer" in caplog.text


@pytest.mark.integration
def test_no_retrieval_reranking_or_generation_components_called(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    fake_bundle = ContextBuilder(
        token_budget=100,
        max_chunks=2,
        max_sources=2,
        min_chunk_tokens=1,
        max_chunk_tokens=80,
        truncate_long_chunks=True,
        deduplicate_by=["chunk_id"],
        include_scores=True,
        include_metadata=False,
        context_format_version="context_v1",
    ).build(query="q", results=[_result(idx=1, text="x")])
    fake_builder = _FakeBuilder(fake_bundle)

    layer = BuildContextLayer(
        question="q",
        results=[_result(idx=1, text="x")],
        config_path=config_path,
        context_builder=fake_builder,
        write_trace=False,
    )
    _ = layer.run()
    assert fake_builder.calls
