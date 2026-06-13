"""Integration tests for online generate-answer layer orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import app.application_layers.online.generate_answer as generate_answer_module
import pytest
from app.application_layers.online.generate_answer import GenerateAnswerLayer, GenerateAnswerResult
from app.domain.models.answer import Confidence, GeneratedAnswer
from app.domain.models.context import ContextBundle
from app.domain.models.retrieval_result import RetrievalResult
from app.domain.models.source import Source


def _write_config(
    config_path: Path,
    *,
    include_full_prompt_in_trace: bool = False,
    include_raw_output_in_trace: bool = False,
    include_full_answer_in_trace: bool = False,
) -> None:
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
context: {{}}
generation:
  provider: "ollama"
  base_url: "http://localhost:11434"
  model_name: "llama3.1:8b"
  temperature: 0.1
  max_tokens: 128
  top_p: null
  timeout_seconds: 30
  context_token_budget: 3500
  min_context_tokens: 1
  prompts_dir: "prompts"
  answer_template_name: "answer_prompt_v1.jinja"
  no_answer_template_name: "no_answer_prompt_v1.jinja"
  write_trace: true
  trace_dir: "{(config_path.parent / "traces").as_posix()}"
  text_preview_chars: 10
  include_full_prompt_in_trace: {str(include_full_prompt_in_trace).lower()}
  include_raw_output_in_trace: {str(include_raw_output_in_trace).lower()}
  include_full_answer_in_trace: {str(include_full_answer_in_trace).lower()}
eval: {{}}
""",
        encoding="utf-8",
    )


def _context_bundle() -> ContextBundle:
    source = Source(
        title="Stripe 3DS2",
        url="https://docs.stripe.com/3ds",
        section="Overview",
        chunk_id="chunk-1",
        document_id="doc-1",
        support_score=0.9,
    )
    result = RetrievalResult(
        chunk_id="chunk-1",
        document_id="doc-1",
        text="3D Secure 2 helps with authentication.",
        source=source,
        retrieval_score=0.9,
        lexical_score=None,
        dense_score=0.8,
        reranker_score=0.95,
        final_score=0.95,
    )
    return ContextBundle(
        query="What is 3D Secure 2?",
        chunks=[result],
        rendered_context=result.text,
        token_count=16,
        sources=[source],
        token_budget=3500,
        truncated=False,
        context_format_version="context_v1",
    )


class _FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0
        self._stats: dict[str, Any] = {
            "provider": "ollama",
            "model_name": "llama3.1:8b",
            "prompt_template": "answer_prompt_v1.jinja",
            "prompt_template_version": "answer_prompt_v1",
            "context_token_count": 16,
            "context_sources_total": 1,
            "context_truncated": False,
            "no_answer_used": False,
            "llm_duration_ms": 7,
            "parse_duration_ms": 2,
            "rendered_prompt": "rendered prompt text",
        }

    def generate(self, *, question: str, context_bundle: ContextBundle) -> GeneratedAnswer:
        self.calls += 1
        return GeneratedAnswer(
            answer=f"Generated: {question}",
            confidence=Confidence.MEDIUM,
            sources=context_bundle.sources,
            raw_output='{"confidence":"medium","answer":"Generated"}',
            parsed_successfully=True,
            metadata={},
        )

    def last_stats(self) -> dict[str, Any]:
        return dict(self._stats)


@pytest.mark.integration
def test_run_returns_generate_answer_result(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    layer = GenerateAnswerLayer(
        question="What is 3D Secure 2?",
        context_bundle=_context_bundle(),
        config_path=config_path,
        answer_generator=_FakeGenerator(),
        write_trace=False,
    )
    result = layer.run()
    assert isinstance(result, GenerateAnswerResult)
    assert result.confidence == "medium"


@pytest.mark.integration
def test_empty_question_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        GenerateAnswerLayer(question=" ", context_bundle=_context_bundle())


@pytest.mark.integration
def test_injected_generator_is_called(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    fake = _FakeGenerator()
    result = GenerateAnswerLayer(
        question="What is 3D Secure 2?",
        context_bundle=_context_bundle(),
        config_path=config_path,
        answer_generator=fake,
        write_trace=False,
    ).run()
    assert fake.calls == 1
    assert result.generated_answer.answer.startswith("Generated:")


@pytest.mark.integration
def test_trace_written_when_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    trace_path = tmp_path / "generation_trace.json"
    result = GenerateAnswerLayer(
        question="What is 3D Secure 2?",
        context_bundle=_context_bundle(),
        config_path=config_path,
        trace_path=trace_path,
        answer_generator=_FakeGenerator(),
        write_trace=True,
    ).run()
    assert result.trace_path == trace_path
    assert trace_path.exists()


@pytest.mark.integration
def test_trace_not_written_when_disabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    trace_path = tmp_path / "generation_trace.json"
    result = GenerateAnswerLayer(
        question="What is 3D Secure 2?",
        context_bundle=_context_bundle(),
        config_path=config_path,
        trace_path=trace_path,
        answer_generator=_FakeGenerator(),
        write_trace=False,
    ).run()
    assert result.trace_path is None
    assert not trace_path.exists()


@pytest.mark.integration
def test_trace_includes_required_fields_and_previews_by_default(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    trace_path = tmp_path / "generation_trace.json"
    _ = GenerateAnswerLayer(
        question="What is 3D Secure 2?",
        context_bundle=_context_bundle(),
        config_path=config_path,
        trace_path=trace_path,
        answer_generator=_FakeGenerator(),
        write_trace=True,
    ).run()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["confidence"] == "medium"
    assert payload["parsed_successfully"] is True
    assert payload["sources_total"] == 1
    assert "answer_preview" in payload
    assert "raw_output_preview" in payload
    assert "answer" not in payload
    assert "raw_output" not in payload
    assert "rendered_prompt" not in payload


@pytest.mark.integration
def test_trace_can_include_full_answer_raw_output_and_prompt(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        include_full_prompt_in_trace=True,
        include_raw_output_in_trace=True,
        include_full_answer_in_trace=True,
    )
    trace_path = tmp_path / "generation_trace.json"
    _ = GenerateAnswerLayer(
        question="What is 3D Secure 2?",
        context_bundle=_context_bundle(),
        config_path=config_path,
        trace_path=trace_path,
        answer_generator=_FakeGenerator(),
        write_trace=True,
    ).run()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert "answer" in payload
    assert "raw_output" in payload
    assert "rendered_prompt" in payload


@pytest.mark.integration
def test_logging_emits_stage_start_and_finish(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    layer = GenerateAnswerLayer(
        question="What is 3D Secure 2?",
        context_bundle=_context_bundle(),
        config_path=config_path,
        answer_generator=_FakeGenerator(),
        write_trace=False,
    )
    with caplog.at_level("INFO"):
        _ = layer.run()
    assert "Starting generate-answer layer" in caplog.text
    assert "Finished generate-answer layer" in caplog.text


@pytest.mark.integration
def test_no_real_ollama_call_when_generator_injected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    def _boom(_: Any) -> Any:
        raise AssertionError("Factory should not be called when generator is injected.")

    monkeypatch.setattr(generate_answer_module, "create_answer_generator", _boom)
    result = GenerateAnswerLayer(
        question="What is 3D Secure 2?",
        context_bundle=_context_bundle(),
        config_path=config_path,
        answer_generator=_FakeGenerator(),
        write_trace=False,
    ).run()
    assert result.confidence == "medium"
