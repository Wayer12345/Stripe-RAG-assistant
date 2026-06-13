"""Unit tests for generation infrastructure components."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from app.domain.models.answer import Confidence, GeneratedAnswer
from app.domain.models.context import ContextBundle
from app.domain.models.retrieval_result import RetrievalResult
from app.domain.models.source import Source
from app.infrastructure.generation import (
    AnswerGenerator,
    OllamaClient,
    OutputParser,
    PromptRenderer,
    create_answer_generator,
)
from app.utils.config import load_settings


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
        text="3D Secure 2 adds a frictionless authentication flow with issuer challenge fallback.",
        source=source,
        retrieval_score=0.9,
        lexical_score=0.7,
        dense_score=0.8,
        reranker_score=0.95,
        final_score=0.95,
    )
    return ContextBundle(
        query="What is 3D Secure 2?",
        chunks=[result],
        rendered_context=result.text,
        token_count=24,
        sources=[source],
        token_budget=3500,
        truncated=False,
        context_format_version="context_v1",
    )


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeHttpClient:
    def __init__(self) -> None:
        self.last_post: dict[str, Any] | None = None
        self.fail_post: Exception | None = None
        self.get_status_code = 200
        self.post_response = _FakeResponse(200, {"response": '{"confidence":"low","answer":"ok","sources":[]}'})

    def get(self, _: str) -> _FakeResponse:
        return _FakeResponse(self.get_status_code, {})

    def post(self, url: str, json: dict[str, Any]) -> _FakeResponse:
        if self.fail_post is not None:
            raise self.fail_post
        self.last_post = {"url": url, "json": json}
        return self.post_response


@pytest.mark.unit
def test_ollama_client_generate_success_and_options() -> None:
    fake_http = _FakeHttpClient()
    fake_http.post_response = _FakeResponse(200, {"response": "raw text"})
    client = OllamaClient(
        base_url="http://localhost:11434",
        model_name="llama3.1:8b",
        timeout_seconds=10,
        temperature=0.1,
        max_tokens=700,
        top_p=0.9,
        stop=["</END>"],
        http_client=fake_http,  # type: ignore[arg-type]
    )
    raw = client.generate("hello")
    assert raw == "raw text"
    assert fake_http.last_post is not None
    assert fake_http.last_post["json"]["options"]["num_predict"] == 700
    assert fake_http.last_post["json"]["options"]["temperature"] == 0.1
    assert fake_http.last_post["json"]["options"]["top_p"] == 0.9
    assert fake_http.last_post["json"]["options"]["stop"] == ["</END>"]


@pytest.mark.unit
def test_ollama_client_empty_prompt_raises() -> None:
    client = OllamaClient(
        base_url="http://localhost:11434",
        model_name="llama3.1:8b",
        timeout_seconds=10,
        temperature=0.1,
        max_tokens=10,
        http_client=_FakeHttpClient(),  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="prompt must not be empty"):
        _ = client.generate(" ")


@pytest.mark.unit
def test_ollama_client_non_200_raises() -> None:
    fake_http = _FakeHttpClient()
    fake_http.post_response = _FakeResponse(500, {"error": "boom"})
    client = OllamaClient(
        base_url="http://localhost:11434",
        model_name="llama3.1:8b",
        timeout_seconds=10,
        temperature=0.1,
        max_tokens=10,
        http_client=fake_http,  # type: ignore[arg-type]
    )
    with pytest.raises(RuntimeError, match="non-200"):
        _ = client.generate("q")


@pytest.mark.unit
def test_ollama_client_missing_response_text_raises() -> None:
    fake_http = _FakeHttpClient()
    fake_http.post_response = _FakeResponse(200, {"done": True})
    client = OllamaClient(
        base_url="http://localhost:11434",
        model_name="llama3.1:8b",
        timeout_seconds=10,
        temperature=0.1,
        max_tokens=10,
        http_client=fake_http,  # type: ignore[arg-type]
    )
    with pytest.raises(RuntimeError, match="missing expected 'response'"):
        _ = client.generate("q")


@pytest.mark.unit
def test_ollama_client_timeout_or_http_error_raises() -> None:
    import httpx

    fake_http = _FakeHttpClient()
    fake_http.fail_post = httpx.TimeoutException("timeout")
    client = OllamaClient(
        base_url="http://localhost:11434",
        model_name="llama3.1:8b",
        timeout_seconds=10,
        temperature=0.1,
        max_tokens=10,
        http_client=fake_http,  # type: ignore[arg-type]
    )
    with pytest.raises(RuntimeError, match="timed out"):
        _ = client.generate("q")


@pytest.mark.unit
def test_ollama_healthcheck_success_and_failure() -> None:
    fake_http = _FakeHttpClient()
    client = OllamaClient(
        base_url="http://localhost:11434",
        model_name="llama3.1:8b",
        timeout_seconds=10,
        temperature=0.1,
        max_tokens=10,
        http_client=fake_http,  # type: ignore[arg-type]
    )
    assert client.healthcheck() is True
    fake_http.get_status_code = 503
    assert client.healthcheck() is False


@pytest.mark.unit
def test_prompt_renderer_renders_template_with_context(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "answer_prompt_v1.jinja").write_text(
        "Q={{ question }} C={{ rendered_context }} S={{ context_sources|length }}",
        encoding="utf-8",
    )
    (prompts_dir / "no_answer_prompt_v1.jinja").write_text("{}", encoding="utf-8")

    bundle = _context_bundle()
    renderer = PromptRenderer(prompts_dir=prompts_dir)
    prompt = renderer.render_answer_prompt(question="What is 3DS2?", context_bundle=bundle)
    assert "What is 3DS2?" in prompt
    assert bundle.rendered_context in prompt
    assert "S=1" in prompt


@pytest.mark.unit
def test_prompt_renderer_missing_template_raises(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "no_answer_prompt_v1.jinja").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Prompt template not found"):
        _ = PromptRenderer(prompts_dir=prompts_dir)


@pytest.mark.unit
def test_output_parser_handles_plain_text_answer() -> None:
    parser = OutputParser()
    bundle = _context_bundle()
    raw_output = "3DS2 improves authentication with issuer challenge fallback."
    answer = parser.parse(raw_output=raw_output, context_bundle=bundle)
    assert answer.parsed_successfully is True
    assert answer.confidence == Confidence.LOW
    assert len(answer.sources) == 1
    assert answer.answer.startswith("3DS2 improves")


@pytest.mark.unit
def test_output_parser_extracts_answer_from_marked_block_and_ignores_explanation() -> None:
    parser = OutputParser()
    bundle = _context_bundle()
    raw_output = """Answer:
BEGIN_RESPONSE
answer: 3DS2 improves authentication with issuer-side checks.
END_RESPONSE
Explanation:
extra details that should be ignored."""
    answer = parser.parse(raw_output=raw_output, context_bundle=bundle)
    assert answer.parsed_successfully is True
    assert answer.confidence == Confidence.LOW
    assert len(answer.sources) == 1
    assert "issuer-side checks" in answer.answer


@pytest.mark.unit
def test_output_parser_keeps_only_first_paragraph() -> None:
    parser = OutputParser()
    bundle = _context_bundle()
    raw_output = (
        "3DS2 improves authentication with issuer-side checks.\n\n"
        "Additional commentary that should not be returned to the user."
    )
    answer = parser.parse(raw_output=raw_output, context_bundle=bundle)
    assert answer.parsed_successfully is True
    assert answer.answer == "3DS2 improves authentication with issuer-side checks."


@pytest.mark.unit
def test_output_parser_skips_question_like_first_paragraph() -> None:
    parser = OutputParser()
    bundle = _context_bundle()
    raw_output = (
        "If payment fails and Smart Retries are on, what happens to a subscription?\n\n"
        "With Smart Retries enabled, Stripe Billing automatically retries failed payments "
        "at optimized times to improve recovery and reduce involuntary churn."
    )
    answer = parser.parse(raw_output=raw_output, context_bundle=bundle)
    assert answer.parsed_successfully is True
    assert answer.answer.startswith("With Smart Retries enabled")


@pytest.mark.unit
def test_output_parser_collapses_repeated_sentence_cycles() -> None:
    parser = OutputParser()
    bundle = _context_bundle()
    raw_output = (
        "Use SetupIntent to save card details without charging now. "
        "Use PaymentIntent when you plan to charge immediately. "
        "Use SetupIntent to save card details without charging now. "
        "Use PaymentIntent when you plan to charge immediately. "
        "Use SetupIntent to save card details without charging now."
    )
    answer = parser.parse(raw_output=raw_output, context_bundle=bundle)
    assert answer.parsed_successfully is True
    assert (
        answer.answer
        == "Use SetupIntent to save card details without charging now. Use PaymentIntent when you plan to charge immediately."
    )


@pytest.mark.unit
def test_output_parser_extracts_single_segment_from_repeated_answer_markers() -> None:
    parser = OutputParser()
    bundle = _context_bundle()
    raw_output = (
        "Is it safe to store order status in Stripe metadata? "
        "Answer: It is usually better to store order state in your own system of record. "
        "Answer: It is usually better to store order state in your own system of record. "
        "Answer: It is usually better to store order state in your own system of record."
    )
    answer = parser.parse(raw_output=raw_output, context_bundle=bundle)
    assert answer.parsed_successfully is True
    assert answer.answer == "It is usually better to store order state in your own system of record."


@pytest.mark.unit
def test_output_parser_truncates_long_multi_sentence_output() -> None:
    parser = OutputParser()
    bundle = _context_bundle()
    raw_output = (
        "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. Sentence six."
    )
    answer = parser.parse(raw_output=raw_output, context_bundle=bundle)
    assert answer.parsed_successfully is True
    assert answer.answer == "Sentence one. Sentence two. Sentence three. Sentence four."


@pytest.mark.unit
def test_output_parser_extracts_answer_from_json_when_present() -> None:
    parser = OutputParser()
    bundle = _context_bundle()
    raw_output = """{
      "confidence":"high",
      "answer":"JSON answer text",
      "sources":[]
    }"""
    answer = parser.parse(raw_output=raw_output, context_bundle=bundle)
    assert answer.parsed_successfully is True
    assert answer.answer == "JSON answer text"
    assert answer.confidence == Confidence.LOW


@pytest.mark.unit
def test_output_parser_falls_back_on_empty_answer() -> None:
    parser = OutputParser()
    bundle = _context_bundle()
    raw_output = "   "
    answer = parser.parse(raw_output=raw_output, context_bundle=bundle)
    assert answer.parsed_successfully is False
    assert answer.confidence == Confidence.NONE


@pytest.mark.unit
def test_output_parser_falls_back_when_model_returns_no_answer_phrase() -> None:
    parser = OutputParser(no_answer_quality_threshold_pct=95.0)
    bundle = _context_bundle()
    raw_output = "I don't have enough information in the indexed Stripe Guides sources to answer this reliably."
    answer = parser.parse(raw_output=raw_output, context_bundle=bundle)
    assert answer.parsed_successfully is False
    assert answer.confidence == Confidence.NONE
    assert answer.sources == []


@pytest.mark.unit
def test_output_parser_falls_back_when_no_answer_phrase_has_extra_text() -> None:
    parser = OutputParser(no_answer_quality_threshold_pct=95.0)
    bundle = _context_bundle()
    raw_output = (
        "I don't have enough information in the indexed Stripe Guides sources to answer this reliably.\n\n"
        "Note: this should have been final, but here are extra thoughts."
    )
    answer = parser.parse(raw_output=raw_output, context_bundle=bundle)
    assert answer.parsed_successfully is False
    assert answer.confidence == Confidence.NONE
    assert answer.sources == []


@pytest.mark.unit
def test_output_parser_uses_best_effort_when_quality_is_above_threshold() -> None:
    parser = OutputParser(no_answer_quality_threshold_pct=50.0)
    bundle = _context_bundle()
    raw_output = "I don't have enough information in the indexed Stripe Guides sources to answer this reliably."
    answer = parser.parse(raw_output=raw_output, context_bundle=bundle)
    assert answer.parsed_successfully is True
    assert answer.confidence == Confidence.LOW
    assert "frictionless authentication flow" in answer.answer


@pytest.mark.unit
def test_output_parser_keeps_answer_when_fallback_phrase_appears_in_correction_text() -> None:
    parser = OutputParser()
    bundle = _context_bundle()
    raw_output = (
        "Partial refunds add up across refund attempts.\n"
        "Answer:\n"
        "I don't have enough information in the indexed Stripe Guides sources to answer this reliably. "
        "was incorrect. Here is the correct answer:\n\n"
        "If a prior partial refund exists, only the remaining amount can be refunded."
    )
    answer = parser.parse(raw_output=raw_output, context_bundle=bundle)
    assert answer.parsed_successfully is True
    assert answer.confidence == Confidence.LOW
    assert "remaining amount can be refunded" in answer.answer


@pytest.mark.unit
def test_output_parser_uses_up_to_three_context_sources() -> None:
    sources = [
        Source(
            title=f"Source {idx}",
            url=f"https://docs.example/{idx}",
            section="Overview",
            chunk_id=f"chunk-{idx}",
            document_id=f"doc-{idx}",
            support_score=0.5,
        )
        for idx in range(1, 6)
    ]
    results = [
        RetrievalResult(
            chunk_id=source.chunk_id,
            document_id=source.document_id,
            text=f"text-{source.chunk_id}",
            source=source,
            final_score=0.9,
        )
        for source in sources
    ]
    bundle = ContextBundle(
        query="What is 3D Secure 2?",
        chunks=results,
        rendered_context="context",
        token_count=256,
        sources=sources,
        token_budget=3500,
        truncated=False,
        context_format_version="context_v1",
    )
    parser = OutputParser()
    answer = parser.parse(raw_output="Short grounded answer.", context_bundle=bundle)
    assert answer.parsed_successfully is True
    assert answer.confidence == Confidence.MEDIUM
    assert len(answer.sources) == 3


class _FakeRenderer:
    def __init__(self) -> None:
        self.calls = 0

    def render_answer_prompt(self, *, question: str, context_bundle: ContextBundle) -> str:
        self.calls += 1
        return f"Q:{question} C:{context_bundle.token_count}"

    def template_name(self) -> str:
        return "answer_prompt_v1.jinja"

    def template_version(self) -> str:
        return "answer_prompt_v1"


class _FakeLlmClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return """BEGIN_RESPONSE
confidence:
low

answer:
Based on context.

sources:
[]
END_RESPONSE"""

    def model_name(self) -> str:
        return "llama3.1:8b"


@pytest.mark.unit
def test_answer_generator_empty_context_short_circuits() -> None:
    bundle = _context_bundle().model_copy(update={"rendered_context": "", "token_count": 0})
    renderer = _FakeRenderer()
    llm = _FakeLlmClient()
    parser = OutputParser()
    generator = AnswerGenerator(
        prompt_renderer=renderer,  # type: ignore[arg-type]
        llm_client=llm,  # type: ignore[arg-type]
        output_parser=parser,
    )
    answer = generator.generate(question="q", context_bundle=bundle)
    assert answer.confidence == Confidence.NONE
    assert renderer.calls == 0
    assert llm.calls == 0
    assert generator.last_stats()["no_answer_used"] is True


@pytest.mark.unit
def test_answer_generator_calls_renderer_llm_and_parser() -> None:
    renderer = _FakeRenderer()
    llm = _FakeLlmClient()
    parser = OutputParser()
    generator = AnswerGenerator(
        prompt_renderer=renderer,  # type: ignore[arg-type]
        llm_client=llm,  # type: ignore[arg-type]
        output_parser=parser,
    )
    answer = generator.generate(question="q", context_bundle=_context_bundle())
    assert renderer.calls == 1
    assert llm.calls == 1
    assert isinstance(answer, GeneratedAnswer)
    stats = generator.last_stats()
    assert stats["provider"] == "ollama"
    assert stats["model_name"] == "llama3.1:8b"
    assert "total_duration_ms" in stats


@pytest.mark.unit
def test_factory_creates_generator_and_uses_config_values() -> None:
    settings = load_settings(Path("configs"))
    generator = create_answer_generator(settings)
    assert isinstance(generator, AnswerGenerator)


@pytest.mark.unit
def test_factory_rejects_unsupported_provider() -> None:
    settings = load_settings(Path("configs"))
    mutated = settings.model_copy(
        update={
            "generation": settings.generation.model_copy(update={"provider": "openai"}),
        }
    )
    with pytest.raises(ValueError, match="Unsupported generation provider"):
        _ = create_answer_generator(mutated)
