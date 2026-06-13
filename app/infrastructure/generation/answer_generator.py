"""Core answer generation orchestration."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from app.domain.models.answer import Confidence, GeneratedAnswer
from app.domain.models.context import ContextBundle
from app.infrastructure.generation.ollama_client import OllamaClient
from app.infrastructure.generation.output_parser import OutputParser
from app.infrastructure.generation.prompt_renderer import PromptRenderer
from app.utils.config import Settings
from app.utils.constants import CONFIDENCE_NONE
from app.utils.logging import get_logger

logger = get_logger(__name__)

_FALLBACK_ANSWER = (
    "I don't have enough information in the indexed Stripe Guides sources to answer this reliably."
)


class AnswerGenerator:
    """Core generation pipeline: render prompt -> LLM -> parse output."""

    def __init__(
        self,
        *,
        prompt_renderer: PromptRenderer,
        llm_client: OllamaClient,
        output_parser: OutputParser,
        min_context_tokens: int = 1,
        warmup_ollama_healthcheck_enabled: bool = True,
        warmup_ollama_generate_enabled: bool = True,
        warmup_ollama_generate_prompt: str = "Respond with: OK",
        warmup_ollama_generate_max_tokens: int = 8,
    ) -> None:
        if min_context_tokens <= 0:
            raise ValueError("min_context_tokens must be > 0.")
        if not warmup_ollama_generate_prompt.strip():
            raise ValueError("warmup_ollama_generate_prompt must not be empty.")
        if warmup_ollama_generate_max_tokens <= 0:
            raise ValueError("warmup_ollama_generate_max_tokens must be > 0.")
        self._prompt_renderer = prompt_renderer
        self._llm_client = llm_client
        self._output_parser = output_parser
        self._min_context_tokens = min_context_tokens
        self._warmup_ollama_healthcheck_enabled = warmup_ollama_healthcheck_enabled
        self._warmup_ollama_generate_enabled = warmup_ollama_generate_enabled
        self._warmup_ollama_generate_prompt = warmup_ollama_generate_prompt.strip()
        self._warmup_ollama_generate_max_tokens = warmup_ollama_generate_max_tokens
        self._last_stats: dict[str, Any] = {}

    def last_stats(self) -> dict[str, Any]:
        """Return stats of the last generation call."""
        return dict(self._last_stats)

    def warmup(self) -> dict[str, bool | str | None]:
        """Warm generation path using Ollama healthcheck and tiny generate."""
        ollama_healthcheck_ok: bool | None = None
        ollama_generate_warmup_ok: bool | None = None

        if self._warmup_ollama_healthcheck_enabled:
            ollama_healthcheck_ok = bool(self._llm_client.healthcheck())

        if self._warmup_ollama_generate_enabled and ollama_healthcheck_ok is not False:
            ollama_generate_warmup_ok = bool(
                self._llm_client.warmup_generate(
                    prompt=self._warmup_ollama_generate_prompt,
                    max_tokens=self._warmup_ollama_generate_max_tokens,
                )
            )

        status = (
            "success"
            if (ollama_healthcheck_ok is not False and ollama_generate_warmup_ok is not False)
            else "failed"
        )
        return {
            "status": status,
            "ollama_healthcheck_ok": ollama_healthcheck_ok,
            "ollama_generate_warmup_ok": ollama_generate_warmup_ok,
        }

    def generate(
        self,
        *,
        question: str,
        context_bundle: ContextBundle,
    ) -> GeneratedAnswer:
        """Generate grounded answer from question and context bundle."""
        if not question.strip():
            raise ValueError("question must not be empty.")
        if not isinstance(context_bundle, ContextBundle):
            raise TypeError("context_bundle must be a ContextBundle.")

        started = perf_counter()
        logger.info(
            "Starting answer generation: provider=ollama model_name=%s context_token_count=%s",
            self._llm_client.model_name(),
            context_bundle.token_count,
        )

        if (
            not context_bundle.rendered_context.strip()
            or context_bundle.token_count < self._min_context_tokens
        ):
            logger.warning(
                "Skipping LLM call due to empty/insufficient context: token_count=%s min_context_tokens=%s",
                context_bundle.token_count,
                self._min_context_tokens,
            )
            answer = GeneratedAnswer(
                answer=_FALLBACK_ANSWER,
                confidence=Confidence.NONE,
                sources=[],
                raw_output="",
                parsed_successfully=True,
                metadata={"no_answer_reason": "empty_context"},
            )
            self._last_stats = {
                "provider": "ollama",
                "model_name": self._llm_client.model_name(),
                "prompt_template": self._prompt_renderer.template_name(),
                "prompt_chars": 0,
                "raw_output_chars": 0,
                "rendered_prompt_token_estimate": 0,
                "context_token_count": context_bundle.token_count,
                "context_sources_total": len(context_bundle.sources),
                "context_truncated": context_bundle.truncated,
                "llm_duration_ms": 0,
                "parse_duration_ms": 0,
                "total_duration_ms": int((perf_counter() - started) * 1000),
                "parsed_successfully": answer.parsed_successfully,
                "no_answer_used": True,
                "confidence": CONFIDENCE_NONE,
            }
            return answer

        prompt = self._prompt_renderer.render_answer_prompt(
            question=question,
            context_bundle=context_bundle,
        )
        llm_started = perf_counter()
        raw_output = self._llm_client.generate(prompt)
        llm_duration_ms = int((perf_counter() - llm_started) * 1000)

        parse_started = perf_counter()
        answer = self._output_parser.parse(
            raw_output=raw_output,
            context_bundle=context_bundle,
        )
        parse_duration_ms = int((perf_counter() - parse_started) * 1000)

        total_duration_ms = int((perf_counter() - started) * 1000)
        no_answer_used = answer.confidence == Confidence.NONE
        self._last_stats = {
            "provider": "ollama",
            "model_name": self._llm_client.model_name(),
            "prompt_template": self._prompt_renderer.template_name(),
            "prompt_template_version": self._prompt_renderer.template_version(),
            "rendered_prompt": prompt,
            "prompt_chars": len(prompt),
            "raw_output_chars": len(raw_output),
            "rendered_prompt_token_estimate": len(prompt.split()),
            "context_token_count": context_bundle.token_count,
            "context_sources_total": len(context_bundle.sources),
            "context_truncated": context_bundle.truncated,
            "llm_duration_ms": llm_duration_ms,
            "parse_duration_ms": parse_duration_ms,
            "total_duration_ms": total_duration_ms,
            "parsed_successfully": answer.parsed_successfully,
            "no_answer_used": no_answer_used,
            "confidence": answer.confidence.value,
        }
        logger.info(
            "Finished answer generation: parsed_successfully=%s no_answer_used=%s duration_ms=%s",
            answer.parsed_successfully,
            no_answer_used,
            total_duration_ms,
        )
        return answer


def create_answer_generator(settings: Settings) -> AnswerGenerator:
    """Create configured answer generator from application settings."""
    generation = settings.generation
    if generation.provider != "ollama":
        raise ValueError(
            f"Unsupported generation provider: {generation.provider!r}. Only 'ollama' is supported."
        )

    prompt_renderer = PromptRenderer(
        prompts_dir=Path(generation.prompts_dir),
        answer_template_name=generation.answer_template_name,
        no_answer_template_name=generation.no_answer_template_name,
    )
    llm_client = OllamaClient(
        base_url=generation.base_url,
        model_name=generation.model_name,
        timeout_seconds=generation.timeout_seconds,
        temperature=generation.temperature,
        max_tokens=generation.max_tokens,
        top_p=generation.top_p,
        keep_alive=generation.keep_alive,
    )
    output_parser = OutputParser(
        no_answer_quality_threshold_pct=generation.no_answer_quality_threshold_pct
    )
    return AnswerGenerator(
        prompt_renderer=prompt_renderer,
        llm_client=llm_client,
        output_parser=output_parser,
        min_context_tokens=generation.min_context_tokens,
        warmup_ollama_healthcheck_enabled=settings.api.warmup.ollama_healthcheck_enabled,
        warmup_ollama_generate_enabled=settings.api.warmup.ollama_generate_enabled,
        warmup_ollama_generate_prompt=settings.api.warmup.ollama_generate_prompt,
        warmup_ollama_generate_max_tokens=settings.api.warmup.ollama_generate_max_tokens,
    )
