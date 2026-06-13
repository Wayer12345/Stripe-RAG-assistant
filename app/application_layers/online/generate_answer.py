"""Online generation layer: context bundle -> grounded generated answer."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from app.domain.models.answer import GeneratedAnswer
from app.domain.models.context import ContextBundle
from app.infrastructure.generation import create_answer_generator
from app.infrastructure.storage.manifest_store import write_json_payload
from app.infrastructure.storage.trace_loader import load_context_bundle_from_input_path
from app.utils.config import load_settings, resolve_config_dir_and_path, to_optional_path
from app.utils.constants import STAGE_GENERATE_ANSWER
from app.utils.logging import get_logger, setup_logging
from app.utils.timing import finish_timed_run, start_timed_run

logger = get_logger(__name__)


@dataclass(frozen=True)
class GenerateAnswerResult:
    """Runtime result for one online generation execution."""

    run_id: str
    question: str
    generated_answer: GeneratedAnswer
    confidence: str
    sources_total: int
    parsed_successfully: bool
    provider: str
    model_name: str
    prompt_template: str
    context_token_count: int
    context_sources_total: int
    context_truncated: bool
    trace_path: Path | None
    duration_ms: int


class GenerateAnswerLayer:
    """Orchestrates answer generation for a single question and context bundle."""

    def __init__(
        self,
        *,
        question: str,
        context_bundle: ContextBundle,
        config_path: Path | str = Path("configs/config.yaml"),
        trace_path: Path | str | None = None,
        write_trace: bool | None = None,
        answer_generator: Any | None = None,
    ) -> None:
        if not question.strip():
            raise ValueError("question must not be empty.")
        if not isinstance(context_bundle, ContextBundle):
            raise TypeError("context_bundle must be a ContextBundle.")

        self._question = question.strip()
        self._context_bundle = context_bundle
        self._config_path = Path(config_path)
        self._trace_path_override = to_optional_path(trace_path)
        self._write_trace_override = write_trace
        self._answer_generator = answer_generator

    @property
    def answer_generator(self) -> Any | None:
        """Return answer generator used by this layer execution."""
        return self._answer_generator

    @classmethod
    def warmup(
        cls,
        *,
        config_path: Path | str = Path("configs/config.yaml"),
        answer_generator: Any | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Warm answer generator and return component with warmup payload."""
        config_dir, _ = resolve_config_dir_and_path(Path(config_path))
        settings = load_settings(config_dir)
        setup_logging(settings)

        resolved_answer_generator = (
            answer_generator if answer_generator is not None else create_answer_generator(settings)
        )
        warmup = getattr(resolved_answer_generator, "warmup", None)
        warmup_payload: dict[str, Any] = {}
        if callable(warmup):
            warmup_payload = cast(dict[str, Any], warmup())
        return resolved_answer_generator, warmup_payload

    def run(self) -> GenerateAnswerResult:
        """Run generation and optionally write stage trace."""
        config_dir, resolved_config_path = resolve_config_dir_and_path(self._config_path)
        settings = load_settings(config_dir)
        setup_logging(settings)

        generation = settings.generation
        write_trace = (
            self._write_trace_override
            if self._write_trace_override is not None
            else generation.write_trace
        )
        trace_dir = generation.trace_dir
        preview_chars = generation.text_preview_chars

        timed_run = start_timed_run(STAGE_GENERATE_ANSWER)
        trace_path = (
            self._trace_path_override
            if self._trace_path_override is not None
            else trace_dir / f"{timed_run.run_id}_generation.json"
        )

        logger.info(
            "Starting generate-answer layer: run_id=%s provider=%s model_name=%s context_token_count=%s context_sources_total=%s",
            timed_run.run_id,
            generation.provider,
            generation.model_name,
            self._context_bundle.token_count,
            len(self._context_bundle.sources),
        )

        generator = self._answer_generator or create_answer_generator(settings)
        self._answer_generator = generator
        try:
            generated_answer = generator.generate(
                question=self._question,
                context_bundle=self._context_bundle,
            )
        except Exception:
            logger.exception("Generator failure in GenerateAnswerLayer.")
            raise

        stats = generator.last_stats() if hasattr(generator, "last_stats") else {}
        finished_at, duration_ms = finish_timed_run(timed_run)

        if generated_answer.confidence.value == "none":
            logger.warning("Generation produced no-answer result.")
        if not generated_answer.sources:
            logger.warning("Generation produced zero valid sources.")

        trace_path_value: Path | None = trace_path
        if write_trace:
            payload = self._build_trace_payload(
                run_id=timed_run.run_id,
                question=self._question,
                started_at=timed_run.started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
                config_path=resolved_config_path,
                generation_settings=generation,
                context_bundle=self._context_bundle,
                generated_answer=generated_answer,
                stats=stats,
                preview_chars=preview_chars,
            )

            try:
                write_json_payload(trace_path, payload)
                logger.info("Wrote generation trace: path=%s", trace_path)
            except Exception:
                logger.exception("Failed to write generation trace: path=%s", trace_path)
                raise
        else:
            logger.warning("Trace writing disabled for generation run.")
            trace_path_value = None

        result = GenerateAnswerResult(
            run_id=timed_run.run_id,
            question=self._question,
            generated_answer=generated_answer,
            confidence=generated_answer.confidence.value,
            sources_total=len(generated_answer.sources),
            parsed_successfully=generated_answer.parsed_successfully,
            provider=str(stats.get("provider", generation.provider)),
            model_name=str(stats.get("model_name", generation.model_name)),
            prompt_template=str(stats.get("prompt_template", generation.answer_template_name)),
            context_token_count=self._context_bundle.token_count,
            context_sources_total=len(self._context_bundle.sources),
            context_truncated=self._context_bundle.truncated,
            trace_path=trace_path_value,
            duration_ms=duration_ms,
        )
        logger.info(
            "Finished generate-answer layer: run_id=%s confidence=%s parsed_successfully=%s duration_ms=%s trace_path=%s",
            result.run_id,
            result.confidence,
            result.parsed_successfully,
            result.duration_ms,
            result.trace_path,
        )
        return result

    @staticmethod
    def _build_trace_payload(
        *,
        run_id: str,
        question: str,
        started_at: str,
        finished_at: str,
        duration_ms: int,
        config_path: Path,
        generation_settings: Any,
        context_bundle: ContextBundle,
        generated_answer: GeneratedAnswer,
        stats: dict[str, Any],
        preview_chars: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": run_id,
            "stage": STAGE_GENERATE_ANSWER,
            "question": question,
            "provider": str(stats.get("provider", generation_settings.provider)),
            "model_name": str(stats.get("model_name", generation_settings.model_name)),
            "prompt_template": str(
                stats.get("prompt_template", generation_settings.answer_template_name)
            ),
            "prompt_template_version": str(
                stats.get(
                    "prompt_template_version",
                    Path(generation_settings.answer_template_name).stem,
                )
            ),
            "context_token_count": int(
                stats.get("context_token_count", context_bundle.token_count)
            ),
            "context_sources_total": int(
                stats.get("context_sources_total", len(context_bundle.sources))
            ),
            "context_truncated": bool(stats.get("context_truncated", context_bundle.truncated)),
            "confidence": generated_answer.confidence.value,
            "sources_total": len(generated_answer.sources),
            "parsed_successfully": generated_answer.parsed_successfully,
            "no_answer_used": bool(
                stats.get("no_answer_used", generated_answer.confidence.value == "none")
            ),
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": duration_ms,
            "llm_duration_ms": int(stats.get("llm_duration_ms", 0)),
            "parse_duration_ms": int(stats.get("parse_duration_ms", 0)),
            "config_path": str(config_path),
            "answer_preview": generated_answer.answer[:preview_chars],
            "raw_output_preview": generated_answer.raw_output[:preview_chars],
            "sources": [source.model_dump(mode="json") for source in generated_answer.sources],
        }
        if generation_settings.include_full_prompt_in_trace and "rendered_prompt" in stats:
            payload["rendered_prompt"] = stats["rendered_prompt"]
        if generation_settings.include_raw_output_in_trace:
            payload["raw_output"] = generated_answer.raw_output
        if generation_settings.include_full_answer_in_trace:
            payload["answer"] = generated_answer.answer
        return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run online answer generation from a serialized context bundle JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--trace-path", type=Path, default=None)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--print-answer", action="store_true")
    parser.add_argument("--print-raw-output", action="store_true")
    return parser


def main() -> None:
    setup_logging()

    args = _build_arg_parser().parse_args()

    try:
        context_bundle = load_context_bundle_from_input_path(args.input_path)
        layer = GenerateAnswerLayer(
            question=args.question,
            context_bundle=context_bundle,
            config_path=args.config,
            trace_path=args.trace_path,
            write_trace=False if args.no_trace else None,
        )

        result = layer.run()
    except Exception:
        logger.exception("GenerateAnswerLayer failed")
        sys.exit(1)

    logger.info(
        "Generation result: confidence=%s parsed_successfully=%s sources_total=%s "
        "provider=%r model_name=%r context_token_count=%s duration_ms=%s trace_path=%s",
        result.confidence,
        result.parsed_successfully,
        result.sources_total,
        result.provider,
        result.model_name,
        result.context_token_count,
        result.duration_ms,
        result.trace_path,
    )
    if args.print_answer:
        print(result.generated_answer.answer)
    if args.print_raw_output:
        print(result.generated_answer.raw_output)


if __name__ == "__main__":
    main()
