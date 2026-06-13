"""Prompt rendering for generation."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

from app.domain.models.context import ContextBundle
from app.utils.logging import get_logger

logger = get_logger(__name__)


class PromptRenderer:
    """Renders generation prompts from Jinja templates."""

    def __init__(
        self,
        *,
        prompts_dir: Path | str,
        answer_template_name: str = "answer_prompt_v1.jinja",
        no_answer_template_name: str = "no_answer_prompt_v1.jinja",
    ) -> None:
        prompts_path = Path(prompts_dir)
        if not str(prompts_path).strip():
            raise ValueError("prompts_dir must not be empty.")
        if not answer_template_name.strip():
            raise ValueError("answer_template_name must not be empty.")
        if not no_answer_template_name.strip():
            raise ValueError("no_answer_template_name must not be empty.")
        if not prompts_path.exists() or not prompts_path.is_dir():
            raise FileNotFoundError(
                f"prompts_dir does not exist or is not a directory: {prompts_path}"
            )

        self._prompts_dir = prompts_path
        self._answer_template_name = answer_template_name.strip()
        self._no_answer_template_name = no_answer_template_name.strip()
        self._environment = Environment(
            loader=FileSystemLoader(str(self._prompts_dir)),
            autoescape=False,
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            auto_reload=False,
        )
        self._validate_template_exists(self._answer_template_name)
        self._validate_template_exists(self._no_answer_template_name)

    def template_name(self) -> str:
        """Return answer template file name."""
        return self._answer_template_name

    def template_version(self) -> str:
        """Return current prompt template version label."""
        return Path(self._answer_template_name).stem

    def render_answer_prompt(
        self,
        *,
        question: str,
        context_bundle: ContextBundle,
    ) -> str:
        """Render answer prompt for model generation."""
        if not question.strip():
            raise ValueError("question must not be empty.")
        if not isinstance(context_bundle, ContextBundle):
            raise TypeError("context_bundle must be a ContextBundle.")

        logger.info(
            "Starting prompt rendering: template=%s context_token_count=%s",
            self._answer_template_name,
            context_bundle.token_count,
        )
        template = self._environment.get_template(self._answer_template_name)
        prompt = template.render(
            question=question.strip(),
            rendered_context=context_bundle.rendered_context,
            context_sources=[source.model_dump(mode="json") for source in context_bundle.sources],
            context_token_count=context_bundle.token_count,
            context_token_budget=context_bundle.token_budget,
            context_truncated=context_bundle.truncated,
            output_schema_instructions=self._output_schema_instructions(),
            no_answer_policy=self._no_answer_policy(),
        )
        if not prompt.strip():
            raise RuntimeError("Rendered answer prompt is empty.")
        logger.info(
            "Finished prompt rendering: template=%s prompt_chars=%s",
            self._answer_template_name,
            len(prompt),
        )
        return prompt

    def render_no_answer_prompt(self, *, question: str, context_bundle: ContextBundle) -> str:
        """Render no-answer template (optional helper for diagnostics)."""
        if not question.strip():
            raise ValueError("question must not be empty.")
        if not isinstance(context_bundle, ContextBundle):
            raise TypeError("context_bundle must be a ContextBundle.")
        template = self._environment.get_template(self._no_answer_template_name)
        output = template.render(
            question=question.strip(),
            rendered_context=context_bundle.rendered_context,
            context_sources=[source.model_dump(mode="json") for source in context_bundle.sources],
        )
        if not output.strip():
            raise RuntimeError("Rendered no-answer prompt is empty.")
        return output

    def _validate_template_exists(self, template_name: str) -> None:
        try:
            self._environment.get_template(template_name)
        except TemplateNotFound as err:
            raise FileNotFoundError(f"Prompt template not found: {template_name}") from err

    @staticmethod
    def _output_schema_instructions() -> dict[str, object]:
        return {
            "type": "object",
            "required": ["confidence", "answer", "sources"],
            "properties": {
                "confidence": {"type": "string", "enum": ["high", "medium", "low", "none"]},
                "answer": {"type": "string"},
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "title",
                            "url",
                            "section",
                            "chunk_id",
                            "document_id",
                            "support_score",
                        ],
                    },
                },
            },
        }

    @staticmethod
    def _no_answer_policy() -> dict[str, str]:
        return {
            "confidence": "none",
            "answer": (
                "I don't have enough information in the indexed Stripe Guides sources "
                "to answer this reliably."
            ),
            "sources": "[]",
        }
