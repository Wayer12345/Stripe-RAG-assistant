"""Domain interface contract for evaluation components."""

from typing import Any, Protocol

from app.domain.models.answer import GeneratedAnswer
from app.domain.models.eval_case import EvalCase


class Evaluator(Protocol):
    """Evaluates predictions over eval cases and returns metric artifacts."""

    def evaluate(
        self,
        cases: list[EvalCase],
        predictions: list[GeneratedAnswer],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return evaluation outputs for the provided cases and predictions."""
        ...

