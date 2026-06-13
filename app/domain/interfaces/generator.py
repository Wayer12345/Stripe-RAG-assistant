"""Domain interface contract for grounded answer generation."""

from typing import Protocol

from app.domain.models.answer import GeneratedAnswer
from app.domain.models.context import ContextBundle


class Generator(Protocol):
    """Generates a structured answer from query and prepared context."""

    def generate(self, query: str, context: ContextBundle) -> GeneratedAnswer:
        """Generate a grounded answer for the query using provided context."""
        ...

