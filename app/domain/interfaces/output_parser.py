"""Domain interface contract for parsing raw generation output."""

from typing import Protocol

from app.domain.models.answer import GeneratedAnswer
from app.domain.models.source import Source


class OutputParser(Protocol):
    """Parses raw model output into a validated structured answer."""

    def parse(self, raw_output: str, *, available_sources: list[Source]) -> GeneratedAnswer:
        """Parse and validate raw output using available context sources."""
        ...

