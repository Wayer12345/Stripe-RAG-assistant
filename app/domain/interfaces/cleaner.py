"""Domain interface contract for cleaning parsed documents."""

from typing import Protocol

from app.domain.models.document import Document


class Cleaner(Protocol):
    """Cleans one parsed document while preserving domain identity."""

    def clean(self, document: Document) -> Document:
        """Return a cleaned document derived from the provided input."""
        ...

