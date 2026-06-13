"""Domain interface contract for document chunking."""

from typing import Protocol

from app.domain.models.chunk import Chunk
from app.domain.models.document import Document


class Chunker(Protocol):
    """Splits one cleaned document into retrieval-ready chunks."""

    def chunk(self, document: Document) -> list[Chunk]:
        """Return chunks derived from the provided document."""
        ...

