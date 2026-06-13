"""Domain interface contract for parsing raw documents."""

from typing import Protocol

from app.domain.interfaces.document_loader import RawDocument
from app.domain.models.document import Document


class Parser(Protocol):
    """Parses one raw source payload into normalized document objects."""

    def supports(self, source_type: str, mime_type: str | None = None) -> bool:
        """Return whether this parser supports the source payload type."""
        ...

    def parse(self, raw_document: RawDocument) -> list[Document]:
        """Parse one raw payload into one or more normalized documents."""
        ...

