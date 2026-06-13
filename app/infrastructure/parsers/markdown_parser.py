"""Parser that converts raw Markdown byte payloads into normalized Document objects."""

import re
from datetime import UTC, datetime
from pathlib import Path

from app.domain.interfaces.document_loader import RawDocument
from app.domain.models.document import Document, DocumentProcessingStage
from app.utils.hashing import sha256_text
from app.utils.ids import make_document_id

_SUPPORTED_SOURCE_TYPES: frozenset[str] = frozenset({"md", "markdown"})
_SUPPORTED_MIME_TYPES: frozenset[str] = frozenset({"text/markdown", "text/x-markdown"})

# ATX-style H1: a line that starts with a single `#` followed by a space.
_H1_RE = re.compile(r"^#\s+(.+)", re.MULTILINE)


def _extract_h1_title(text: str) -> str | None:
    """Return the text of the first ATX H1 heading, or ``None``."""
    m = _H1_RE.search(text)
    return m.group(1).strip() if m else None


class MarkdownParser:
    """Parses raw Markdown payloads into normalized Document objects.

    Implements the :class:`~app.domain.interfaces.parser.Parser` Protocol.

    The raw Markdown text is preserved as-is; it is not converted to HTML.
    UTF-8 decoding is attempted first; failures fall back to replacement
    characters.  Empty or whitespace-only content raises :class:`ValueError`.
    """

    def supports(self, source_type: str, mime_type: str | None = None) -> bool:
        """Return ``True`` for Markdown source types and MIME types."""
        return source_type.lower() in _SUPPORTED_SOURCE_TYPES or (
            mime_type is not None and mime_type.lower() in _SUPPORTED_MIME_TYPES
        )

    def supported_source_types(self) -> set[str]:
        """Return the set of source types this parser declares support for."""
        return set(_SUPPORTED_SOURCE_TYPES)

    def parse(self, raw_document: RawDocument) -> list[Document]:
        """Parse one raw Markdown payload into a single Document.

        The Markdown structure (headings, lists, code blocks) is preserved
        verbatim in ``text``; no conversion to HTML is performed.

        Args:
            raw_document: Raw payload produced by a file loader.

        Returns:
            Single-element list with the parsed Document.

        Raises:
            ValueError: If the decoded text is empty or whitespace-only.
        """
        try:
            text = raw_document.content.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_document.content.decode("utf-8", errors="replace")

        text = text.replace("\r\n", "\n").replace("\r", "\n")

        if not text.strip():
            raise ValueError(
                f"Markdown document is empty or whitespace-only: "
                f"source_path={raw_document.source_path!r}"
            )

        content_hash = sha256_text(text)
        source_identity = raw_document.source_path or raw_document.source_name or "unknown"
        doc_id = make_document_id(source_identity, content_hash)

        h1_title = _extract_h1_title(text)
        if h1_title:
            title = h1_title
        elif raw_document.source_name:
            title = raw_document.source_name
        elif raw_document.source_path:
            title = Path(raw_document.source_path).name
        else:
            title = "Untitled Markdown document"

        return [
            Document(
                id=doc_id,
                source_type=raw_document.source_type,
                source_path=raw_document.source_path,
                source_name=raw_document.source_name,
                source_mime_type=raw_document.mime_type,
                title=title,
                text=text,
                content_hash=content_hash,
                created_at=datetime.now(UTC),
                processing_stage=DocumentProcessingStage.PARSED,
                metadata=dict(raw_document.metadata),
            )
        ]
