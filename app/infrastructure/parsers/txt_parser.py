"""Parser that converts raw TXT byte payloads into normalized Document objects."""

from datetime import UTC, datetime
from pathlib import Path

from app.domain.interfaces.document_loader import RawDocument
from app.domain.models.document import Document, DocumentProcessingStage
from app.utils.hashing import sha256_text
from app.utils.ids import make_document_id

_SUPPORTED_SOURCE_TYPES: frozenset[str] = frozenset({"txt"})
_SUPPORTED_MIME_TYPES: frozenset[str] = frozenset({"text/plain"})


class TxtParser:
    """Parses raw TXT payloads into normalized :class:`~app.domain.models.document.Document` objects.

    Implements the :class:`~app.domain.interfaces.parser.Parser` Protocol.

    Decoding is attempted with UTF-8 first; if that fails, the content is
    decoded with replacement characters so parsing always succeeds for files
    that are valid Latin-1 or contain isolated non-UTF-8 bytes.  Empty or
    whitespace-only content raises :class:`ValueError`.
    """

    def supports(self, source_type: str, mime_type: str | None = None) -> bool:
        """Return ``True`` for ``source_type="txt"`` or ``mime_type="text/plain"``."""
        return source_type.lower() in _SUPPORTED_SOURCE_TYPES or (
            mime_type is not None and mime_type.lower() in _SUPPORTED_MIME_TYPES
        )

    def supported_source_types(self) -> set[str]:
        """Return the set of source types this parser declares support for."""
        return set(_SUPPORTED_SOURCE_TYPES)

    def parse(self, raw_document: RawDocument) -> list[Document]:
        """Parse one raw TXT payload into a single Document.

        Args:
            raw_document: Raw payload produced by a file loader.

        Returns:
            A single-element list containing the parsed
            :class:`~app.domain.models.document.Document`.

        Raises:
            ValueError: If the decoded text is empty or whitespace-only.
        """
        try:
            text = raw_document.content.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_document.content.decode("utf-8", errors="replace")

        # Normalise line endings for stable hashing and downstream processing.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        if not text.strip():
            raise ValueError(
                f"TXT document is empty or whitespace-only: "
                f"source_path={raw_document.source_path!r}"
            )

        content_hash = sha256_text(text)

        source_identity = raw_document.source_path or raw_document.source_name or "unknown"
        doc_id = make_document_id(source_identity, content_hash)

        if raw_document.source_name:
            title = raw_document.source_name
        elif raw_document.source_path:
            title = Path(raw_document.source_path).name
        else:
            title = "Untitled TXT document"

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
