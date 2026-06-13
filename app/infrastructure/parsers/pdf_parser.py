"""Parser that converts raw PDF byte payloads into normalized Document objects."""

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.domain.interfaces.document_loader import RawDocument
from app.domain.models.document import Document, DocumentProcessingStage
from app.utils.hashing import sha256_text
from app.utils.ids import make_document_id

_SUPPORTED_SOURCE_TYPES: frozenset[str] = frozenset({"pdf"})
_SUPPORTED_MIME_TYPES: frozenset[str] = frozenset({"application/pdf"})


class PdfParser:
    """Parses raw PDF payloads into normalized Document objects.

    Implements the :class:`~app.domain.interfaces.parser.Parser` Protocol.

    Pages are concatenated in order with human-readable ``[Page N]`` boundary
    markers.  ``page_count`` is stored in document metadata.  No OCR is
    performed; only embedded text layers are extracted.
    """

    def supports(self, source_type: str, mime_type: str | None = None) -> bool:
        """Return ``True`` for PDF source types and MIME types."""
        return source_type.lower() in _SUPPORTED_SOURCE_TYPES or (
            mime_type is not None and mime_type.lower() in _SUPPORTED_MIME_TYPES
        )

    def supported_source_types(self) -> set[str]:
        """Return the set of source types this parser declares support for."""
        return set(_SUPPORTED_SOURCE_TYPES)

    def parse(self, raw_document: RawDocument) -> list[Document]:
        """Parse one raw PDF payload into a single Document.

        Pages are joined with ``[Page N]`` markers.  PDF metadata title is
        used when available; otherwise falls back through source_name, path
        filename, and a default string.

        Args:
            raw_document: Raw payload produced by a file loader.

        Returns:
            Single-element list with the parsed Document.

        Raises:
            ValueError: If the PDF cannot be read or yields no extractable text.
        """
        try:
            reader = PdfReader(BytesIO(raw_document.content))
        except PdfReadError as exc:
            raise ValueError(f"Cannot read PDF: source_path={raw_document.source_path!r}") from exc
        except Exception as exc:
            raise ValueError(
                f"Unexpected error reading PDF: source_path={raw_document.source_path!r}"
            ) from exc

        page_count = len(reader.pages)
        page_chunks: list[str] = []
        for i, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_chunks.append(f"[Page {i}]\n{page_text}")

        text = "\n".join(page_chunks)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        if not text.strip():
            raise ValueError(
                f"PDF yields no extractable text: source_path={raw_document.source_path!r}"
            )

        content_hash = sha256_text(text)
        source_identity = raw_document.source_path or raw_document.source_name or "unknown"
        doc_id = make_document_id(source_identity, content_hash)

        # Title: PDF metadata → source_name → path filename → default
        pdf_title: str | None = None
        if reader.metadata and reader.metadata.title:
            pdf_title = reader.metadata.title.strip() or None

        title = (
            pdf_title
            or raw_document.source_name
            or (Path(raw_document.source_path).name if raw_document.source_path else None)
            or "Untitled PDF document"
        )

        metadata = dict(raw_document.metadata)
        metadata["page_count"] = page_count
        metadata["parser"] = "pypdf"

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
                metadata=metadata,
            )
        ]
