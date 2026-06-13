"""Parser that converts raw HTML byte payloads into normalized Document objects."""

from datetime import UTC, datetime
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from app.domain.interfaces.document_loader import RawDocument
from app.domain.models.document import Document, DocumentProcessingStage
from app.utils.hashing import sha256_text
from app.utils.ids import make_document_id

_SUPPORTED_SOURCE_TYPES: frozenset[str] = frozenset({"html", "htm"})
_SUPPORTED_MIME_TYPES: frozenset[str] = frozenset({"text/html"})

# Tags whose content is never reader-visible.
_NOISE_TAGS: frozenset[str] = frozenset({"script", "style", "noscript", "template"})


def _get_tag_text(tag: Tag | None) -> str | None:
    """Return stripped inner text of *tag*, or ``None`` if absent or empty."""
    if tag is None:
        return None
    text = tag.get_text(strip=True)
    return text if text else None


class HtmlParser:
    """Parses raw HTML payloads into normalized Document objects.

    Implements the :class:`~app.domain.interfaces.parser.Parser` Protocol.

    Non-content technical elements (``script``, ``style``, ``noscript``,
    ``template``) are removed before text extraction.  Full boilerplate
    cleaning (nav bars, footers, cookie banners) is left for the cleaning
    layer.  Empty or whitespace-only extracted text raises :class:`ValueError`.
    """

    def supports(self, source_type: str, mime_type: str | None = None) -> bool:
        """Return ``True`` for HTML source types and MIME types."""
        return source_type.lower() in _SUPPORTED_SOURCE_TYPES or (
            mime_type is not None and mime_type.lower() in _SUPPORTED_MIME_TYPES
        )

    def supported_source_types(self) -> set[str]:
        """Return the set of source types this parser declares support for."""
        return set(_SUPPORTED_SOURCE_TYPES)

    def parse(self, raw_document: RawDocument) -> list[Document]:
        """Parse one raw HTML payload into a single Document.

        Args:
            raw_document: Raw payload produced by a file loader.

        Returns:
            Single-element list with the parsed Document.

        Raises:
            ValueError: If the extracted text body is empty or whitespace-only.
        """
        try:
            html_str = raw_document.content.decode("utf-8")
        except UnicodeDecodeError:
            html_str = raw_document.content.decode("utf-8", errors="replace")

        soup = BeautifulSoup(html_str, "lxml")

        for tag in soup.find_all(_NOISE_TAGS):
            tag.decompose()

        # Title resolution: <title> → first <h1> → source_name → path → default
        page_title = (
            _get_tag_text(soup.find("title"))
            or _get_tag_text(soup.find("h1"))
            or raw_document.source_name
            or (Path(raw_document.source_path).name if raw_document.source_path else None)
            or "Untitled HTML document"
        )

        body = soup.find("body") or soup
        text = body.get_text(separator="\n")
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        if not text.strip():
            raise ValueError(
                f"HTML document yields no usable text: source_path={raw_document.source_path!r}"
            )

        content_hash = sha256_text(text)
        source_identity = raw_document.source_path or raw_document.source_name or "unknown"
        doc_id = make_document_id(source_identity, content_hash)

        return [
            Document(
                id=doc_id,
                source_type=raw_document.source_type,
                source_path=raw_document.source_path,
                source_name=raw_document.source_name,
                source_mime_type=raw_document.mime_type,
                title=page_title,
                text=text,
                content_hash=content_hash,
                created_at=datetime.now(UTC),
                processing_stage=DocumentProcessingStage.PARSED,
                metadata=dict(raw_document.metadata),
            )
        ]
