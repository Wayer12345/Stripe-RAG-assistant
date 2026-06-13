"""Parser that converts raw CSV byte payloads into normalized Document objects."""

import csv
import io
from datetime import UTC, datetime
from pathlib import Path

from app.domain.interfaces.document_loader import RawDocument
from app.domain.models.document import Document, DocumentProcessingStage
from app.utils.hashing import sha256_text
from app.utils.ids import make_document_id

_SUPPORTED_SOURCE_TYPES: frozenset[str] = frozenset({"csv"})
_SUPPORTED_MIME_TYPES: frozenset[str] = frozenset({"text/csv"})


class CsvParser:
    """Parses raw CSV payloads into a single normalized Document object.

    Implements the :class:`~app.domain.interfaces.parser.Parser` Protocol.

    Rows are rendered as readable header-prefixed lines:

    ``header1: value1 | header2: value2``

    When no headers are present, values are joined with `` | ``.
    Empty CSVs or files with no data rows raise :class:`ValueError`.
    Pandas is not used; the Python standard library ``csv`` module is used.
    """

    def supports(self, source_type: str, mime_type: str | None = None) -> bool:
        """Return ``True`` for CSV source types and MIME types."""
        return source_type.lower() in _SUPPORTED_SOURCE_TYPES or (
            mime_type is not None and mime_type.lower() in _SUPPORTED_MIME_TYPES
        )

    def supported_source_types(self) -> set[str]:
        """Return the set of source types this parser declares support for."""
        return set(_SUPPORTED_SOURCE_TYPES)

    def parse(self, raw_document: RawDocument) -> list[Document]:
        """Parse one raw CSV payload into a single Document.

        Args:
            raw_document: Raw payload produced by a file loader.

        Returns:
            Single-element list with the parsed Document.

        Raises:
            ValueError: If the CSV has no data rows or cannot be decoded.
        """
        try:
            decoded = raw_document.content.decode("utf-8")
        except UnicodeDecodeError:
            decoded = raw_document.content.decode("utf-8", errors="replace")

        decoded = decoded.replace("\r\n", "\n").replace("\r", "\n")

        reader = csv.reader(io.StringIO(decoded))
        try:
            rows = list(reader)
        except csv.Error as exc:
            raise ValueError(
                f"Cannot parse CSV: source_path={raw_document.source_path!r} — {exc}"
            ) from exc

        if not rows:
            raise ValueError(f"CSV is empty: source_path={raw_document.source_path!r}")

        # Heuristic: treat the first row as headers when it looks like one.
        # A header row has non-empty string cells and no row looks more like
        # a data row above it.  We always treat the first row as headers when
        # the file has more than one row; otherwise we treat the single row
        # as data.
        headers: list[str] | None = None
        data_rows: list[list[str]] = rows

        if len(rows) > 1:
            headers = [h.strip() for h in rows[0]]
            data_rows = rows[1:]

        if not data_rows:
            raise ValueError(f"CSV has no data rows: source_path={raw_document.source_path!r}")

        lines: list[str] = []
        for row in data_rows:
            if headers:
                pairs = [
                    f"{h}: {v.strip()}" for h, v in zip(headers, row, strict=False) if v.strip()
                ]
                if pairs:
                    lines.append(" | ".join(pairs))
            else:
                values = [v.strip() for v in row if v.strip()]
                if values:
                    lines.append(" | ".join(values))

        if not lines:
            raise ValueError(f"CSV has no usable content: source_path={raw_document.source_path!r}")

        text = "\n".join(lines)
        content_hash = sha256_text(text)
        source_identity = raw_document.source_path or raw_document.source_name or "unknown"
        doc_id = make_document_id(source_identity, content_hash)

        title = (
            raw_document.source_name
            or (Path(raw_document.source_path).name if raw_document.source_path else None)
            or "Untitled CSV document"
        )

        metadata = dict(raw_document.metadata)
        metadata["row_count"] = len(data_rows)
        if headers:
            metadata["headers"] = headers

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
