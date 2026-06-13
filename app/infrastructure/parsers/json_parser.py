"""Parser that converts raw JSON and JSONL byte payloads into normalized Document objects."""

import json
from datetime import UTC, datetime
from pathlib import Path

from app.domain.interfaces.document_loader import RawDocument
from app.domain.models.document import Document, DocumentProcessingStage
from app.utils.hashing import sha256_text
from app.utils.ids import make_document_id

_SUPPORTED_SOURCE_TYPES: frozenset[str] = frozenset({"json", "jsonl"})
_SUPPORTED_MIME_TYPES: frozenset[str] = frozenset(
    {"application/json", "application/x-ndjson", "application/jsonl"}
)

# Candidate field names scanned in order for extracting text content.
_TEXT_FIELDS: tuple[str, ...] = (
    "text",
    "content",
    "body",
    "markdown",
    "html",
    "answer",
    "description",
)
_TITLE_FIELDS: tuple[str, ...] = ("title", "name", "heading", "question")


def _pick_text(obj: dict) -> str | None:
    """Return the value of the first recognised text field in *obj*, or ``None``."""
    for key in _TEXT_FIELDS:
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def _pick_title(obj: dict) -> str | None:
    """Return the value of the first recognised title field in *obj*, or ``None``."""
    for key in _TITLE_FIELDS:
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _obj_to_text(obj: object) -> str:
    """Serialise *obj* as pretty-printed JSON text."""
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _build_document(
    raw: RawDocument,
    text: str,
    title: str,
    extra_metadata: dict | None = None,
    source_identity_suffix: str = "",
) -> Document:
    """Construct a validated Document from extracted text and title."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    content_hash = sha256_text(text)
    base_identity = raw.source_path or raw.source_name or "unknown"
    source_identity = (
        f"{base_identity}{source_identity_suffix}" if source_identity_suffix else base_identity
    )
    doc_id = make_document_id(source_identity, content_hash)
    metadata = dict(raw.metadata)
    if extra_metadata:
        metadata.update(extra_metadata)
    return Document(
        id=doc_id,
        source_type=raw.source_type,
        source_path=raw.source_path,
        source_name=raw.source_name,
        source_mime_type=raw.mime_type,
        title=title,
        text=text,
        content_hash=content_hash,
        created_at=datetime.now(UTC),
        processing_stage=DocumentProcessingStage.PARSED,
        metadata=metadata,
    )


def _default_title(raw: RawDocument, fallback: str) -> str:
    return raw.source_name or (Path(raw.source_path).name if raw.source_path else None) or fallback


class JsonParser:
    """Parses raw JSON and JSONL payloads into normalized Document objects.

    Implements the :class:`~app.domain.interfaces.parser.Parser` Protocol.

    **JSON mode** (``source_type == "json"``)

    - A JSON array whose elements are objects produces one Document per item.
    - A JSON object with a recognised text field uses that field as ``text``.
    - Any other value is serialized to pretty-printed JSON as a fallback.

    **JSONL mode** (``source_type == "jsonl"`` or NDJSON MIME type)

    - Each non-empty line is parsed independently.
    - One Document is produced per valid line.
    - Invalid lines raise :class:`ValueError`.
    """

    def supports(self, source_type: str, mime_type: str | None = None) -> bool:
        """Return ``True`` for JSON/JSONL source types and MIME types."""
        return source_type.lower() in _SUPPORTED_SOURCE_TYPES or (
            mime_type is not None and mime_type.lower() in _SUPPORTED_MIME_TYPES
        )

    def supported_source_types(self) -> set[str]:
        """Return the set of source types this parser declares support for."""
        return set(_SUPPORTED_SOURCE_TYPES)

    def parse(self, raw_document: RawDocument) -> list[Document]:
        """Parse one raw JSON or JSONL payload into one or more Documents.

        Args:
            raw_document: Raw payload produced by a file loader.

        Returns:
            List of parsed Documents (one per JSON item or JSONL line).

        Raises:
            ValueError: If the content is malformed, empty, or yields no
                usable text.
        """
        is_jsonl = raw_document.source_type.lower() == "jsonl" or (
            raw_document.mime_type is not None
            and raw_document.mime_type.lower() in {"application/x-ndjson", "application/jsonl"}
        )

        try:
            decoded = raw_document.content.decode("utf-8")
        except UnicodeDecodeError:
            decoded = raw_document.content.decode("utf-8", errors="replace")

        if is_jsonl:
            return self._parse_jsonl(raw_document, decoded)
        return self._parse_json(raw_document, decoded)

    def _parse_json(self, raw: RawDocument, text: str) -> list[Document]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: source_path={raw.source_path!r} — {exc}") from exc

        if isinstance(data, list):
            return self._parse_json_array(raw, data)

        if isinstance(data, dict):
            return self._parse_json_object(raw, data, index=None)

        # Scalar value: serialize as text
        obj_text = _obj_to_text(data)
        title = _default_title(raw, "Untitled JSON document")
        return [_build_document(raw, obj_text, title)]

    def _parse_json_array(self, raw: RawDocument, items: list) -> list[Document]:
        if not items:
            raise ValueError(f"JSON array is empty: source_path={raw.source_path!r}")
        docs: list[Document] = []
        for i, item in enumerate(items):
            docs.extend(self._parse_json_object(raw, item, index=i))
        return docs

    def _parse_json_object(
        self, raw: RawDocument, obj: object, index: int | None
    ) -> list[Document]:
        suffix = f"[{index}]" if index is not None else ""
        extra: dict = {}
        if index is not None:
            extra["item_index"] = index

        if not isinstance(obj, dict):
            obj_text = _obj_to_text(obj)
            title = _default_title(raw, "Untitled JSON document")
            return [_build_document(raw, obj_text, title, extra, suffix)]

        picked_text = _pick_text(obj)
        text = picked_text or _obj_to_text(obj)
        title = _pick_title(obj) or _default_title(raw, "Untitled JSON document")
        return [_build_document(raw, text, title, extra, suffix)]

    def _parse_jsonl(self, raw: RawDocument, content: str) -> list[Document]:
        lines = content.splitlines()
        docs: list[Document] = []
        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_num}: source_path={raw.source_path!r} — {exc}"
                ) from exc

            suffix = f"[line:{line_num}]"
            extra: dict = {"line_number": line_num}

            if isinstance(obj, dict):
                picked_text = _pick_text(obj)
                text = picked_text if picked_text else _obj_to_text(obj)
                title = _pick_title(obj) or _default_title(raw, f"JSONL item (line {line_num})")
            else:
                text = _obj_to_text(obj)
                title = _default_title(raw, f"JSONL item (line {line_num})")

            docs.append(_build_document(raw, text, title, extra, suffix))

        if not docs:
            raise ValueError(
                f"JSONL document contains no usable lines: source_path={raw.source_path!r}"
            )
        return docs
