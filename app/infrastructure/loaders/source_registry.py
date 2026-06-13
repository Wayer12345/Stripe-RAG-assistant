"""Parser registry that resolves the right Parser for a given RawDocument."""

from app.domain.interfaces.document_loader import RawDocument
from app.domain.interfaces.parser import Parser


class SourceRegistry:
    """Resolves the appropriate Parser for a given RawDocument.

    Parsers are tried in registration order; the first one whose
    :meth:`~app.domain.interfaces.parser.Parser.supports` method returns
    ``True`` is returned.

    Args:
        parsers: Ordered list of Parser implementations to query.
    """

    def __init__(self, parsers: list[Parser]) -> None:
        self._parsers = list(parsers)

    def resolve(self, raw_document: RawDocument) -> Parser:
        """Return the first parser that supports *raw_document*.

        Args:
            raw_document: The raw document needing a parser.

        Returns:
            A :class:`~app.domain.interfaces.parser.Parser` instance.

        Raises:
            ValueError: If no registered parser supports the document's
                ``source_type`` / ``mime_type`` combination.
        """
        for parser in self._parsers:
            if parser.supports(raw_document.source_type, raw_document.mime_type):
                return parser

        raise ValueError(
            f"No registered parser supports source_type={raw_document.source_type!r}, "
            f"mime_type={raw_document.mime_type!r}. "
            f"Registered parsers: {[type(p).__name__ for p in self._parsers]}."
        )

    def supported_source_types(self) -> set[str]:
        """Return the union of source types declared by all registered parsers.

        Parsers that expose a ``supported_source_types()`` method contribute
        their declared types.  Parsers without such a method are silently
        skipped; only explicit declarations are included.

        Returns:
            Set of supported source-type strings (e.g. ``{"txt", "html"}``).
        """
        types: set[str] = set()
        for parser in self._parsers:
            if callable(getattr(parser, "supported_source_types", None)):
                types.update(parser.supported_source_types())
        return types
