"""Domain interface contract for lexical index operations."""

from pathlib import Path
from typing import Any, Protocol

from app.domain.models.chunk import Chunk
from app.domain.models.retrieval_result import RetrievalResult


class LexicalIndex(Protocol):
    """Keyword retrieval index contract independent of specific engines."""

    def build(self, chunks: list[Chunk]) -> None:
        """Build or rebuild a lexical index from chunks."""
        ...

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Return lexical retrieval candidates."""
        ...

    def save(self, path: Path) -> None:
        """Persist lexical index artifacts to the provided path."""
        ...

    def load(self, path: Path) -> None:
        """Load lexical index artifacts from the provided path."""
        ...

    def count(self) -> int:
        """Return indexed chunk count."""
        ...

