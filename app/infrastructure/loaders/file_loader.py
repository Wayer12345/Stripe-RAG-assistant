"""File-system loader that reads raw files into RawDocument byte payloads."""

import mimetypes
from pathlib import Path
from typing import ClassVar

from app.domain.interfaces.document_loader import RawDocument


class FileLoader:
    """Loads files from a local directory as RawDocument byte payloads.

    Implements the :class:`~app.domain.interfaces.document_loader.DocumentLoader`
    Protocol.  Files are read as raw bytes; no parsing or cleaning is performed.

    Args:
        input_dir: Directory to scan for source files.
        supported_extensions: Set of file extensions to accept, including the
            leading dot (e.g. ``{".txt"}``).  Defaults to ``{".txt"}``.
        recursive: Whether to descend into sub-directories.  Defaults to
            ``True``.

    Raises:
        ValueError: If ``input_dir`` does not exist or is not a directory.
    """

    DEFAULT_EXTENSIONS: ClassVar[set[str]] = {".txt"}

    def __init__(
        self,
        input_dir: Path,
        supported_extensions: set[str] | None = None,
        recursive: bool = True,
    ) -> None:
        if not input_dir.exists():
            raise ValueError(f"input_dir does not exist: {input_dir}")
        if not input_dir.is_dir():
            raise ValueError(f"input_dir is not a directory: {input_dir}")

        self._input_dir = input_dir
        self._supported_extensions = (
            supported_extensions if supported_extensions is not None else self.DEFAULT_EXTENSIONS
        )
        self._recursive = recursive

    def load(self) -> list[RawDocument]:
        """Scan ``input_dir`` and return one RawDocument per supported file.

        Files are sorted deterministically by their resolved path.  Directories
        and files with unsupported extensions are silently skipped.

        Returns:
            List of :class:`~app.domain.interfaces.document_loader.RawDocument`
            objects, one per accepted file, in stable path order.
        """
        pattern = "**/*" if self._recursive else "*"
        all_paths = sorted(self._input_dir.glob(pattern))

        raw_docs: list[RawDocument] = []
        for path in all_paths:
            if not path.is_file():
                continue

            ext = path.suffix.lower()
            if ext not in self._supported_extensions:
                continue

            source_type = ext.lstrip(".").lower()
            content = path.read_bytes()
            mime_type, _ = mimetypes.guess_type(str(path))

            raw_docs.append(
                RawDocument(
                    source_type=source_type,
                    content=content,
                    source_path=str(path),
                    source_name=path.name,
                    mime_type=mime_type,
                    metadata={
                        "file_size": path.stat().st_size,
                        "extension": ext,
                    },
                )
            )

        return raw_docs
