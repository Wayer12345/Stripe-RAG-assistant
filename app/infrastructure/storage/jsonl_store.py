"""JSONL artifact storage for document pipeline outputs."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _normalize_item(item: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(item, BaseModel):
        return item.model_dump(mode="json")
    if isinstance(item, dict):
        return item
    raise TypeError(f"Expected BaseModel or dict, got {type(item).__name__!r}.")


def write_jsonl(path: Path, items: Iterable[BaseModel | dict[str, Any]]) -> None:
    """Write one JSON object per line into a UTF-8 JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(_normalize_item(item), ensure_ascii=False) + "\n")


def read_jsonl[TModel: BaseModel](
    path: Path,
    model_type: type[TModel],
    *,
    limit: int | None = None,
) -> list[TModel]:
    """Read a JSONL file and validate each non-empty row into ``model_type``.

    Args:
        path: Path to the JSONL file.
        model_type: Pydantic model class used to validate each row.
        limit: When provided, stop reading after this many valid records.
            Avoids loading the full file into memory when only a subset is needed.

    Returns:
        List of validated model instances, at most ``limit`` items when set.
    """
    records: list[TModel] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            records.append(model_type.model_validate(payload))
            if limit is not None and len(records) >= limit:
                break
    return records


def count_jsonl(path: Path) -> int:
    """Count non-empty records in a JSONL file without parsing JSON payloads.

    Reads only the raw bytes and counts non-blank lines, which is O(n) I/O
    but O(1) memory — useful to get the total size of a file before applying
    a read limit so that ``skipped`` counts can still be reported accurately.

    Args:
        path: Path to the JSONL file.

    Returns:
        Number of non-blank lines in the file.
    """
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


class JsonlStore:
    """Reads and writes JSONL (newline-delimited JSON) artifact files.

    Each record is serialized as one JSON object per line using UTF-8 encoding.
    Pydantic models are serialized with ``model_dump(mode="json")`` to ensure
    all field types are JSON-compatible. Plain dicts are passed directly to
    ``json.dumps``.
    """

    def write(
        self,
        path: Path,
        items: list[BaseModel | dict[str, Any]],
    ) -> None:
        """Write *items* to a JSONL file, creating parent directories as needed.

        Args:
            path: Output file path.
            items: Ordered list of Pydantic models or plain dicts to serialize.
                An empty list produces an empty file.

        Raises:
            TypeError: If an item is neither a ``BaseModel`` nor a ``dict``.
            ValueError: If a dict item cannot be serialized to JSON.
        """
        write_jsonl(path, items)

    def read(self, path: Path) -> list[dict[str, Any]]:
        """Read a JSONL file and return a list of parsed dicts.

        Blank lines are skipped. Each non-empty line must be a valid JSON object.

        Args:
            path: Path to the JSONL file.

        Returns:
            List of parsed dicts, one per non-empty line, in file order.

        Raises:
            json.JSONDecodeError: If any non-empty line is not valid JSON.
        """
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records
