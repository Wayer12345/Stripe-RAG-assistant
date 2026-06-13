"""Local filesystem embedding cache keyed by deterministic hashes."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path
from typing import Any


def _sha256_text_allow_empty(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def build_embedding_cache_key(
    *,
    text: str,
    model_name: str,
    normalize_embeddings: bool,
    prefix_mode: str,
    input_type: str,
    prefix: str,
) -> str:
    """Build a deterministic cache key from embedding-affecting inputs."""
    payload = {
        "text_hash": _sha256_text_allow_empty(text),
        "prefixed_text_hash": _sha256_text_allow_empty(f"{prefix}{text}"),
        "model_name": model_name,
        "normalize_embeddings": normalize_embeddings,
        "prefix_mode": prefix_mode,
        "input_type": input_type,
        "prefix": prefix,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Simple local cache using one JSON file per embedding key."""

    def __init__(self, cache_path: Path | str) -> None:
        self._cache_path = Path(cache_path)
        self._cache_path.mkdir(parents=True, exist_ok=True)

    @property
    def cache_path(self) -> Path:
        return self._cache_path

    def _entry_path(self, key: str) -> Path:
        return self._cache_path / f"{key}.json"

    @staticmethod
    def _validate_vector(key: str, vector: Any) -> list[float]:
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"Corrupt cache entry for key={key!r}: vector must be a non-empty list.")
        typed_vector: list[float] = []
        for value in vector:
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(
                    f"Corrupt cache entry for key={key!r}: vector contains non-finite numeric value."
                )
            typed_vector.append(float(value))
        return typed_vector

    def get(self, key: str) -> list[float] | None:
        """Get one vector from cache by key."""
        entry_path = self._entry_path(key)
        if not entry_path.exists():
            return None

        try:
            payload = json.loads(entry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt cache entry for key={key!r}: invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"Corrupt cache entry for key={key!r}: payload must be an object.")
        if payload.get("key") != key:
            raise ValueError(f"Corrupt cache entry for key={key!r}: key mismatch.")
        return self._validate_vector(key, payload.get("vector"))

    def set(
        self,
        key: str,
        vector: list[float],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Set one vector in cache by key."""
        normalized_vector = self._validate_vector(key, vector)
        payload: dict[str, Any] = {"key": key, "vector": normalized_vector}
        if metadata:
            payload.update(metadata)

        entry_path = self._entry_path(key)
        entry_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        """Get all present cache vectors for the provided keys."""
        hits: dict[str, list[float]] = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                hits[key] = value
        return hits

    def set_many(
        self,
        items: dict[str, list[float]],
        *,
        metadata_by_key: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Set multiple cache entries in one call."""
        for key, vector in items.items():
            metadata = metadata_by_key.get(key) if metadata_by_key is not None else None
            self.set(key, vector, metadata=metadata)
