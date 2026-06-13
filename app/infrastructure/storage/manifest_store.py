"""Manifest storage for pipeline run metadata."""

import json
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from app.utils.constants import ARTIFACT_SCHEMA_VERSION


def _normalize_manifest(manifest: dict[str, Any] | BaseModel) -> dict[str, Any]:
    if isinstance(manifest, BaseModel):
        return manifest.model_dump(mode="json")
    if isinstance(manifest, dict):
        return manifest
    raise TypeError(f"Expected dict or BaseModel, got {type(manifest).__name__!r}.")


def write_manifest(path: Path, manifest: dict[str, Any] | BaseModel) -> None:
    """Write a manifest as pretty UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_manifest = _normalize_manifest(manifest)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(normalized_manifest, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")


def write_json_payload(path: Path, payload: dict[str, Any] | BaseModel) -> None:
    """Write any JSON payload using the shared manifest JSON formatting."""
    write_manifest(path, payload)


def read_manifest(path: Path) -> dict[str, Any]:
    """Read a manifest JSON file into a dictionary."""
    with path.open("r", encoding="utf-8") as fh:
        return cast(dict[str, Any], json.load(fh))


def build_base_manifest(
    *,
    run_id: str,
    stage: str,
    started_at: str,
    finished_at: str,
    duration_ms: int,
    config_path: Path | str,
    errors: list[dict[str, Any]] | None = None,
    schema_version: str = ARTIFACT_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Return shared base fields for all stage manifests."""
    return {
        "run_id": run_id,
        "stage": stage,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": duration_ms,
        "errors": errors or [],
        "config_path": str(config_path),
        "schema_version": schema_version,
    }


class ManifestStore:
    """Writes pipeline run manifests as deterministically formatted JSON files.

    Output uses ``indent=2`` and ``sort_keys=True`` for stable diffs and
    human readability.  Parent directories are created automatically.
    """

    def read(self, path: Path) -> dict[str, Any]:
        """Read a manifest JSON file and return it as a dict.

        Args:
            path: Path to the manifest JSON file.

        Returns:
            Parsed manifest dictionary.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        return read_manifest(path)

    def write(self, path: Path, manifest: dict[str, Any] | BaseModel) -> None:
        """Write *manifest* to a pretty-printed JSON file.

        Args:
            path: Output file path.
            manifest: Manifest dictionary to serialize.  Must contain only
                JSON-serializable values.

        Raises:
            ValueError: If the manifest contains non-JSON-serializable values.
        """
        write_manifest(path, manifest)
