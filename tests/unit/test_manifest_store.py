"""Unit tests for app/infrastructure/storage/manifest_store.py."""

import json
from pathlib import Path
from typing import Any

import pytest
from app.infrastructure.storage.manifest_store import ManifestStore


def _sample_manifest() -> dict[str, Any]:
    return {
        "run_id": "ingestion_20260530T142300Z",
        "started_at": "2026-05-30T14:23:00+00:00",
        "finished_at": "2026-05-30T14:23:05+00:00",
        "input_count": 3,
        "parsed_documents_count": 3,
        "cleaned_documents_count": 3,
        "failed_items_count": 0,
        "parsed_output_path": "data/interim/parsed_documents.jsonl",
        "cleaned_output_path": "data/interim/cleaned_documents.jsonl",
        "manifest_output_path": "data/manifests/ingestion_manifest.json",
        "source_type_counts": {"txt": 2, "md": 1},
        "errors": [],
    }


@pytest.mark.unit
class TestManifestStoreWrite:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        store = ManifestStore()
        path = tmp_path / "manifest.json"

        store.write(path, _sample_manifest())

        content = path.read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        store = ManifestStore()
        path = tmp_path / "deep" / "nested" / "manifest.json"

        store.write(path, _sample_manifest())

        assert path.exists()

    def test_expected_fields_preserved(self, tmp_path: Path) -> None:
        store = ManifestStore()
        path = tmp_path / "manifest.json"
        manifest = _sample_manifest()

        store.write(path, manifest)

        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["run_id"] == manifest["run_id"]
        assert result["input_count"] == manifest["input_count"]
        assert result["parsed_documents_count"] == manifest["parsed_documents_count"]
        assert result["cleaned_documents_count"] == manifest["cleaned_documents_count"]
        assert result["failed_items_count"] == manifest["failed_items_count"]
        assert result["source_type_counts"] == manifest["source_type_counts"]
        assert result["errors"] == manifest["errors"]

    def test_output_is_pretty_printed(self, tmp_path: Path) -> None:
        store = ManifestStore()
        path = tmp_path / "manifest.json"

        store.write(path, _sample_manifest())

        content = path.read_text(encoding="utf-8")
        # Pretty-printed JSON has newlines and indentation.
        assert "\n" in content
        assert "  " in content

    def test_output_is_sorted(self, tmp_path: Path) -> None:
        store = ManifestStore()
        path = tmp_path / "manifest.json"
        store.write(path, {"z_key": 1, "a_key": 2})

        content = path.read_text(encoding="utf-8")
        pos_a = content.index("a_key")
        pos_z = content.index("z_key")
        assert pos_a < pos_z

    def test_deterministic_output(self, tmp_path: Path) -> None:
        store = ManifestStore()
        path1 = tmp_path / "m1.json"
        path2 = tmp_path / "m2.json"
        manifest = _sample_manifest()

        store.write(path1, manifest)
        store.write(path2, manifest)

        assert path1.read_text(encoding="utf-8") == path2.read_text(encoding="utf-8")

    def test_file_ends_with_newline(self, tmp_path: Path) -> None:
        store = ManifestStore()
        path = tmp_path / "manifest.json"

        store.write(path, _sample_manifest())

        content = path.read_text(encoding="utf-8")
        assert content.endswith("\n")

    def test_utf8_values_preserved(self, tmp_path: Path) -> None:
        store = ManifestStore()
        path = tmp_path / "manifest.json"

        store.write(path, {"title": "café ☕"})

        content = path.read_bytes().decode("utf-8")
        assert "café" in content
