"""Unit tests for storage helper functions."""

import json
from pathlib import Path

import pytest
from app.infrastructure.storage.jsonl_store import read_jsonl, write_jsonl
from app.infrastructure.storage.manifest_store import read_manifest, write_manifest
from pydantic import BaseModel


class _Item(BaseModel):
    name: str
    value: int


@pytest.mark.unit
def test_write_and_read_jsonl_models(tmp_path: Path) -> None:
    path = tmp_path / "artifacts" / "parsed.jsonl"
    write_jsonl(path, [_Item(name="a", value=1), _Item(name="b", value=2)])

    items = read_jsonl(path, _Item)
    assert [item.model_dump() for item in items] == [
        {"name": "a", "value": 1},
        {"name": "b", "value": 2},
    ]


@pytest.mark.unit
def test_jsonl_skips_blank_lines_when_reading(tmp_path: Path) -> None:
    path = tmp_path / "parsed.jsonl"
    path.write_text('{"name":"x","value":1}\n\n{"name":"y","value":2}\n', encoding="utf-8")

    items = read_jsonl(path, _Item)
    assert [item.name for item in items] == ["x", "y"]


@pytest.mark.unit
def test_manifest_write_and_read_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "manifests" / "ingestion_manifest.json"
    manifest = {"run_id": "ingestion_1", "errors": []}

    write_manifest(path, manifest)
    loaded = read_manifest(path)

    assert path.exists()
    assert loaded == manifest


@pytest.mark.unit
def test_manifest_is_pretty_json_utf8(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_manifest(path, {"title": "café", "ok": True})

    content = path.read_text(encoding="utf-8")
    parsed = json.loads(content)
    assert "\n" in content
    assert "café" in content
    assert parsed["ok"] is True
