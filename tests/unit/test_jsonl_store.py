"""Unit tests for app/infrastructure/storage/jsonl_store.py."""

import json
from pathlib import Path

import pytest
from app.infrastructure.storage.jsonl_store import JsonlStore
from pydantic import BaseModel


class _SampleModel(BaseModel):
    name: str
    value: int


@pytest.mark.unit
class TestJsonlStoreWrite:
    def test_writes_pydantic_models(self, tmp_path: Path) -> None:
        store = JsonlStore()
        path = tmp_path / "out.jsonl"
        models = [_SampleModel(name="alpha", value=1), _SampleModel(name="beta", value=2)]

        store.write(path, models)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"name": "alpha", "value": 1}
        assert json.loads(lines[1]) == {"name": "beta", "value": 2}

    def test_writes_dicts(self, tmp_path: Path) -> None:
        store = JsonlStore()
        path = tmp_path / "out.jsonl"
        items = [{"key": "a", "num": 10}, {"key": "b", "num": 20}]

        store.write(path, items)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"key": "a", "num": 10}
        assert json.loads(lines[1]) == {"key": "b", "num": 20}

    def test_writes_mixed_pydantic_and_dicts(self, tmp_path: Path) -> None:
        store = JsonlStore()
        path = tmp_path / "mixed.jsonl"
        items: list = [_SampleModel(name="x", value=99), {"raw": True}]

        store.write(path, items)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"name": "x", "value": 99}
        assert json.loads(lines[1]) == {"raw": True}

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        store = JsonlStore()
        path = tmp_path / "nested" / "deep" / "out.jsonl"

        store.write(path, [{"x": 1}])

        assert path.exists()

    def test_empty_list_creates_empty_file(self, tmp_path: Path) -> None:
        store = JsonlStore()
        path = tmp_path / "empty.jsonl"

        store.write(path, [])

        assert path.exists()
        assert path.read_text(encoding="utf-8") == ""

    def test_invalid_type_raises_type_error(self, tmp_path: Path) -> None:
        store = JsonlStore()
        path = tmp_path / "bad.jsonl"

        with pytest.raises(TypeError, match="Expected BaseModel or dict"):
            store.write(path, ["not a model or dict"])  # type: ignore[list-item]

    def test_dict_with_unserializable_value_raises(self, tmp_path: Path) -> None:
        store = JsonlStore()
        path = tmp_path / "bad.jsonl"

        with pytest.raises((TypeError, ValueError)):
            store.write(path, [{"bad": {1, 2, 3}}])  # sets are not JSON-serializable

    def test_output_is_utf8(self, tmp_path: Path) -> None:
        store = JsonlStore()
        path = tmp_path / "unicode.jsonl"

        store.write(path, [{"text": "café ☕ 支付"}])

        raw = path.read_bytes()
        decoded = raw.decode("utf-8")
        assert "café" in decoded
        assert "支付" in decoded


@pytest.mark.unit
class TestJsonlStoreRead:
    def test_reads_back_written_dicts(self, tmp_path: Path) -> None:
        store = JsonlStore()
        path = tmp_path / "data.jsonl"
        original = [{"a": 1}, {"b": 2}]

        store.write(path, original)
        result = store.read(path)

        assert result == original

    def test_reads_back_pydantic_model_data(self, tmp_path: Path) -> None:
        store = JsonlStore()
        path = tmp_path / "models.jsonl"
        store.write(path, [_SampleModel(name="z", value=7)])

        result = store.read(path)

        assert result == [{"name": "z", "value": 7}]

    def test_reads_empty_file_as_empty_list(self, tmp_path: Path) -> None:
        store = JsonlStore()
        path = tmp_path / "empty.jsonl"
        store.write(path, [])

        result = store.read(path)

        assert result == []

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        store = JsonlStore()
        path = tmp_path / "blanks.jsonl"
        path.write_text('{"x": 1}\n\n{"x": 2}\n', encoding="utf-8")

        result = store.read(path)

        assert result == [{"x": 1}, {"x": 2}]

    def test_read_returns_list_of_dicts(self, tmp_path: Path) -> None:
        store = JsonlStore()
        path = tmp_path / "typed.jsonl"
        store.write(path, [{"key": "val"}])

        result = store.read(path)

        assert isinstance(result, list)
        assert isinstance(result[0], dict)
