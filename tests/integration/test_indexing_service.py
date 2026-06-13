"""Integration tests for the current indexing_service CLI orchestration."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import app.application.indexing_service as indexing_service
import pytest


class _FakeParser:
    def __init__(self, config: Path) -> None:
        self._config = config

    def parse_args(self) -> Namespace:
        return Namespace(config=self._config)


def _make_layer(
    stage_name: str,
    calls: list[str],
    init_configs: dict[str, Path],
    *,
    fail: bool = False,
) -> type:
    class _Layer:
        def __init__(self, *, config_path: Path) -> None:
            init_configs[stage_name] = config_path

        def run(self) -> Any:
            calls.append(stage_name)
            if fail:
                raise RuntimeError(f"{stage_name} failed")
            return object()

    return _Layer


@pytest.mark.integration
def test_arg_parser_default_config_path() -> None:
    parser = indexing_service._build_arg_parser()
    parsed = parser.parse_args([])
    assert parsed.config == Path("configs/config.yaml")


@pytest.mark.integration
def test_main_runs_all_layers_in_order_with_same_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    init_configs: dict[str, Path] = {}
    config_path = Path("/tmp/custom-config.yaml")

    monkeypatch.setattr(indexing_service, "_build_arg_parser", lambda: _FakeParser(config_path))
    monkeypatch.setattr(
        indexing_service,
        "BuildParsedDocsLayer",
        _make_layer("parsed", calls, init_configs),
    )
    monkeypatch.setattr(
        indexing_service,
        "BuildCleanedDocsLayer",
        _make_layer("cleaned", calls, init_configs),
    )
    monkeypatch.setattr(
        indexing_service,
        "BuildChunksLayer",
        _make_layer("chunks", calls, init_configs),
    )
    monkeypatch.setattr(
        indexing_service,
        "BuildEmbeddingsLayer",
        _make_layer("embeddings", calls, init_configs),
    )
    monkeypatch.setattr(
        indexing_service,
        "BuildVectorIndexLayer",
        _make_layer("vector_index", calls, init_configs),
    )

    indexing_service.main()

    assert calls == ["parsed", "cleaned", "chunks", "embeddings", "vector_index"]
    assert init_configs == {
        "parsed": config_path,
        "cleaned": config_path,
        "chunks": config_path,
        "embeddings": config_path,
        "vector_index": config_path,
    }


@pytest.mark.integration
def test_main_stops_when_a_layer_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    init_configs: dict[str, Path] = {}
    config_path = Path("configs/config.yaml")

    monkeypatch.setattr(indexing_service, "_build_arg_parser", lambda: _FakeParser(config_path))
    monkeypatch.setattr(
        indexing_service,
        "BuildParsedDocsLayer",
        _make_layer("parsed", calls, init_configs),
    )
    monkeypatch.setattr(
        indexing_service,
        "BuildCleanedDocsLayer",
        _make_layer("cleaned", calls, init_configs, fail=True),
    )
    monkeypatch.setattr(
        indexing_service,
        "BuildChunksLayer",
        _make_layer("chunks", calls, init_configs),
    )
    monkeypatch.setattr(
        indexing_service,
        "BuildEmbeddingsLayer",
        _make_layer("embeddings", calls, init_configs),
    )
    monkeypatch.setattr(
        indexing_service,
        "BuildVectorIndexLayer",
        _make_layer("vector_index", calls, init_configs),
    )

    # main() catches the RuntimeError and calls sys.exit(1) — test for that.
    with pytest.raises(SystemExit) as exc_info:
        indexing_service.main()
    assert exc_info.value.code == 1
    assert calls == ["parsed", "cleaned"]
