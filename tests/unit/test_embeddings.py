"""Unit tests for embedding infrastructure components."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.infrastructure.embeddings.embedder_factory import create_embedder
from app.infrastructure.embeddings.embedding_cache import (
    EmbeddingCache,
    build_embedding_cache_key,
)
from app.infrastructure.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from app.utils.config import load_settings


def test_prefix_application_none_mode() -> None:
    embedder = SentenceTransformerEmbedder(prefix_mode="none")
    assert embedder.query_prefix_for_mode("none") == ""
    assert embedder.document_prefix_for_mode("none") == ""


def test_prefix_application_bge_mode() -> None:
    embedder = SentenceTransformerEmbedder(prefix_mode="bge")
    assert (
        embedder.query_prefix_for_mode("bge")
        == "Represent this sentence for searching relevant passages: "
    )
    assert embedder.document_prefix_for_mode("bge") == ""


def test_prefix_application_e5_mode() -> None:
    embedder = SentenceTransformerEmbedder(prefix_mode="e5")
    assert embedder.query_prefix_for_mode("e5") == "query: "
    assert embedder.document_prefix_for_mode("e5") == "passage: "


def test_cache_key_changes_when_model_changes() -> None:
    key_1 = build_embedding_cache_key(
        text="same",
        model_name="model-a",
        normalize_embeddings=True,
        prefix_mode="bge",
        input_type="document",
        prefix="",
    )
    key_2 = build_embedding_cache_key(
        text="same",
        model_name="model-b",
        normalize_embeddings=True,
        prefix_mode="bge",
        input_type="document",
        prefix="",
    )
    assert key_1 != key_2


def test_cache_key_changes_when_prefix_mode_changes() -> None:
    key_1 = build_embedding_cache_key(
        text="same",
        model_name="model-a",
        normalize_embeddings=True,
        prefix_mode="bge",
        input_type="document",
        prefix="",
    )
    key_2 = build_embedding_cache_key(
        text="same",
        model_name="model-a",
        normalize_embeddings=True,
        prefix_mode="e5",
        input_type="document",
        prefix="passage: ",
    )
    assert key_1 != key_2


def test_cache_key_changes_when_input_type_changes() -> None:
    key_1 = build_embedding_cache_key(
        text="same",
        model_name="model-a",
        normalize_embeddings=True,
        prefix_mode="e5",
        input_type="query",
        prefix="query: ",
    )
    key_2 = build_embedding_cache_key(
        text="same",
        model_name="model-a",
        normalize_embeddings=True,
        prefix_mode="e5",
        input_type="document",
        prefix="passage: ",
    )
    assert key_1 != key_2


def test_embedding_cache_write_and_read_vector(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "embedding_cache")
    key = "cache-key"
    vector = [0.1, 0.2, 0.3]
    cache.set(key, vector, metadata={"embedding_model": "fake"})
    assert cache.get(key) == vector


def test_embedding_cache_corrupt_entry_raises_clear_error(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "embedding_cache")
    path = cache.cache_path / "broken.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt cache entry"):
        cache.get("broken")


def test_factory_creates_sentence_transformers_embedder() -> None:
    settings = load_settings()
    embedder = create_embedder(settings)
    assert isinstance(embedder, SentenceTransformerEmbedder)


def test_factory_raises_for_unsupported_provider(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "app": {"name": "x", "environment": "local", "log_level": "INFO"},
        "paths": {
            "raw_dir": "data/raw",
            "interim_dir": "data/interim",
            "processed_dir": "data/processed",
            "indexes_dir": "data/indexes",
            "manifests_dir": "data/manifests",
            "eval_dir": "eval",
        },
        "ingestion": {
            "input_dir": "data/raw",
            "recursive": True,
            "supported_extensions": [".md"],
            "outputs": {
                "parsed_documents_path": "data/interim/parsed_documents.jsonl",
                "cleaned_documents_path": "data/interim/cleaned_documents.jsonl",
                "manifest_path": "data/manifests/ingestion_manifest.json",
            },
            "parser_options": {},
            "failure_policy": {},
        },
        "cleaning": {
            "mode": "conservative",
            "steps": {},
            "duplicate_lines": {"window_size": 5},
            "blank_lines": {"max_blank_lines": 1},
            "boilerplate": {"phrases": ["Support"], "max_line_length": 80},
            "quality": {},
            "preserve": {},
        },
        "chunking": {},
        "embeddings": {"provider": "unsupported-provider"},
    }
    (config_dir / "config.yaml").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    settings = load_settings(config_dir)
    with pytest.raises(ValueError, match=r"Unsupported embeddings\.provider"):
        create_embedder(settings)
