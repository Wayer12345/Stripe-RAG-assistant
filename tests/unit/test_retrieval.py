"""Unit tests for dense retrieval infrastructure."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import app.infrastructure.retrieval.retriever_factory as retriever_factory
import pytest
from app.infrastructure.retrieval.dense_retriever import DenseRetriever
from app.infrastructure.retrieval.retriever_factory import create_retriever
from app.utils.config import load_settings


class _FakeEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return self._vector


class _FakeVectorStore:
    def __init__(self, hits: list[Any]) -> None:
        self._hits = hits
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]:
        self.calls.append({"query_vector": query_vector, "top_k": top_k, "filters": filters})
        return self._hits


class _FakeQdrantClient:
    def __init__(self, points: list[Any]) -> None:
        self._points = points

    def query_points(self, **_: Any) -> Any:
        return SimpleNamespace(points=self._points)


@pytest.mark.unit
def test_dense_retriever_embeds_query_and_passes_vector_and_filters() -> None:
    fake_embedder = _FakeEmbedder([0.1, 0.2, 0.3])
    fake_store = _FakeVectorStore([])
    retriever = DenseRetriever(embedder=fake_embedder, vector_store=fake_store, default_top_k=10)

    results = retriever.retrieve(
        "What is 3D Secure 2?",
        top_k=4,
        filters={"document_id": "doc-1"},
    )

    assert results == []
    assert fake_embedder.queries == ["What is 3D Secure 2?"]
    assert fake_store.calls[0]["query_vector"] == [0.1, 0.2, 0.3]
    assert fake_store.calls[0]["top_k"] == 4
    assert fake_store.calls[0]["filters"] == {"document_id": "doc-1"}


@pytest.mark.unit
def test_dense_retriever_respects_top_k() -> None:
    fake_embedder = _FakeEmbedder([0.5, 0.6])
    fake_store = _FakeVectorStore([])
    retriever = DenseRetriever(embedder=fake_embedder, vector_store=fake_store, default_top_k=7)

    retriever.retrieve("q", top_k=2)

    assert fake_store.calls[0]["top_k"] == 2


@pytest.mark.unit
def test_qdrant_hits_map_to_retrieval_result_with_source_fields(tmp_path: Path) -> None:
    pytest.importorskip("qdrant_client")
    from app.infrastructure.vector_stores.qdrant_store import QdrantStore

    point = SimpleNamespace(
        id="pt-1",
        score=0.91,
        payload={
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "text": "Stripe guide chunk",
            "title": "3D Secure docs",
            "url": "https://docs.example/3ds",
            "source_type": "markdown",
            "source_path": "docs/3ds.md",
            "source_name": "3ds.md",
            "section": "Authentication",
            "token_count": 42,
        },
    )
    store = QdrantStore(
        collection_name="test",
        local_path=tmp_path / "qdrant",
        client=_FakeQdrantClient([point]),
    )

    results = store.search([0.1, 0.2], top_k=3, filters={"document_id": "doc-1"})

    assert len(results) == 1
    result = results[0]
    assert result.chunk_id == "chunk-1"
    assert result.document_id == "doc-1"
    assert result.source.title == "3D Secure docs"
    assert result.source.url == "https://docs.example/3ds"
    assert result.source.source_type == "markdown"
    assert result.source.support_score == pytest.approx(0.91)
    assert result.dense_score == pytest.approx(0.91)
    assert result.retrieval_score == pytest.approx(0.91)
    assert result.final_score == pytest.approx(0.91)
    assert result.lexical_score is None
    assert result.reranker_score is None


@pytest.mark.unit
def test_qdrant_mapping_with_missing_optional_fields_does_not_crash(tmp_path: Path) -> None:
    pytest.importorskip("qdrant_client")
    from app.infrastructure.vector_stores.qdrant_store import QdrantStore

    point = SimpleNamespace(
        id="pt-2",
        score=0.5,
        payload={
            "chunk_id": "chunk-2",
            "document_id": "doc-2",
            "text": "Minimal payload chunk",
        },
    )
    store = QdrantStore(
        collection_name="test",
        local_path=tmp_path / "qdrant",
        client=_FakeQdrantClient([point]),
    )

    results = store.search([0.2, 0.3], top_k=1)

    assert len(results) == 1
    assert results[0].source.url is None
    assert results[0].source.section is None
    assert results[0].source.source_type is None


@pytest.mark.unit
def test_qdrant_search_empty_points_returns_empty_list(tmp_path: Path) -> None:
    pytest.importorskip("qdrant_client")
    from app.infrastructure.vector_stores.qdrant_store import QdrantStore

    store = QdrantStore(
        collection_name="test",
        local_path=tmp_path / "qdrant",
        client=_FakeQdrantClient([]),
    )

    assert store.search([0.2, 0.3], top_k=2) == []


@pytest.mark.unit
def test_factory_unsupported_retrieval_strategy_raises(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_dir.joinpath("config.yaml").write_text(
        """
app:
  name: "x"
  environment: "local"
  log_level: "INFO"
paths:
  raw_dir: "data/raw"
  interim_dir: "data/interim"
  processed_dir: "data/processed"
  indexes_dir: "data/indexes"
  manifests_dir: "data/manifests"
  eval_dir: "eval"
ingestion:
  input_dir: "data/raw"
  recursive: true
  supported_extensions: [".md"]
  outputs:
    parsed_documents_path: "data/interim/parsed_documents.jsonl"
    cleaned_documents_path: "data/interim/cleaned_documents.jsonl"
    manifest_path: "data/manifests/ingestion_manifest.json"
cleaning:
  mode: "conservative"
  boilerplate:
    phrases: ["Sign in"]
chunking: {}
embeddings: {}
vector_store: {}
indexing: {}
retrieval:
  strategy: "hybrid"
  dense_top_k: 10
  write_trace: true
  trace_dir: "data/traces/queries"
""",
        encoding="utf-8",
    )
    settings = load_settings(config_dir)

    with pytest.raises(ValueError, match="Unsupported retrieval strategy for this implementation"):
        create_retriever(settings)


@pytest.mark.unit
def test_cached_retriever_reuses_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeSettings:
        class retrieval:
            strategy = "dense"
            dense_top_k = 30

        class embeddings:
            provider = "sentence_transformers"
            model_name = "BAAI/bge-small-en-v1.5"
            batch_size = 32
            normalize_embeddings = True
            prefix_mode = "bge"

        class vector_store:
            provider = "qdrant"
            mode = "embedded"
            local_path = Path("data/indexes/qdrant")
            collection_name = "stripe_guides_v1"
            distance = "cosine"
            timeout = 30
            wait = True
            upsert_batch_size = 64

        class api:
            class warmup:
                qdrant_healthcheck_enabled = True
                retrieval_embed_query_enabled = True
                retrieval_embed_query_text = "warmup query"
                retrieval_tiny_search_enabled = False
                retrieval_tiny_search_top_k = 1

    retriever_factory._RETRIEVER_CACHE.clear()
    created_objects: list[object] = []

    def _fake_create_retriever(_: object) -> object:
        obj = object()
        created_objects.append(obj)
        return obj

    monkeypatch.setattr(retriever_factory, "create_retriever", _fake_create_retriever)
    settings = _FakeSettings()
    first = retriever_factory.create_cached_retriever(settings)  # type: ignore[arg-type]
    second = retriever_factory.create_cached_retriever(settings)  # type: ignore[arg-type]

    assert first is second
    assert len(created_objects) == 1


@pytest.mark.unit
def test_shutdown_retriever_cache_closes_cached_vector_store() -> None:
    class _FakeVectorStore:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class _FakeRetriever:
        def __init__(self, vector_store: _FakeVectorStore) -> None:
            self._vector_store = vector_store

    fake_store = _FakeVectorStore()
    fake_retriever = _FakeRetriever(fake_store)
    retriever_factory._RETRIEVER_CACHE.clear()
    retriever_factory._RETRIEVER_CACHE[("k",)] = fake_retriever  # type: ignore[assignment]

    retriever_factory.shutdown_retriever_cache()

    assert fake_store.closed is True
    assert retriever_factory._RETRIEVER_CACHE == {}
