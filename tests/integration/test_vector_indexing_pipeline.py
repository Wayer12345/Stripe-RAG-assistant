"""Integration tests for offline vector indexing stage with injected fake store."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from app.application_layers.offline.build_vector_index import BuildVectorIndexLayer
from app.domain.models.chunk import Chunk
from app.domain.models.embedded_chunk import EmbeddedChunk
from app.infrastructure.storage.jsonl_store import write_jsonl
from app.infrastructure.vector_stores.qdrant_store import QdrantStore
from app.utils.hashing import sha256_text


class FakeVectorStore:
    def __init__(
        self,
        *,
        exists: bool = False,
        health_ok: bool = True,
        expected_dim: int | None = None,
    ) -> None:
        self.exists = exists
        self.health_ok = health_ok
        self.expected_dim = expected_dim
        self.create_calls = 0
        self.recreate_calls = 0
        self.validate_calls = 0
        self.payload_index_calls = 0
        self.upsert_calls = 0
        self.upserted_chunks: list[EmbeddedChunk] = []
        self.indexed_count = 0

    def healthcheck(self) -> bool:
        return self.health_ok

    def collection_exists(self) -> bool:
        return self.exists

    def create_collection(self, *, recreate: bool = False, vector_dim: int | None = None) -> None:
        if recreate:
            self.recreate_calls += 1
            self.indexed_count = 0
        else:
            self.create_calls += 1
        if vector_dim is not None:
            self.expected_dim = vector_dim
        self.exists = True

    def validate_collection(self, embedding_dim: int) -> None:
        self.validate_calls += 1
        if self.expected_dim is not None and embedding_dim != self.expected_dim:
            raise ValueError(
                f"Collection vector dimension mismatch: expected {self.expected_dim}, "
                f"got {embedding_dim}."
            )

    def create_payload_indexes(self) -> None:
        self.payload_index_calls += 1

    def upsert(self, chunks: list[EmbeddedChunk]) -> int:
        self.upsert_calls += 1
        self.upserted_chunks.extend(chunks)
        self.indexed_count += len(chunks)
        return len(chunks)

    def count(self) -> int:
        return self.indexed_count


def _make_embedded_chunk(
    *,
    chunk_id: str,
    document_id: str,
    text: str,
    vector: list[float],
    embedding_model: str = "fake-embedder",
    normalized: bool = True,
) -> EmbeddedChunk:
    chunk = Chunk(
        id=chunk_id,
        document_id=document_id,
        text=text,
        chunk_index=0,
        token_count=max(1, len(text.split())),
        content_hash=sha256_text(text),
        chunking_strategy="semantic",
        section="Setup",
        char_start=0,
        char_end=len(text),
        heading_path=["Docs", "Setup"],
        metadata={
            "title": f"Title {document_id}",
            "url": f"https://example.com/{document_id}",
            "source_path": f"docs/{document_id}.md",
            "source_name": f"{document_id}.md",
            "source_type": "markdown",
            "category": "guides",
            "document_content_hash": f"doc-hash-{document_id}",
            "chunker_name": "semantic_chunker",
        },
    )
    return EmbeddedChunk(
        chunk=chunk,
        vector=vector,
        embedding_model=embedding_model,
        embedding_dim=len(vector),
        normalized=normalized,
    )


def _manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.integration
class TestBuildVectorIndexLayer:
    def test_run_reads_jsonl_writes_manifest_and_upserts_single_chunk(
        self, tmp_path: Path
    ) -> None:
        input_path = tmp_path / "embedded_chunks.jsonl"
        manifest_path = tmp_path / "index_manifest.json"
        chunk = _make_embedded_chunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="alpha text",
            vector=[0.1, 0.2, 0.3],
        )
        write_jsonl(input_path, [chunk])
        fake_store = FakeVectorStore(exists=False)

        result = BuildVectorIndexLayer(
            input_path=input_path,
            manifest_path=manifest_path,
            vector_store=fake_store,
        ).run()

        manifest = _manifest(manifest_path)
        assert result.embedded_chunks_total == 1
        assert result.upserted_points_total == 1
        assert fake_store.create_calls == 1
        assert fake_store.upsert_calls == 1
        assert manifest["stage"] == "vector_indexing"
        assert manifest["embedded_chunks_total"] == 1
        assert manifest["upserted_points_total"] == 1

    def test_upserts_multiple_embedded_chunks(self, tmp_path: Path) -> None:
        input_path = tmp_path / "embedded_chunks.jsonl"
        manifest_path = tmp_path / "index_manifest.json"
        chunks = [
            _make_embedded_chunk(
                chunk_id="chunk-1", document_id="doc-1", text="alpha", vector=[0.1, 0.2]
            ),
            _make_embedded_chunk(
                chunk_id="chunk-2", document_id="doc-2", text="beta", vector=[0.3, 0.4]
            ),
        ]
        write_jsonl(input_path, chunks)
        fake_store = FakeVectorStore(exists=False)

        result = BuildVectorIndexLayer(
            input_path=input_path,
            manifest_path=manifest_path,
            vector_store=fake_store,
        ).run()

        assert result.upserted_points_total == 2
        assert len(fake_store.upserted_chunks) == 2

    def test_upsert_preserves_payload_metadata(self, tmp_path: Path) -> None:
        input_path = tmp_path / "embedded_chunks.jsonl"
        manifest_path = tmp_path / "index_manifest.json"
        chunk = _make_embedded_chunk(
            chunk_id="chunk-abc",
            document_id="doc-xyz",
            text="metadata rich text",
            vector=[0.9, 0.8, 0.7],
        )
        write_jsonl(input_path, [chunk])
        fake_store = FakeVectorStore(exists=False)

        BuildVectorIndexLayer(
            input_path=input_path,
            manifest_path=manifest_path,
            vector_store=fake_store,
        ).run()

        payload = QdrantStore.point_payload_from_embedded_chunk(fake_store.upserted_chunks[0])
        assert payload["chunk_id"] == "chunk-abc"
        assert payload["document_id"] == "doc-xyz"
        assert payload["source_path"] == "docs/doc-xyz.md"
        assert payload["source_type"] == "markdown"
        assert payload["embedding_model"] == "fake-embedder"

    def test_recreate_collection_when_flag_enabled(self, tmp_path: Path) -> None:
        input_path = tmp_path / "embedded_chunks.jsonl"
        manifest_path = tmp_path / "index_manifest.json"
        write_jsonl(
            input_path,
            [_make_embedded_chunk(chunk_id="chunk-1", document_id="doc-1", text="x", vector=[0.1])],
        )
        fake_store = FakeVectorStore(exists=True)

        result = BuildVectorIndexLayer(
            input_path=input_path,
            manifest_path=manifest_path,
            vector_store=fake_store,
            recreate_collection=True,
        ).run()

        assert fake_store.recreate_calls == 1
        assert result.recreate_collection is True

    def test_existing_compatible_collection_is_validated_and_reused(
        self, tmp_path: Path
    ) -> None:
        input_path = tmp_path / "embedded_chunks.jsonl"
        manifest_path = tmp_path / "index_manifest.json"
        write_jsonl(
            input_path,
            [
                _make_embedded_chunk(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    text="reused",
                    vector=[0.1, 0.2, 0.3],
                )
            ],
        )
        fake_store = FakeVectorStore(exists=True, expected_dim=3)

        BuildVectorIndexLayer(
            input_path=input_path,
            manifest_path=manifest_path,
            vector_store=fake_store,
        ).run()

        assert fake_store.create_calls == 0
        assert fake_store.validate_calls >= 1

    def test_dimension_mismatch_fails_clearly(self, tmp_path: Path) -> None:
        input_path = tmp_path / "embedded_chunks.jsonl"
        manifest_path = tmp_path / "index_manifest.json"
        write_jsonl(
            input_path,
            [_make_embedded_chunk(chunk_id="chunk-1", document_id="doc-1", text="x", vector=[0.1, 0.2])],
        )
        fake_store = FakeVectorStore(exists=True, expected_dim=3)

        with pytest.raises(ValueError, match="vector dimension mismatch"):
            BuildVectorIndexLayer(
                input_path=input_path,
                manifest_path=manifest_path,
                vector_store=fake_store,
            ).run()

    def test_inconsistent_embedding_models_fail(self, tmp_path: Path) -> None:
        input_path = tmp_path / "embedded_chunks.jsonl"
        manifest_path = tmp_path / "index_manifest.json"
        chunks = [
            _make_embedded_chunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                text="a",
                vector=[0.1, 0.2],
                embedding_model="model-a",
            ),
            _make_embedded_chunk(
                chunk_id="chunk-2",
                document_id="doc-2",
                text="b",
                vector=[0.3, 0.4],
                embedding_model="model-b",
            ),
        ]
        write_jsonl(input_path, chunks)

        with pytest.raises(ValueError, match="inconsistent embedding metadata"):
            BuildVectorIndexLayer(
                input_path=input_path,
                manifest_path=manifest_path,
                vector_store=FakeVectorStore(),
            ).run()

    def test_inconsistent_vector_dims_across_chunks_fail(self, tmp_path: Path) -> None:
        input_path = tmp_path / "embedded_chunks.jsonl"
        manifest_path = tmp_path / "index_manifest.json"
        chunks = [
            _make_embedded_chunk(chunk_id="chunk-1", document_id="doc-1", text="a", vector=[0.1, 0.2]),
            _make_embedded_chunk(
                chunk_id="chunk-2",
                document_id="doc-2",
                text="b",
                vector=[0.1, 0.2, 0.3],
            ),
        ]
        write_jsonl(input_path, chunks)

        with pytest.raises(ValueError, match="inconsistent embedding metadata"):
            BuildVectorIndexLayer(
                input_path=input_path,
                manifest_path=manifest_path,
                vector_store=FakeVectorStore(),
            ).run()

    def test_limit_skips_chunks_and_updates_skipped_total(self, tmp_path: Path) -> None:
        input_path = tmp_path / "embedded_chunks.jsonl"
        manifest_path = tmp_path / "index_manifest.json"
        chunks = [
            _make_embedded_chunk(chunk_id="chunk-1", document_id="doc-1", text="a", vector=[0.1]),
            _make_embedded_chunk(chunk_id="chunk-2", document_id="doc-2", text="b", vector=[0.2]),
            _make_embedded_chunk(chunk_id="chunk-3", document_id="doc-3", text="c", vector=[0.3]),
        ]
        write_jsonl(input_path, chunks)

        result = BuildVectorIndexLayer(
            input_path=input_path,
            manifest_path=manifest_path,
            vector_store=FakeVectorStore(),
            limit=2,
        ).run()

        assert result.embedded_chunks_total == 2
        assert result.skipped_chunks_total == 1
        manifest = _manifest(manifest_path)
        assert manifest["skipped_chunks_total"] == 1

    def test_manifest_counts_are_internally_consistent(self, tmp_path: Path) -> None:
        input_path = tmp_path / "embedded_chunks.jsonl"
        manifest_path = tmp_path / "index_manifest.json"
        chunks = [
            _make_embedded_chunk(chunk_id="chunk-1", document_id="doc-1", text="a", vector=[0.1]),
            _make_embedded_chunk(chunk_id="chunk-2", document_id="doc-2", text="b", vector=[0.2]),
        ]
        write_jsonl(input_path, chunks)

        BuildVectorIndexLayer(
            input_path=input_path,
            manifest_path=manifest_path,
            vector_store=FakeVectorStore(),
        ).run()

        manifest = _manifest(manifest_path)
        assert manifest["embedded_chunks_total"] == 2
        assert manifest["upserted_points_total"] == 2
        assert manifest["failed_chunks_total"] == 0

    def test_payload_indexes_created_when_enabled(self, tmp_path: Path) -> None:
        input_path = tmp_path / "embedded_chunks.jsonl"
        manifest_path = tmp_path / "index_manifest.json"
        write_jsonl(
            input_path,
            [_make_embedded_chunk(chunk_id="chunk-1", document_id="doc-1", text="x", vector=[0.1])],
        )
        fake_store = FakeVectorStore()

        BuildVectorIndexLayer(
            input_path=input_path,
            manifest_path=manifest_path,
            vector_store=fake_store,
            create_payload_indexes=True,
        ).run()

        assert fake_store.payload_index_calls == 1

    def test_payload_indexes_not_created_when_disabled(self, tmp_path: Path) -> None:
        input_path = tmp_path / "embedded_chunks.jsonl"
        manifest_path = tmp_path / "index_manifest.json"
        write_jsonl(
            input_path,
            [_make_embedded_chunk(chunk_id="chunk-1", document_id="doc-1", text="x", vector=[0.1])],
        )
        fake_store = FakeVectorStore()

        BuildVectorIndexLayer(
            input_path=input_path,
            manifest_path=manifest_path,
            vector_store=fake_store,
            create_payload_indexes=False,
        ).run()

        assert fake_store.payload_index_calls == 0

    def test_healthcheck_failure_fails_clearly(self, tmp_path: Path) -> None:
        input_path = tmp_path / "embedded_chunks.jsonl"
        manifest_path = tmp_path / "index_manifest.json"
        write_jsonl(
            input_path,
            [_make_embedded_chunk(chunk_id="chunk-1", document_id="doc-1", text="x", vector=[0.1])],
        )

        with pytest.raises(RuntimeError, match="healthcheck failed"):
            BuildVectorIndexLayer(
                input_path=input_path,
                manifest_path=manifest_path,
                vector_store=FakeVectorStore(health_ok=False),
            ).run()

    @pytest.mark.qdrant
    @pytest.mark.skipif(
        condition=("RUN_QDRANT_TESTS" not in os.environ),
        reason="Requires RUN_QDRANT_TESTS=1 and local Qdrant",
    )
    def test_optional_real_qdrant_marker_placeholder(self) -> None:
        assert True

