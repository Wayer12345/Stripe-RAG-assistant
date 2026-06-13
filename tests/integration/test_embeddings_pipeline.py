"""Integration tests for offline embeddings build stage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.application_layers.offline.build_embeddings import BuildEmbeddingsLayer
from app.domain.models.chunk import Chunk
from app.domain.models.embedded_chunk import EmbeddedChunk
from app.infrastructure.storage.jsonl_store import read_jsonl, write_jsonl
from app.utils.hashing import sha256_text


class DeterministicFakeEmbedder:
    """Fake embedder used to avoid loading sentence-transformers in tests."""

    def __init__(self, *, dim: int = 4, normalized: bool = True) -> None:
        self._dim = dim
        self._normalized = normalized
        self.embed_documents_calls = 0

    @staticmethod
    def _base_value(text: str) -> float:
        digest = sha256_text(text)
        return float(int(digest[:8], 16) % 1000)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_documents_calls += 1
        vectors: list[list[float]] = []
        for text in texts:
            base = self._base_value(text)
            vector = [base + float(offset) for offset in range(self._dim)]
            if self._normalized:
                norm = sum(value * value for value in vector) ** 0.5
                vector = [value / norm for value in vector]
            vectors.append(vector)
        return vectors

    def embed_query(self, query: str) -> list[float]:
        return self.embed_documents([query])[0]

    def embedding_dim(self) -> int:
        return self._dim

    def model_name(self) -> str:
        return "fake-embedder"


class PartiallyFailingFakeEmbedder(DeterministicFakeEmbedder):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        for text in texts:
            if "FAIL" in text:
                raise ValueError("Intentional embedding failure for test.")
        return super().embed_documents(texts)


def _make_chunk(*, chunk_id: str, document_id: str, text: str, chunk_index: int) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        text=text,
        chunk_index=chunk_index,
        token_count=max(1, len(text.split())),
        content_hash=sha256_text(text),
        metadata={
            "source_path": f"data/interim/{document_id}.md",
            "source_name": f"{document_id}.md",
            "source_type": "markdown",
            "title": f"Title {document_id}",
        },
    )


@pytest.mark.integration
class TestBuildEmbeddingsLayer:
    def test_run_creates_embedded_chunks_and_manifest_with_preserved_order(
        self, tmp_path: Path
    ) -> None:
        chunks = [
            _make_chunk(chunk_id="chunk-1", document_id="doc-1", text="alpha", chunk_index=0),
            _make_chunk(chunk_id="chunk-2", document_id="doc-1", text="beta", chunk_index=1),
            _make_chunk(chunk_id="chunk-3", document_id="doc-2", text="gamma", chunk_index=0),
        ]
        input_path = tmp_path / "chunks.jsonl"
        output_path = tmp_path / "embedded_chunks.jsonl"
        manifest_path = tmp_path / "embedding_manifest.json"
        cache_path = tmp_path / "embedding_cache"
        write_jsonl(input_path, chunks)

        fake_embedder = DeterministicFakeEmbedder(dim=4, normalized=True)
        result = BuildEmbeddingsLayer(
            input_path=input_path,
            output_path=output_path,
            manifest_path=manifest_path,
            cache_path=cache_path,
            embedder=fake_embedder,
        ).run()

        embedded_chunks = read_jsonl(output_path, EmbeddedChunk)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert output_path.exists()
        assert manifest_path.exists()
        assert result.chunks_total == len(chunks)
        assert result.embedded_chunks_total == len(chunks)
        assert result.failed_chunks_total == 0
        assert result.skipped_chunks_total == 0
        assert [item.chunk.id for item in embedded_chunks] == [item.id for item in chunks]
        assert [item.chunk.text for item in embedded_chunks] == [item.text for item in chunks]

        for embedded in embedded_chunks:
            assert embedded.embedding_dim == 4
            assert len(embedded.vector) == 4
            assert embedded.embedding_model == "fake-embedder"
            assert embedded.normalized is True

        assert manifest["stage"] == "embeddings"
        assert manifest["chunks_total"] == len(chunks)
        assert manifest["embedded_chunks_total"] == len(chunks)
        assert manifest["failed_chunks_total"] == 0
        assert manifest["cache_enabled"] is True
        assert manifest["cache_hits"] == 0
        assert manifest["cache_misses"] == len(chunks)
        assert manifest["vector_count"] == len(chunks)
        assert manifest["vector_dim"] == 4

    def test_limit_and_cache_hits_and_no_cache_mode(self, tmp_path: Path) -> None:
        chunks = [
            _make_chunk(chunk_id="chunk-1", document_id="doc-1", text="alpha", chunk_index=0),
            _make_chunk(chunk_id="chunk-2", document_id="doc-1", text="beta", chunk_index=1),
            _make_chunk(chunk_id="chunk-3", document_id="doc-2", text="gamma", chunk_index=0),
        ]
        input_path = tmp_path / "chunks.jsonl"
        output_path = tmp_path / "embedded_chunks.jsonl"
        manifest_path = tmp_path / "embedding_manifest.json"
        cache_path = tmp_path / "embedding_cache"
        write_jsonl(input_path, chunks)

        fake_embedder = DeterministicFakeEmbedder(dim=4, normalized=True)
        first_result = BuildEmbeddingsLayer(
            input_path=input_path,
            output_path=output_path,
            manifest_path=manifest_path,
            cache_path=cache_path,
            embedder=fake_embedder,
            limit=2,
        ).run()
        first_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert first_result.embedded_chunks_total == 2
        assert first_result.skipped_chunks_total == 1
        assert first_manifest["cache_hits"] == 0
        assert first_manifest["cache_misses"] == 2
        assert fake_embedder.embed_documents_calls == 1

        second_result = BuildEmbeddingsLayer(
            input_path=input_path,
            output_path=output_path,
            manifest_path=manifest_path,
            cache_path=cache_path,
            embedder=fake_embedder,
            limit=2,
        ).run()
        second_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert second_result.embedded_chunks_total == 2
        assert second_manifest["cache_hits"] == 2
        assert second_manifest["cache_misses"] == 0
        assert fake_embedder.embed_documents_calls == 1

        no_cache_result = BuildEmbeddingsLayer(
            input_path=input_path,
            output_path=output_path,
            manifest_path=manifest_path,
            cache_path=cache_path,
            embedder=fake_embedder,
            cache_enabled=False,
            limit=2,
        ).run()
        no_cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert no_cache_result.embedded_chunks_total == 2
        assert no_cache_manifest["cache_enabled"] is False
        assert no_cache_manifest["cache_hits"] == 0
        assert no_cache_manifest["cache_misses"] == 2

    def test_partial_failures_are_recorded_and_successes_are_written(
        self, tmp_path: Path
    ) -> None:
        chunks = [
            _make_chunk(chunk_id="chunk-1", document_id="doc-1", text="ok text", chunk_index=0),
            _make_chunk(chunk_id="chunk-2", document_id="doc-1", text="FAIL text", chunk_index=1),
            _make_chunk(chunk_id="chunk-3", document_id="doc-2", text="still ok", chunk_index=0),
        ]
        input_path = tmp_path / "chunks.jsonl"
        output_path = tmp_path / "embedded_chunks.jsonl"
        manifest_path = tmp_path / "embedding_manifest.json"
        cache_path = tmp_path / "embedding_cache"
        write_jsonl(input_path, chunks)

        result = BuildEmbeddingsLayer(
            input_path=input_path,
            output_path=output_path,
            manifest_path=manifest_path,
            cache_path=cache_path,
            embedder=PartiallyFailingFakeEmbedder(),
        ).run()

        embedded_chunks = read_jsonl(output_path, EmbeddedChunk)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert result.chunks_total == 3
        assert result.embedded_chunks_total == 2
        assert result.failed_chunks_total == 1
        assert [item.chunk.id for item in embedded_chunks] == ["chunk-1", "chunk-3"]
        assert manifest["failed_chunks_total"] == 1
        assert manifest["embedded_chunks_total"] == 2
        assert len(manifest["errors"]) == 1
        assert manifest["errors"][0]["chunk_id"] == "chunk-2"
