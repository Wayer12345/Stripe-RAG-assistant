"""Integration tests for offline chunking build stage."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pytest
from app.application_layers.offline.build_chunks import BuildChunksLayer
from app.domain.models.chunk import Chunk
from app.domain.models.document import Document, DocumentProcessingStage
from app.infrastructure.storage.jsonl_store import read_jsonl, write_jsonl
from app.utils.hashing import sha256_text


def _make_cleaned_document(
    *,
    document_id: str,
    text: str,
    source_path: str,
    source_name: str,
    title: str,
) -> Document:
    return Document(
        id=document_id,
        source_type="markdown",
        source_path=source_path,
        source_name=source_name,
        title=title,
        url=f"https://example.com/{document_id}",
        processing_stage=DocumentProcessingStage.CLEANED,
        text=text,
        metadata={"source": "integration_test"},
        content_hash=sha256_text(text),
        created_at=datetime.now(UTC),
    )


@pytest.mark.integration
class TestBuildChunksLayer:
    def test_run_creates_chunks_and_manifest_and_preserves_traceability(
        self, tmp_path: Path
    ) -> None:
        short_doc = _make_cleaned_document(
            document_id="doc-short",
            text="# Intro\n\nStripe lets businesses accept payments.",
            source_path="data/interim/doc-short.md",
            source_name="doc-short.md",
            title="Short Guide",
        )
        long_text = (
            "# Billing Guide\n\n"
            + "Stripe Billing supports subscriptions and invoices. " * 80
            + "\n\n## Details\n\n"
            + "You can configure trials, coupons, and proration behavior. " * 80
        )
        long_doc = _make_cleaned_document(
            document_id="doc-long",
            text=long_text,
            source_path="data/interim/doc-long.md",
            source_name="doc-long.md",
            title="Long Guide",
        )

        input_path = tmp_path / "cleaned_documents.jsonl"
        output_path = tmp_path / "chunks.jsonl"
        manifest_path = tmp_path / "chunking_manifest.json"
        write_jsonl(input_path, [short_doc, long_doc])

        result = BuildChunksLayer(
            input_path=input_path,
            output_path=output_path,
            manifest_path=manifest_path,
        ).run()

        chunks = read_jsonl(output_path, Chunk)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert output_path.exists()
        assert manifest_path.exists()
        assert result.cleaned_documents_total == 2
        assert result.chunks_total == len(chunks)
        assert result.failed_documents_total == 0
        assert result.skipped_documents_total == 0
        assert len(chunks) >= 2

        ids_by_doc: dict[str, list[str]] = defaultdict(list)
        indexes_by_doc: dict[str, list[int]] = defaultdict(list)
        for chunk in chunks:
            ids_by_doc[chunk.document_id].append(chunk.id)
            indexes_by_doc[chunk.document_id].append(chunk.chunk_index)

            assert chunk.text.strip()
            assert chunk.metadata["source_path"] in {
                "data/interim/doc-short.md",
                "data/interim/doc-long.md",
            }
            assert chunk.metadata["source_name"] in {"doc-short.md", "doc-long.md"}
            assert chunk.metadata["title"] in {"Short Guide", "Long Guide"}
            assert chunk.char_start is not None
            assert chunk.char_end is not None
            assert chunk.char_start <= chunk.char_end

        assert set(ids_by_doc) == {"doc-short", "doc-long"}
        assert len(ids_by_doc["doc-short"]) == 1
        assert len(ids_by_doc["doc-long"]) > 1
        assert indexes_by_doc["doc-short"] == list(range(len(indexes_by_doc["doc-short"])))
        assert indexes_by_doc["doc-long"] == list(range(len(indexes_by_doc["doc-long"])))

        assert manifest["stage"] == "chunking"
        assert manifest["cleaned_documents_total"] == 2
        assert manifest["chunks_total"] == len(chunks)
        assert manifest["failed_documents_total"] == 0
        assert manifest["skipped_documents_total"] == 0
        assert manifest["chunking_strategy"] in {"semantic", "structure_aware"}
        assert manifest["chunking_options"]["use_semantic_boundaries"] is False
        assert manifest["documents_with_zero_chunks"] == 0
        assert (
            manifest["failed_documents_total"] + manifest["skipped_documents_total"]
            <= manifest["cleaned_documents_total"]
        )

    def test_chunk_ids_are_deterministic_across_repeated_runs(
        self, tmp_path: Path
    ) -> None:
        document = _make_cleaned_document(
            document_id="doc-deterministic",
            text="# Title\n\n" + ("Deterministic chunking should be stable. " * 120),
            source_path="data/interim/doc-deterministic.md",
            source_name="doc-deterministic.md",
            title="Deterministic Guide",
        )
        input_path = tmp_path / "cleaned_documents.jsonl"
        output_path_1 = tmp_path / "chunks_run_1.jsonl"
        output_path_2 = tmp_path / "chunks_run_2.jsonl"
        manifest_path_1 = tmp_path / "manifest_run_1.json"
        manifest_path_2 = tmp_path / "manifest_run_2.json"
        write_jsonl(input_path, [document])

        BuildChunksLayer(
            input_path=input_path,
            output_path=output_path_1,
            manifest_path=manifest_path_1,
        ).run()
        BuildChunksLayer(
            input_path=input_path,
            output_path=output_path_2,
            manifest_path=manifest_path_2,
        ).run()

        chunks_run_1 = read_jsonl(output_path_1, Chunk)
        chunks_run_2 = read_jsonl(output_path_2, Chunk)
        ids_run_1 = [chunk.id for chunk in chunks_run_1]
        ids_run_2 = [chunk.id for chunk in chunks_run_2]

        assert ids_run_1
        assert ids_run_1 == ids_run_2
        assert [chunk.text for chunk in chunks_run_1] == [chunk.text for chunk in chunks_run_2]

    def test_limit_skips_documents_and_updates_manifest_counts(
        self, tmp_path: Path
    ) -> None:
        docs = [
            _make_cleaned_document(
                document_id="doc-1",
                text="# One\n\nFirst document text.",
                source_path="data/interim/doc-1.md",
                source_name="doc-1.md",
                title="Doc One",
            ),
            _make_cleaned_document(
                document_id="doc-2",
                text="# Two\n\nSecond document text.",
                source_path="data/interim/doc-2.md",
                source_name="doc-2.md",
                title="Doc Two",
            ),
            _make_cleaned_document(
                document_id="doc-3",
                text="# Three\n\nThird document text.",
                source_path="data/interim/doc-3.md",
                source_name="doc-3.md",
                title="Doc Three",
            ),
        ]
        input_path = tmp_path / "cleaned_documents.jsonl"
        output_path = tmp_path / "chunks.jsonl"
        manifest_path = tmp_path / "chunking_manifest.json"
        write_jsonl(input_path, docs)

        result = BuildChunksLayer(
            input_path=input_path,
            output_path=output_path,
            manifest_path=manifest_path,
            limit=2,
        ).run()

        chunks = read_jsonl(output_path, Chunk)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunk_doc_ids = {chunk.document_id for chunk in chunks}

        assert result.cleaned_documents_total == 3
        assert result.skipped_documents_total == 1
        assert chunk_doc_ids == {"doc-1", "doc-2"}

        assert manifest["cleaned_documents_total"] == 3
        assert manifest["skipped_documents_total"] == 1
        assert manifest["failed_documents_total"] == 0
        assert manifest["chunks_total"] == len(chunks)
