"""Integration tests for the offline cleaned-documents build stage."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from app.application_layers.offline.build_cleaned_docs import BuildCleanedDocsLayer
from app.domain.models.document import Document, DocumentProcessingStage
from app.infrastructure.storage.jsonl_store import read_jsonl, write_jsonl
from app.utils.hashing import sha256_text


def _make_parsed_document(
    *,
    document_id: str,
    text: str,
    source_path: str,
    source_name: str,
) -> Document:
    return Document(
        id=document_id,
        source_type="txt",
        source_path=source_path,
        source_name=source_name,
        title=source_name,
        processing_stage=DocumentProcessingStage.PARSED,
        text=text,
        metadata={"source": "integration_test"},
        content_hash=sha256_text(text),
        created_at=datetime.now(UTC),
    )


@pytest.mark.integration
class TestBuildCleanedDocsLayer:
    def test_run_creates_cleaned_jsonl_and_manifest_and_cleans_document(
        self, tmp_path: Path
    ) -> None:
        original_text = "<div>Title</div>\n\nSome   text   here."
        parsed_document = _make_parsed_document(
            document_id="doc-1",
            text=original_text,
            source_path="raw/guide_1.txt",
            source_name="guide_1.txt",
        )
        input_path = tmp_path / "parsed_documents.jsonl"
        output_path = tmp_path / "cleaned_documents.jsonl"
        manifest_path = tmp_path / "cleaning_manifest.json"
        write_jsonl(input_path, [parsed_document])

        result = BuildCleanedDocsLayer(
            input_path=input_path,
            output_path=output_path,
            manifest_path=manifest_path,
        ).run()

        cleaned_documents = read_jsonl(output_path, Document)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert output_path.exists()
        assert manifest_path.exists()
        assert result.parsed_documents_total == 1
        assert result.cleaned_documents_total == 1
        assert result.failed_documents_total == 0
        assert len(cleaned_documents) == 1

        cleaned = cleaned_documents[0]
        assert cleaned.processing_stage == DocumentProcessingStage.CLEANED
        assert "cleaning" in cleaned.metadata
        assert cleaned.text != original_text
        assert cleaned.content_hash != parsed_document.content_hash
        assert isinstance(cleaned, Document)

        assert manifest["stage"] == "cleaning"
        assert manifest["cleaner_name"] == "TextCleaner"
        assert manifest["parsed_documents_total"] == 1
        assert manifest["cleaned_documents_total"] == 1
        assert manifest["failed_documents_total"] == 0

    def test_multiple_parsed_documents_produce_multiple_cleaned_documents(
        self, tmp_path: Path
    ) -> None:
        docs = [
            _make_parsed_document(
                document_id="doc-1",
                text="First  text",
                source_path="raw/a.txt",
                source_name="a.txt",
            ),
            _make_parsed_document(
                document_id="doc-2",
                text="Second\t\ttext",
                source_path="raw/b.txt",
                source_name="b.txt",
            ),
        ]
        input_path = tmp_path / "parsed_documents.jsonl"
        output_path = tmp_path / "cleaned_documents.jsonl"
        manifest_path = tmp_path / "cleaning_manifest.json"
        write_jsonl(input_path, docs)

        result = BuildCleanedDocsLayer(
            input_path=input_path,
            output_path=output_path,
            manifest_path=manifest_path,
        ).run()
        cleaned_documents = read_jsonl(output_path, Document)

        assert result.parsed_documents_total == 2
        assert result.cleaned_documents_total == 2
        assert len(cleaned_documents) == 2

    def test_failed_document_is_recorded_and_successful_document_still_written(
        self, tmp_path: Path
    ) -> None:
        valid = _make_parsed_document(
            document_id="doc-valid",
            text="Alpha   beta",
            source_path="raw/ok.txt",
            source_name="ok.txt",
        )
        # This passes Document validation but becomes empty after boilerplate removal.
        bad = _make_parsed_document(
            document_id="doc-bad",
            text="Sign in",
            source_path="raw/bad.txt",
            source_name="bad.txt",
        )
        input_path = tmp_path / "parsed_documents.jsonl"
        output_path = tmp_path / "cleaned_documents.jsonl"
        manifest_path = tmp_path / "cleaning_manifest.json"
        write_jsonl(input_path, [valid, bad])

        result = BuildCleanedDocsLayer(
            input_path=input_path,
            output_path=output_path,
            manifest_path=manifest_path,
        ).run()

        cleaned_documents = read_jsonl(output_path, Document)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert result.cleaned_documents_total == 1
        assert result.failed_documents_total == 1
        assert len(cleaned_documents) == 1
        assert manifest["failed_documents_total"] == 1
        assert len(manifest["errors"]) == 1
        assert manifest["errors"][0]["document_id"] == "doc-bad"
        assert manifest["errors"][0]["source_path"] == "raw/bad.txt"
        assert manifest["errors"][0]["error_type"] == "ValueError"

        assert manifest["parsed_documents_total"] == 2
        assert manifest["cleaned_documents_total"] == 1
        assert manifest["skipped_documents_total"] == 0
        assert (
            manifest["cleaned_documents_total"] + manifest["failed_documents_total"]
            == manifest["parsed_documents_total"] - manifest["skipped_documents_total"]
        )

    def test_limit_skips_documents_and_updates_manifest_counts(
        self, tmp_path: Path
    ) -> None:
        docs = [
            _make_parsed_document(
                document_id="doc-1",
                text="Doc one",
                source_path="raw/one.txt",
                source_name="one.txt",
            ),
            _make_parsed_document(
                document_id="doc-2",
                text="Doc two",
                source_path="raw/two.txt",
                source_name="two.txt",
            ),
            _make_parsed_document(
                document_id="doc-3",
                text="Doc three",
                source_path="raw/three.txt",
                source_name="three.txt",
            ),
        ]
        input_path = tmp_path / "parsed_documents.jsonl"
        output_path = tmp_path / "cleaned_documents.jsonl"
        manifest_path = tmp_path / "cleaning_manifest.json"
        write_jsonl(input_path, docs)

        result = BuildCleanedDocsLayer(
            input_path=input_path,
            output_path=output_path,
            manifest_path=manifest_path,
            limit=2,
        ).run()

        cleaned_documents = read_jsonl(output_path, Document)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert result.parsed_documents_total == 3
        assert result.cleaned_documents_total == 2
        assert result.skipped_documents_total == 1
        assert len(cleaned_documents) == 2

        assert manifest["parsed_documents_total"] == 3
        assert manifest["cleaned_documents_total"] == 2
        assert manifest["failed_documents_total"] == 0
        assert manifest["skipped_documents_total"] == 1
