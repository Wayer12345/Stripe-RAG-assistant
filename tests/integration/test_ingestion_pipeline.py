"""Integration tests for the offline parsed-documents build stage."""

import json
from pathlib import Path

import pytest
import yaml
from app.application_layers.offline.build_parsed_docs import BuildParsedDocsLayer
from app.domain.models.document import Document
from app.infrastructure.storage.jsonl_store import read_jsonl


def _write_temp_config(
    *,
    tmp_path: Path,
    input_dir: Path,
    output_path: Path,
    manifest_path: Path,
    extra_extensions: list[str] | None = None,
) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    base_config = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text(encoding="utf-8"))
    base_config["ingestion"]["input_dir"] = str(input_dir)
    base_config["ingestion"]["outputs"]["parsed_documents_path"] = str(output_path)
    base_config["ingestion"]["outputs"]["manifest_path"] = str(manifest_path)
    if extra_extensions:
        base_config["ingestion"]["supported_extensions"].extend(extra_extensions)

    config_path = tmp_path / "configs" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.mark.integration
class TestBuildParsedDocs:
    def test_build_creates_artifacts_and_parses_txt(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "guide.txt").write_text("Stripe docs text.", encoding="utf-8")

        output_path = tmp_path / "parsed_documents.jsonl"
        manifest_path = tmp_path / "ingestion_manifest.json"
        result = BuildParsedDocsLayer(
            input_dir=raw_dir,
            output_path=output_path,
            manifest_path=manifest_path,
        ).run()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert output_path.exists()
        assert manifest_path.exists()
        assert result.parsed_documents_total >= 1
        assert manifest["parsed_documents_total"] == result.parsed_documents_total
        assert manifest["failed_documents_total"] == 0

    def test_multiple_files_produce_multiple_documents(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "a.txt").write_text("A", encoding="utf-8")
        (raw_dir / "b.txt").write_text("B", encoding="utf-8")

        output_path = tmp_path / "parsed_documents.jsonl"
        result = BuildParsedDocsLayer(
            input_dir=raw_dir,
            output_path=output_path,
            manifest_path=tmp_path / "ingestion_manifest.json",
        ).run()
        docs = read_jsonl(output_path, Document)

        assert result.raw_documents_total == 2
        assert result.parsed_documents_total == 2
        assert len(docs) == 2

    def test_broken_file_is_recorded_but_successful_file_is_still_written(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "ok.txt").write_text("Valid content", encoding="utf-8")
        (raw_dir / "broken.txt").write_text("   \n\t  ", encoding="utf-8")

        output_path = tmp_path / "parsed_documents.jsonl"
        manifest_path = tmp_path / "ingestion_manifest.json"
        result = BuildParsedDocsLayer(
            input_dir=raw_dir,
            output_path=output_path,
            manifest_path=manifest_path,
        ).run()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        docs = read_jsonl(output_path, Document)

        assert result.failed_documents_total == 1
        assert result.parsed_documents_total == 1
        assert manifest["failed_documents_total"] == result.failed_documents_total
        assert len(manifest["errors"]) == 1
        assert len(docs) == 1

    def test_unsupported_file_type_is_recorded_in_manifest_errors(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "ok.txt").write_text("good", encoding="utf-8")
        (raw_dir / "unknown.unsupported").write_text("unsupported", encoding="utf-8")

        output_path = tmp_path / "parsed_documents.jsonl"
        manifest_path = tmp_path / "ingestion_manifest.json"
        config_path = _write_temp_config(
            tmp_path=tmp_path,
            input_dir=raw_dir,
            output_path=output_path,
            manifest_path=manifest_path,
            extra_extensions=[".unsupported"],
        )
        result = BuildParsedDocsLayer(config_path=config_path).run()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert manifest["failed_documents_total"] == 1
        assert manifest["parsed_documents_total"] == 1
        assert result.failed_documents_total == 1
        assert manifest["errors"][0]["source_type"] == "unsupported"
        assert manifest["errors"][0]["error_type"] == "ValueError"

    def test_jsonl_round_trip_returns_document_models(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "doc.txt").write_text("hello", encoding="utf-8")

        output_path = tmp_path / "parsed_documents.jsonl"
        BuildParsedDocsLayer(
            input_dir=raw_dir,
            output_path=output_path,
            manifest_path=tmp_path / "ingestion_manifest.json",
        ).run()
        docs = read_jsonl(output_path, Document)

        assert docs
        assert isinstance(docs[0], Document)

    def test_manifest_counts_are_consistent(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "ok.txt").write_text("ok", encoding="utf-8")
        (raw_dir / "bad.txt").write_text("\n\n", encoding="utf-8")

        manifest_path = tmp_path / "ingestion_manifest.json"
        result = BuildParsedDocsLayer(
            input_dir=raw_dir,
            output_path=tmp_path / "parsed_documents.jsonl",
            manifest_path=manifest_path,
        ).run()
        on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert on_disk["run_id"] == result.run_id
        assert on_disk["failed_documents_total"] == len(on_disk["errors"])
        assert on_disk["raw_documents_total"] >= (
            on_disk["failed_documents_total"] + on_disk["skipped_documents_total"]
        )
