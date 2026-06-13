"""Unit tests for eval records, dataset helpers, and dataset builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.dataset_builder import (
    build_audit_subset,
    build_eval_dataset_from_chunks,
    build_negative_samples,
    build_robustness_samples,
    build_source_grounded_samples,
    extract_chunk_id,
    extract_chunk_text,
    extract_document_id,
    extract_section,
    extract_title,
    extract_token_count,
    extract_url,
    load_chunk_records,
)
from app.evaluation.datasets import (
    build_dataset_manifest,
    dedupe_eval_sample_ids,
    ensure_unique_eval_samples_file,
    export_eval_dataset_dir,
    filter_eval_samples,
    load_eval_samples,
)
from app.evaluation.records import (
    EvalDataset,
    EvalDifficulty,
    EvalExpectedBehavior,
    EvalQueryType,
    EvalSample,
    EvalSubset,
)


def _sample(
    *,
    sample_id: str = "eval_001",
    question: str = "What does Stripe docs say about disputes?",
    subset: EvalSubset = EvalSubset.SYNTHETIC_SOURCE_GROUNDED,
    query_type: EvalQueryType = EvalQueryType.FACTOID,
    behavior: EvalExpectedBehavior = EvalExpectedBehavior.ANSWER,
) -> EvalSample:
    return EvalSample(
        id=sample_id,
        question=question,
        subset=subset,
        type=query_type,
        difficulty=EvalDifficulty.EASY,
        expected_behavior=behavior,
        expected_chunk_ids=["chunk_a", "chunk_a", "chunk_b"],
        expected_document_ids=["doc_a"],
        expected_urls=["https://docs.stripe.com/x", "https://docs.stripe.com/x"],
        metadata={},
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@pytest.mark.unit
class TestEvalRecords:
    def test_eval_sample_validates_valid_sample(self) -> None:
        sample = _sample()
        assert sample.id == "eval_001"

    def test_empty_question_raises(self) -> None:
        with pytest.raises(ValueError):
            _sample(question="   ")

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValueError):
            _sample(sample_id="")

    def test_expected_lists_are_deduplicated(self) -> None:
        sample = _sample()
        assert sample.expected_chunk_ids == ["chunk_a", "chunk_b"]
        assert sample.expected_urls == ["https://docs.stripe.com/x"]

    def test_metadata_defaults_to_empty_dict(self) -> None:
        sample = EvalSample(
            id="eval_meta",
            question="What is Stripe Radar?",
            subset=EvalSubset.SYNTHETIC_SOURCE_GROUNDED,
            type=EvalQueryType.DEFINITION,
            difficulty=EvalDifficulty.EASY,
            expected_behavior=EvalExpectedBehavior.ANSWER,
            expected_chunk_ids=["chunk_1"],
            expected_document_ids=[],
            expected_urls=[],
        )
        assert sample.metadata == {}

    def test_negative_abstain_accepts_empty_expected_ids(self) -> None:
        sample = EvalSample(
            id="eval_neg",
            question="Will Stripe stock double next week?",
            subset=EvalSubset.NEGATIVE,
            type=EvalQueryType.OOD,
            difficulty=EvalDifficulty.MEDIUM,
            expected_behavior=EvalExpectedBehavior.ABSTAIN,
            expected_chunk_ids=[],
            expected_document_ids=[],
            expected_urls=[],
            reference_answer=None,
            metadata={},
        )
        assert sample.expected_chunk_ids == []

    def test_dataset_count_helpers_work(self) -> None:
        answer = _sample(sample_id="eval_answer", behavior=EvalExpectedBehavior.ANSWER)
        abstain = EvalSample(
            id="eval_abstain",
            question="What is the weather tomorrow?",
            subset=EvalSubset.NEGATIVE,
            type=EvalQueryType.OOD,
            difficulty=EvalDifficulty.MEDIUM,
            expected_behavior=EvalExpectedBehavior.ABSTAIN,
            expected_chunk_ids=[],
            expected_document_ids=[],
            expected_urls=[],
            metadata={},
        )
        dataset = EvalDataset(dataset_id="eval_ds", samples=[answer, abstain])
        assert len(dataset) == 2
        assert dataset.answerable_count() == 1
        assert dataset.abstain_count() == 1
        assert dataset.subset_counts()["negative"] == 1


@pytest.mark.unit
class TestEvalDatasets:
    def test_load_jsonl_eval_samples(self) -> None:
        fixture = (
            Path(__file__).resolve().parents[1] / "fixtures" / "sample_eval.jsonl"
        )
        samples = load_eval_samples(fixture)
        assert len(samples) == 2
        assert samples[0].id.startswith("eval_")

    def test_invalid_jsonl_raises_clear_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text(
            (
                '{"id":"eval_ok","question":"What is Stripe Radar?","subset":"synthetic_source_grounded",'
                '"type":"definition","difficulty":"easy","expected_behavior":"answer","expected_chunk_ids":["chunk_1"],'
                '"expected_document_ids":["doc_1"],"expected_urls":[],"reference_answer":null,"metadata":{}}\n'
                '{"id": bad-json}\n'
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_eval_samples(bad)

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_eval_samples(tmp_path / "missing.jsonl")

    def test_dedupe_eval_sample_ids_renames_duplicates(self) -> None:
        first = _sample(sample_id="eval_dup")
        second = _sample(sample_id="eval_dup", question="Another question")
        deduped, renamed_total = dedupe_eval_sample_ids([first, second])
        assert renamed_total == 1
        assert deduped[0].id == "eval_dup"
        assert deduped[1].id.startswith("eval_dup__dup")

    def test_ensure_unique_eval_samples_file_rewrites_dataset(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.jsonl"
        duplicated = _sample(sample_id="eval_dup")
        _write_jsonl(
            dataset_path,
            [
                duplicated.model_dump(mode="json"),
                duplicated.model_dump(mode="json"),
            ],
        )
        _, renamed_total = ensure_unique_eval_samples_file(dataset_path)
        assert renamed_total == 1
        ids = [sample.id for sample in load_eval_samples(dataset_path)]
        assert len(ids) == len(set(ids))

    def test_filter_by_subset(self) -> None:
        items = [_sample(sample_id="a"), _sample(sample_id="b", subset=EvalSubset.NEGATIVE)]
        filtered = filter_eval_samples(items, subsets={"negative"})
        assert [item.id for item in filtered] == ["b"]

    def test_filter_by_type(self) -> None:
        items = [
            _sample(sample_id="a", query_type=EvalQueryType.FACTOID),
            _sample(sample_id="b", query_type=EvalQueryType.HOW_TO),
        ]
        filtered = filter_eval_samples(items, types={"how_to"})
        assert [item.id for item in filtered] == ["b"]

    def test_filter_by_expected_behavior(self) -> None:
        items = [
            _sample(sample_id="a", behavior=EvalExpectedBehavior.ANSWER),
            _sample(sample_id="b", behavior=EvalExpectedBehavior.EITHER),
        ]
        filtered = filter_eval_samples(items, expected_behaviors={"either"})
        assert [item.id for item in filtered] == ["b"]

    def test_limit_validates_positive(self) -> None:
        with pytest.raises(ValueError, match="limit must be > 0"):
            filter_eval_samples([_sample()], limit=0)

    def test_seeded_shuffle_is_deterministic(self) -> None:
        items = [_sample(sample_id=f"eval_{idx}") for idx in range(10)]
        first = [s.id for s in filter_eval_samples(items, seed=42)]
        second = [s.id for s in filter_eval_samples(items, seed=42)]
        assert first == second

    def test_build_manifest_counts(self) -> None:
        samples = [
            _sample(sample_id="eval_a", query_type=EvalQueryType.FACTOID),
            _sample(
                sample_id="eval_b",
                subset=EvalSubset.NEGATIVE,
                query_type=EvalQueryType.OOD,
                behavior=EvalExpectedBehavior.ABSTAIN,
            ),
        ]
        manifest = build_dataset_manifest(dataset_id="eval_ds", samples=samples)
        assert manifest.samples_total == 2
        assert manifest.subsets["synthetic_source_grounded"] == 1
        assert manifest.subsets["negative"] == 1
        assert manifest.types["factoid"] == 1
        assert manifest.types["ood"] == 1

    def test_export_dataset_dir_writes_files(self, tmp_path: Path) -> None:
        source = _sample(sample_id="eval_src")
        audit = EvalSample(
            id="eval_audit",
            question=source.question,
            subset=EvalSubset.AUDIT,
            type=source.type,
            difficulty=source.difficulty,
            expected_behavior=source.expected_behavior,
            expected_chunk_ids=source.expected_chunk_ids,
            expected_document_ids=source.expected_document_ids,
            expected_urls=source.expected_urls,
            metadata={},
        )
        outputs = export_eval_dataset_dir(
            tmp_path / "dataset",
            dataset_id="my_eval",
            samples=[source, audit],
        )
        assert outputs["dataset"].name == "dataset.jsonl"
        assert outputs["manifest"].name == "manifest.json"
        assert outputs["audit"].name == "audit.jsonl"
        assert outputs["dataset"].exists()
        assert outputs["manifest"].exists()
        assert outputs["audit"].exists()


@pytest.mark.unit
class TestDatasetBuilder:
    def test_load_chunk_records_from_jsonl(self, tmp_path: Path) -> None:
        chunks_path = tmp_path / "chunks.jsonl"
        _write_jsonl(
            chunks_path,
            [{"id": "chunk_1", "document_id": "doc_1", "text": "a" * 400, "metadata": {}}],
        )
        rows = load_chunk_records(chunks_path)
        assert len(rows) == 1
        assert rows[0]["id"] == "chunk_1"

    def test_extract_fields_shape_a(self) -> None:
        record = {
            "id": "chunk_a",
            "document_id": "doc_a",
            "text": "hello world",
            "metadata": {"title": "Title A", "url": "https://a", "section": "Section A", "token_count": 123},
        }
        assert extract_chunk_id(record) == "chunk_a"
        assert extract_document_id(record) == "doc_a"
        assert extract_chunk_text(record) == "hello world"
        assert extract_title(record) == "Title A"
        assert extract_url(record) == "https://a"
        assert extract_section(record) == "Section A"
        assert extract_token_count(record) == 123

    def test_extract_fields_shape_b(self) -> None:
        record = {
            "chunk_id": "chunk_b",
            "document_id": "doc_b",
            "text": "hello world",
            "source": {"title": "Title B", "url": "https://b", "section": "Section B"},
            "metadata": {},
        }
        assert extract_chunk_id(record) == "chunk_b"
        assert extract_document_id(record) == "doc_b"
        assert extract_title(record) == "Title B"
        assert extract_url(record) == "https://b"
        assert extract_section(record) == "Section B"

    def test_low_quality_chunk_is_skipped(self) -> None:
        chunk = {"id": "chunk_short", "document_id": "doc_1", "text": "too short", "metadata": {}}
        samples = build_source_grounded_samples([chunk], min_chunk_chars=300)
        assert samples == []

    def test_source_grounded_samples_include_expected_fields(self) -> None:
        chunk = {
            "id": "chunk_1",
            "document_id": "doc_1",
            "text": "Stripe supports configuring webhooks and events. " * 20,
            "metadata": {"url": "https://docs.stripe.com/webhooks", "title": "Webhooks"},
        }
        samples = build_source_grounded_samples([chunk], target_size=1)
        assert len(samples) == 1
        sample = samples[0]
        assert sample.expected_chunk_ids == ["chunk_1"]
        assert sample.expected_document_ids == ["doc_1"]
        assert sample.expected_urls == ["https://docs.stripe.com/webhooks"]

    def test_generated_sample_ids_are_stable(self) -> None:
        chunk = {
            "id": "chunk_1",
            "document_id": "doc_1",
            "text": "Stripe supports creating payment intents with confirmations. " * 20,
            "metadata": {"title": "Payment intents"},
        }
        first = build_source_grounded_samples([chunk], target_size=1, seed=1)[0].id
        second = build_source_grounded_samples([chunk], target_size=1, seed=99)[0].id
        assert first == second

    def test_negative_samples_are_abstain_with_empty_expected_fields(self) -> None:
        negatives = build_negative_samples(target_size=5, seed=42)
        assert len(negatives) == 5
        for sample in negatives:
            assert sample.expected_behavior == EvalExpectedBehavior.ABSTAIN
            assert sample.expected_chunk_ids == []
            assert sample.expected_document_ids == []
            assert sample.expected_urls == []

    def test_robustness_typo_preserves_expected_ids(self) -> None:
        base = _sample(sample_id="base_1")
        robust = build_robustness_samples([base], target_size=1, seed=42)[0]
        assert robust.type == EvalQueryType.TYPO
        assert robust.expected_chunk_ids == base.expected_chunk_ids
        assert robust.expected_document_ids == base.expected_document_ids

    def test_audit_subset_is_deterministic(self) -> None:
        samples = [_sample(sample_id=f"eval_{idx}") for idx in range(10)]
        first = [s.id for s in build_audit_subset(samples, target_size=5, seed=7)]
        second = [s.id for s in build_audit_subset(samples, target_size=5, seed=7)]
        assert first == second

    def test_build_eval_dataset_from_chunks_returns_dataset(self, tmp_path: Path) -> None:
        chunks_path = tmp_path / "chunks.jsonl"
        _write_jsonl(
            chunks_path,
            [
                {
                    "id": "chunk_1",
                    "document_id": "doc_1",
                    "text": "Stripe webhooks let you receive event notifications securely. " * 20,
                    "metadata": {"title": "Webhooks", "url": "https://docs.stripe.com/webhooks"},
                }
            ],
        )
        dataset = build_eval_dataset_from_chunks(
            chunks_path=chunks_path,
            dataset_id="eval_test",
            synthetic_target_size=1,
            negative_target_size=1,
            robustness_target_size=1,
            audit_target_size=1,
            seed=42,
        )
        assert isinstance(dataset, EvalDataset)
        assert dataset.dataset_id == "eval_test"
        assert len(dataset.samples) == 4

    def test_build_eval_dataset_from_chunks_writes_files(self, tmp_path: Path) -> None:
        chunks_path = tmp_path / "chunks.jsonl"
        _write_jsonl(
            chunks_path,
            [
                {
                    "id": "chunk_1",
                    "document_id": "doc_1",
                    "text": "Stripe docs explain account verification and onboarding steps clearly. " * 20,
                    "metadata": {"title": "Onboarding", "url": "https://docs.stripe.com/connect"},
                }
            ],
        )
        output_dir = tmp_path / "out"
        build_eval_dataset_from_chunks(
            chunks_path=chunks_path,
            dataset_id="eval_out",
            output_dir=output_dir,
            synthetic_target_size=1,
            negative_target_size=1,
            robustness_target_size=1,
            audit_target_size=1,
            seed=42,
        )
        target = output_dir / "eval_out"
        assert (target / "dataset.jsonl").exists()
        assert (target / "manifest.json").exists()
        assert (target / "audit.jsonl").exists()

    def test_manifest_includes_source_artifact_and_build_config(self, tmp_path: Path) -> None:
        chunks_path = tmp_path / "chunks.jsonl"
        _write_jsonl(
            chunks_path,
            [
                {
                    "id": "chunk_1",
                    "document_id": "doc_1",
                    "text": "Stripe docs cover retries, idempotency keys, and API status handling. " * 20,
                    "metadata": {"title": "Reliability", "url": "https://docs.stripe.com/api"},
                }
            ],
        )
        dataset = build_eval_dataset_from_chunks(
            chunks_path=chunks_path,
            dataset_id="eval_manifest",
            synthetic_target_size=1,
            negative_target_size=1,
            robustness_target_size=1,
            audit_target_size=1,
            seed=21,
            min_chunk_chars=300,
        )
        assert dataset.manifest is not None
        assert dataset.manifest.source_artifacts["chunks_path"] == str(chunks_path)
        assert dataset.manifest.build_config["seed"] == 21
