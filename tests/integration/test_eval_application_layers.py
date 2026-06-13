"""Integration-style tests for eval application-layer orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from app.application_layers.eval.build_eval_dataset import BuildEvalDatasetLayer
from app.application_layers.eval.run_citation_eval import RunCitationEvalLayer
from app.application_layers.eval.run_context_eval import RunContextEvalLayer
from app.application_layers.eval.run_generation_eval import RunGenerationEvalLayer
from app.application_layers.eval.run_rerank_eval import RunRerankEvalLayer
from app.application_layers.eval.run_retrieval_eval import RunRetrievalEvalLayer
from app.application_layers.eval.run_robustness_eval import RunRobustnessEvalLayer
from app.evaluation.records import EvalCaseResult


def _write_chunks(path: Path) -> None:
    record = {
        "id": "chunk_1",
        "document_id": "doc_1",
        "text": "Stripe docs explain webhook endpoints, event delivery, and signatures. " * 20,
        "metadata": {"title": "Webhooks", "url": "https://docs.stripe.com/webhooks"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def _fake_case(sample_id: str) -> EvalCaseResult:
    return EvalCaseResult(
        sample_id=sample_id,
        question="How do Stripe webhook signatures work for event verification?",
        subset="synthetic_source_grounded",
        type="factoid",
        difficulty="easy",
        expected_behavior="answer",
        expected_chunk_ids=["chunk_1"],
        expected_document_ids=["doc_1"],
        expected_urls=["https://docs.stripe.com/webhooks"],
        metrics={
            "retrieval.chunk_hit_at_1": 0.31,
            "retrieval.chunk_recall_at_10": 0.87,
            "retrieval.chunk_mrr_at_10": 0.53,
            "retrieval.document_recall_at_10": 0.88,
            "retrieval.url_recall_at_10": 0.83,
            "rerank.mrr_delta": 0.12,
            "rerank.kept_rate": 0.9,
            "rerank.recall_after_at_10": 0.89,
            "context.context_chunk_recall": 0.77,
            "context.context_document_recall": 0.79,
            "context.expected_chunk_dropped_rate": 0.05,
            "generation.parsed_successfully": 1.0,
            "generation.valid_generation_output": 0.96,
            "generation.no_answer": 0.04,
            "citation.valid_citation_rate": 0.93,
            "citation.invented_source_rate": 0.03,
            "citation.citation_recall": 0.81,
            "confidence.abstained": 0.18,
            "confidence.answer_on_unanswerable": 0.02,
            "robustness.ood_abstained": 0.84,
            "robustness.typo_answered": 0.74,
            "robustness.adversarial_valid_sources": 0.81,
        },
        latency_ms={"retrieve": 12.0, "rerank": 10.0, "context": 16.0, "generation": 20.0, "total": 58.0},
        passed=True,
    )


@pytest.mark.integration
class TestEvalApplicationLayers:
    def test_build_eval_dataset_layer_builds_dataset(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        chunks_path = tmp_path / "chunks.jsonl"
        _write_chunks(chunks_path)
        result = BuildEvalDatasetLayer(
            chunks_path=chunks_path,
            output_dir=tmp_path / "datasets",
            dataset_id="test_dataset",
            synthetic_target_size=1,
            negative_target_size=1,
            robustness_target_size=1,
            audit_target_size=1,
            config_path=Path("configs/config.yaml"),
        ).run()
        assert result.dataset_path.exists()
        assert result.manifest_path.exists()
        assert result.samples_total >= 1
        assert "Starting eval dataset build" in caplog.text
        assert "Eval dataset build result" in caplog.text

    def test_build_eval_dataset_validates_missing_chunks(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.ERROR)
        with pytest.raises(FileNotFoundError):
            BuildEvalDatasetLayer(chunks_path=tmp_path / "missing.jsonl").run()
        assert "chunks path not found" in caplog.text

    def test_run_retrieval_eval_layer_writes_artifacts(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        dataset = Path("tests/fixtures/sample_eval.jsonl")

        captured: dict[str, Any] = {}

        def fake_runner(samples: list[Any], *, mode: str, options: Any, **_: Any) -> tuple[list[EvalCaseResult], list[dict[str, Any]]]:
            captured["mode"] = mode
            return [_fake_case(sample.id) for sample in samples], []

        result = RunRetrievalEvalLayer(
            dataset_path=dataset,
            runs_dir=tmp_path,
            run_id="retrieval_run",
            runner_fn=fake_runner,
        ).run()
        assert captured["mode"] == "retrieval"
        assert result.report_path.exists()
        assert "Starting retrieval eval" in caplog.text
        assert "Retrieval eval metrics summary" in caplog.text
        assert "chunk_recall_at_10" in caplog.text
        assert "0.8700" in caplog.text

    def test_stage_layers_pass_expected_modes(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        dataset = Path("tests/fixtures/sample_eval.jsonl")
        seen: dict[str, str] = {}

        def fake_runner(samples: list[Any], *, mode: str, options: Any, **_: Any) -> tuple[list[EvalCaseResult], list[dict[str, Any]]]:
            seen["mode"] = mode
            return [_fake_case(sample.id) for sample in samples], []

        assert (
            RunRerankEvalLayer(dataset_path=dataset, runs_dir=tmp_path, run_id="rr", runner_fn=fake_runner).run().mode
            == "rerank"
        )
        assert seen["mode"] == "rerank"
        assert "Rerank eval metrics summary" in caplog.text
        assert "mrr_delta" in caplog.text
        assert "0.1200" in caplog.text
        assert (
            RunContextEvalLayer(dataset_path=dataset, runs_dir=tmp_path, run_id="ctx", runner_fn=fake_runner).run().mode
            == "context"
        )
        assert seen["mode"] == "context"
        assert "Context eval metrics summary" in caplog.text
        assert "context_chunk_recall" in caplog.text
        assert "0.7700" in caplog.text
        assert (
            RunGenerationEvalLayer(dataset_path=dataset, runs_dir=tmp_path, run_id="gen", runner_fn=fake_runner).run().mode
            == "generation"
        )
        assert seen["mode"] == "generation"
        assert "Generation eval metrics summary" in caplog.text
        assert "valid_generation_output" in caplog.text
        assert "0.9600" in caplog.text
        assert (
            RunCitationEvalLayer(dataset_path=dataset, runs_dir=tmp_path, run_id="cit", runner_fn=fake_runner).run().mode
            == "citation"
        )
        assert seen["mode"] == "citation"
        assert "Citation eval metrics summary" in caplog.text
        assert "invented_source_rate" in caplog.text
        assert "0.0300" in caplog.text
        assert (
            RunRobustnessEvalLayer(dataset_path=dataset, runs_dir=tmp_path, run_id="rob", runner_fn=fake_runner).run().mode
            == "robustness"
        )
        assert seen["mode"] == "robustness"
        assert "Robustness eval metrics summary" in caplog.text
        assert "ood_abstained" in caplog.text
        assert "0.8400" in caplog.text

    def test_limit_filtering_works(self, tmp_path: Path) -> None:
        dataset = Path("tests/fixtures/sample_eval.jsonl")

        def fake_runner(samples: list[Any], *, mode: str, options: Any, **_: Any) -> tuple[list[EvalCaseResult], list[dict[str, Any]]]:
            return [_fake_case(sample.id) for sample in samples], []

        result = RunRetrievalEvalLayer(
            dataset_path=dataset,
            runs_dir=tmp_path,
            run_id="limited",
            limit=1,
            runner_fn=fake_runner,
        ).run()
        assert result.cases_total == 1

    def test_empty_filtered_dataset_raises(self, tmp_path: Path) -> None:
        dataset = Path("tests/fixtures/sample_eval.jsonl")
        with pytest.raises(ValueError, match="zero samples"):
            RunRetrievalEvalLayer(
                dataset_path=dataset,
                runs_dir=tmp_path,
                run_id="empty",
                subsets={"does_not_exist"},
                runner_fn=lambda samples, **kwargs: ([], []),
            ).run()

    def test_missing_metrics_are_logged_as_none(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        dataset = Path("tests/fixtures/sample_eval.jsonl")

        def fake_runner(samples: list[Any], *, mode: str, options: Any, **_: Any) -> tuple[list[EvalCaseResult], list[dict[str, Any]]]:
            return [
                EvalCaseResult(
                    sample_id=sample.id,
                    question="Short question",
                    subset="synthetic_source_grounded",
                    type="factoid",
                    difficulty="easy",
                    expected_behavior="answer",
                    expected_chunk_ids=[],
                    expected_document_ids=[],
                    expected_urls=[],
                    metrics={},
                    latency_ms={"total": 1.0},
                    passed=True,
                )
                for sample in samples
            ], []

        RunGenerationEvalLayer(
            dataset_path=dataset,
            runs_dir=tmp_path,
            run_id="missing_metrics",
            runner_fn=fake_runner,
        ).run()
        assert "Generation eval metrics summary" in caplog.text
        assert "valid_generation_output=None" in caplog.text

