"""Unit tests for eval report artifact builders."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.records import EvalCaseResult
from app.evaluation.reports import (
    build_eval_run_manifest,
    build_eval_run_paths,
    build_eval_run_summary,
    build_failure_rows,
    build_metrics_summary,
    build_report_markdown,
    build_worst_cases,
    load_eval_run_artifacts,
    serialize_eval_case_result,
    write_eval_run_artifacts,
)


def _result(
    *,
    sample_id: str,
    retrieval_score: float = 0.8,
    context_score: float = 0.8,
    invented_source_rate: float = 0.0,
    total_latency_ms: float = 100.0,
    error: str | None = None,
    passed: bool = True,
) -> EvalCaseResult:
    return EvalCaseResult(
        sample_id=sample_id,
        question="How does Stripe docs explain webhooks?",
        subset="synthetic_source_grounded",
        type="factoid",
        difficulty="easy",
        expected_behavior="answer",
        expected_chunk_ids=["chunk_1"],
        expected_document_ids=["doc_1"],
        expected_urls=["https://docs.stripe.com/webhooks"],
        metrics={
            "retrieval.chunk_recall_at_10": retrieval_score,
            "context.context_chunk_recall": context_score,
            "citation.invented_source_rate": invented_source_rate,
            "generation.valid_generation_output": 1.0,
        },
        trace_paths={"retrieve": "trace_retrieve.json"},
        latency_ms={"retrieve": 10.0, "rerank": 20.0, "context": 30.0, "generation": 40.0, "total": total_latency_ms},
        passed=passed,
        error=error,
    )


@pytest.mark.unit
class TestEvaluationReports:
    def test_build_eval_run_paths(self, tmp_path: Path) -> None:
        paths = build_eval_run_paths(runs_dir=tmp_path, run_id="run_1")
        assert paths.run_dir.endswith("run_1")
        assert paths.cases_path is not None and paths.cases_path.endswith("cases.jsonl")

    def test_serialize_eval_case_result(self) -> None:
        payload = serialize_eval_case_result(_result(sample_id="s1"))
        assert payload["sample_id"] == "s1"
        assert isinstance(payload["metrics"], dict)

    def test_build_failure_rows_extracts_failed_cases(self) -> None:
        rows = build_failure_rows(
            [_result(sample_id="ok"), _result(sample_id="bad", error="RuntimeError: boom", passed=False)]
        )
        assert len(rows) == 1
        assert rows[0]["sample_id"] == "bad"

    def test_build_metrics_summary_groups_namespaced_metrics(self) -> None:
        summary = build_metrics_summary([_result(sample_id="s1"), _result(sample_id="s2")])
        assert "retrieval" in summary
        assert "context" in summary
        assert "retrieval.chunk_recall_at_10_mean" not in summary

    def test_build_metrics_summary_computes_stats(self) -> None:
        summary = build_metrics_summary(
            [_result(sample_id="s1", retrieval_score=0.2), _result(sample_id="s2", retrieval_score=0.8)]
        )
        retrieval = summary["retrieval"]
        assert retrieval["chunk_recall_at_10_mean"] == pytest.approx(0.5)
        assert retrieval["chunk_recall_at_10_min"] == pytest.approx(0.2)
        assert retrieval["chunk_recall_at_10_max"] == pytest.approx(0.8)
        assert retrieval["chunk_recall_at_10_count"] == 2.0

    def test_build_metrics_summary_handles_empty_results(self) -> None:
        summary = build_metrics_summary([])
        assert "latency" in summary
        assert summary["latency"]["total_mean_ms"] == 0.0

    def test_build_worst_cases_identifies_lowest_score(self) -> None:
        worst = build_worst_cases(
            [_result(sample_id="good", retrieval_score=1.0), _result(sample_id="bad", retrieval_score=0.1)]
        )
        assert worst["lowest_overall_score"][0]["sample_id"] == "bad"

    def test_build_worst_cases_identifies_highest_latency(self) -> None:
        worst = build_worst_cases(
            [_result(sample_id="slow", total_latency_ms=900.0), _result(sample_id="fast", total_latency_ms=90.0)]
        )
        assert worst["highest_latency"][0]["sample_id"] == "slow"

    def test_build_eval_run_summary_counts(self) -> None:
        results = [_result(sample_id="ok"), _result(sample_id="bad", error="oops", passed=False)]
        summary = build_eval_run_summary(run_id="run_1", results=results, metrics=build_metrics_summary(results))
        assert summary.cases_total == 2
        assert summary.cases_successful == 1
        assert summary.cases_failed == 1

    def test_build_eval_run_manifest_includes_artifact_paths(self, tmp_path: Path) -> None:
        paths = build_eval_run_paths(runs_dir=tmp_path, run_id="run_1")
        manifest = build_eval_run_manifest(
            run_id="run_1",
            dataset_id="dataset_1",
            dataset_path="dataset.jsonl",
            run_paths=paths,
        )
        assert manifest.artifact_paths is not None
        assert manifest.artifact_paths.manifest_path.endswith("manifest.json")

    def test_build_report_markdown_contains_sections(self, tmp_path: Path) -> None:
        results = [_result(sample_id="s1")]
        paths = build_eval_run_paths(runs_dir=tmp_path, run_id="run_1")
        metrics = build_metrics_summary(results)
        summary = build_eval_run_summary(run_id="run_1", results=results, metrics=metrics)
        manifest = build_eval_run_manifest(
            run_id="run_1",
            dataset_id="dataset_1",
            dataset_path="dataset.jsonl",
            run_paths=paths,
        )
        markdown = build_report_markdown(
            manifest=manifest,
            summary=summary,
            metrics=metrics,
            worst_cases=build_worst_cases(results),
            failures=build_failure_rows(results),
        )
        assert "# Eval Report" in markdown
        assert "## Summary" in markdown
        assert "## Metric Groups" in markdown

    def test_write_eval_run_artifacts_writes_required_files(self, tmp_path: Path) -> None:
        results = [_result(sample_id="s1"), _result(sample_id="s2")]
        paths = write_eval_run_artifacts(run_dir=tmp_path, run_id="run_1", results=results)
        for key in ("manifest", "cases", "metrics", "summary", "failures", "worst_cases", "report"):
            assert Path(paths[key]).exists()

    def test_write_eval_run_artifacts_with_empty_results(self, tmp_path: Path) -> None:
        paths = write_eval_run_artifacts(run_dir=tmp_path, run_id="run_empty", results=[])
        loaded = load_eval_run_artifacts(Path(paths["run_dir"]))
        assert loaded["summary"]["cases_total"] == 0
        assert loaded["failures"] == []

    def test_load_eval_run_artifacts_reads_written_payload(self, tmp_path: Path) -> None:
        write_eval_run_artifacts(
            run_dir=tmp_path,
            run_id="run_2",
            results=[_result(sample_id="s1")],
            dataset_id="dataset_1",
        )
        loaded = load_eval_run_artifacts(tmp_path / "run_2")
        assert loaded["manifest"]["run_id"] == "run_2"
        assert len(loaded["cases"]) == 1
