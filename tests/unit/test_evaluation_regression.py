"""Unit tests for eval regression comparison helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.regression import (
    RegressionGate,
    apply_regression_gates,
    build_regression_report_markdown,
    compare_eval_runs,
    compute_metric_deltas,
    default_regression_gates,
    flatten_metrics,
    get_nested_metric,
    write_regression_report,
)
from app.evaluation.reports import write_eval_run_artifacts
from app.evaluation.records import EvalCaseResult


def _result(sample_id: str, *, retrieval: float, latency_total: float) -> EvalCaseResult:
    return EvalCaseResult(
        sample_id=sample_id,
        question="Q",
        subset="synthetic_source_grounded",
        type="factoid",
        difficulty="easy",
        expected_behavior="answer",
        expected_chunk_ids=["chunk_1"],
        expected_document_ids=["doc_1"],
        expected_urls=["https://docs.stripe.com"],
        metrics={
            "retrieval.document_recall_at_10": retrieval,
            "retrieval.chunk_recall_at_10": retrieval,
            "context.context_chunk_recall": retrieval,
            "generation.parsed_successfully": 1.0,
            "citation.valid_citation_rate": 1.0,
            "citation.invented_source_rate": 0.0,
        },
        latency_ms={"retrieve": 10.0, "rerank": 10.0, "context": 10.0, "generation": 10.0, "total": latency_total},
        passed=True,
    )


@pytest.mark.unit
class TestEvaluationRegression:
    def test_get_nested_metric_reads_dotted_paths(self) -> None:
        payload = {"retrieval": {"chunk_recall_at_10_mean": 0.75}}
        assert get_nested_metric(payload, "retrieval.chunk_recall_at_10_mean") == 0.75

    def test_get_nested_metric_returns_none_for_missing(self) -> None:
        assert get_nested_metric({"retrieval": {}}, "retrieval.missing") is None

    def test_flatten_metrics_flattens_nested_payload(self) -> None:
        flattened = flatten_metrics({"retrieval": {"a": 1.0}, "latency": {"x": 10}})
        assert flattened["retrieval.a"] == 1.0
        assert flattened["latency.x"] == 10.0

    def test_compute_metric_deltas_candidate_minus_baseline(self) -> None:
        deltas = compute_metric_deltas(
            {"retrieval": {"a": 0.5}},
            {"retrieval": {"a": 0.8}},
        )
        assert deltas["retrieval.a"] == pytest.approx(0.3)

    def test_default_regression_gates_non_empty(self) -> None:
        gates = default_regression_gates()
        assert len(gates) > 0

    def test_apply_regression_gates_passes_within_tolerance(self) -> None:
        gates = [RegressionGate(metric_path="retrieval.score", max_drop=0.1, required=True)]
        results = apply_regression_gates(
            baseline_metrics={"retrieval": {"score": 0.8}},
            candidate_metrics={"retrieval": {"score": 0.75}},
            gates=gates,
        )
        assert results[0].passed

    def test_apply_regression_gates_fails_on_drop(self) -> None:
        gates = [RegressionGate(metric_path="retrieval.score", max_drop=0.05, required=True)]
        results = apply_regression_gates(
            baseline_metrics={"retrieval": {"score": 0.8}},
            candidate_metrics={"retrieval": {"score": 0.6}},
            gates=gates,
        )
        assert not results[0].passed

    def test_missing_required_metric_fails(self) -> None:
        gates = [RegressionGate(metric_path="retrieval.score", required=True)]
        results = apply_regression_gates(
            baseline_metrics={"retrieval": {}},
            candidate_metrics={"retrieval": {}},
            gates=gates,
        )
        assert not results[0].passed

    def test_missing_non_required_metric_passes_with_reason(self) -> None:
        gates = [RegressionGate(metric_path="retrieval.score", required=False)]
        results = apply_regression_gates(
            baseline_metrics={"retrieval": {}},
            candidate_metrics={"retrieval": {}},
            gates=gates,
        )
        assert results[0].passed
        assert "skipped" in results[0].reason.lower()

    def test_relative_latency_increase_gate(self) -> None:
        gates = [
            RegressionGate(
                metric_path="latency.total_p95_ms",
                max_relative_increase=0.25,
                required=True,
            )
        ]
        results = apply_regression_gates(
            baseline_metrics={"latency": {"total_p95_ms": 100.0}},
            candidate_metrics={"latency": {"total_p95_ms": 140.0}},
            gates=gates,
        )
        assert not results[0].passed

    def test_compare_eval_runs_from_temp_dirs(self, tmp_path: Path) -> None:
        baseline_dir = tmp_path / "runs"
        candidate_dir = tmp_path / "runs"
        write_eval_run_artifacts(
            run_dir=baseline_dir,
            run_id="baseline",
            results=[_result("b1", retrieval=0.8, latency_total=100.0)],
        )
        write_eval_run_artifacts(
            run_dir=candidate_dir,
            run_id="candidate",
            results=[_result("c1", retrieval=0.7, latency_total=130.0)],
        )
        comparison = compare_eval_runs(
            baseline_run_dir=baseline_dir / "baseline",
            candidate_run_dir=candidate_dir / "candidate",
        )
        assert comparison.baseline_run_id == "baseline"
        assert comparison.candidate_run_id == "candidate"
        assert len(comparison.gate_results) > 0

    def test_build_regression_report_markdown_contains_table(self, tmp_path: Path) -> None:
        write_eval_run_artifacts(
            run_dir=tmp_path,
            run_id="baseline",
            results=[_result("b1", retrieval=0.8, latency_total=100.0)],
        )
        write_eval_run_artifacts(
            run_dir=tmp_path,
            run_id="candidate",
            results=[_result("c1", retrieval=0.85, latency_total=110.0)],
        )
        comparison = compare_eval_runs(
            baseline_run_dir=tmp_path / "baseline",
            candidate_run_dir=tmp_path / "candidate",
        )
        markdown = build_regression_report_markdown(comparison)
        assert "| Metric | Baseline | Candidate | Delta | Passed | Reason |" in markdown

    def test_write_regression_report_writes_file(self, tmp_path: Path) -> None:
        write_eval_run_artifacts(
            run_dir=tmp_path,
            run_id="baseline",
            results=[_result("b1", retrieval=0.8, latency_total=100.0)],
        )
        write_eval_run_artifacts(
            run_dir=tmp_path,
            run_id="candidate",
            results=[_result("c1", retrieval=0.85, latency_total=110.0)],
        )
        comparison = compare_eval_runs(
            baseline_run_dir=tmp_path / "baseline",
            candidate_run_dir=tmp_path / "candidate",
        )
        report_path = write_regression_report(
            result=comparison,
            output_path=tmp_path / "regression.md",
        )
        assert report_path.exists()
