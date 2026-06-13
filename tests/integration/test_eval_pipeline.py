"""Integration pipeline smoke test for eval artifacts with fakes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.application_layers.eval.run_generation_eval import RunGenerationEvalLayer
from app.evaluation.records import EvalCaseResult


def _fake_case(sample_id: str) -> EvalCaseResult:
    return EvalCaseResult(
        sample_id=sample_id,
        question="Q",
        subset="synthetic_source_grounded",
        type="factoid",
        difficulty="easy",
        expected_behavior="answer",
        expected_chunk_ids=["chunk_1"],
        expected_document_ids=["doc_1"],
        expected_urls=["https://docs.stripe.com/webhooks"],
        metrics={"retrieval.chunk_recall_at_10": 1.0, "latency.total_ms": 12.0},
        latency_ms={"total": 12.0},
        passed=True,
    )


@pytest.mark.integration
def test_eval_pipeline_end_to_end_with_fakes(tmp_path: Path) -> None:
    dataset = Path("tests/fixtures/sample_eval.jsonl")
    assert dataset.exists()

    def fake_runner(samples: list[Any], *, mode: str, options: Any, **_: Any) -> tuple[list[EvalCaseResult], list[dict[str, Any]]]:
        _ = (mode, options)
        return [_fake_case(sample.id) for sample in samples], []

    first = RunGenerationEvalLayer(
        dataset_path=dataset,
        runs_dir=tmp_path,
        run_id="candidate",
        runner_fn=fake_runner,
    ).run()
    second = RunGenerationEvalLayer(
        dataset_path=dataset,
        runs_dir=tmp_path,
        run_id="baseline",
        runner_fn=fake_runner,
    ).run()

    assert first.report_path.exists()
    assert second.summary_path.exists()
