"""Unit tests for deterministic evaluation metric modules."""

from __future__ import annotations

import pytest
from app.evaluation.citation_metrics import (
    answer_without_sources_flag,
    citation_precision,
    citation_recall,
    high_confidence_invalid_source_flag,
    invented_source_rate,
    valid_citation_rate,
)
from app.evaluation.confidence_metrics import (
    abstention_on_answerable_rate,
    abstention_rate,
    answer_on_unanswerable_rate,
    confidence_distribution,
    high_confidence_without_sources_rate,
)
from app.evaluation.context_metrics import (
    context_chunk_recall,
    context_document_recall,
    dedup_rate,
    empty_context_rate,
    expected_source_dropped_rate,
    token_budget_violation,
)
from app.evaluation.generation_metrics import (
    empty_answer_flag,
    no_answer_flag,
    reference_completeness,
    reference_token_f1,
    valid_generation_output_flag,
)
from app.evaluation.latency_metrics import (
    latency_budget_violation_rate,
    latency_summary,
    mean_latency,
    percentile_latency,
    stage_latency_summary,
)
from app.evaluation.rerank_metrics import (
    cache_hit_rate,
    expected_source_kept_rate,
    first_relevant_rank,
    mrr_delta,
    rank_delta,
)
from app.evaluation.retrieval_metrics import (
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    url_recall_at_k,
)
from app.evaluation.robustness_metrics import (
    adversarial_valid_source_rate,
    ambiguous_abstention_or_low_confidence_rate,
    ood_abstention_rate,
    typo_answer_rate,
)


@pytest.mark.unit
class TestRetrievalMetrics:
    def test_hit_at_k_hit(self) -> None:
        assert hit_at_k(["a", "b", "c"], ["b"], 2) == 1.0

    def test_hit_at_k_miss(self) -> None:
        assert hit_at_k(["a", "b"], ["x"], 2) == 0.0

    def test_recall_at_k_multiple_expected(self) -> None:
        assert recall_at_k(["a", "b", "c"], ["b", "c", "d"], 3) == pytest.approx(2 / 3)

    def test_precision_at_k_handles_duplicates(self) -> None:
        assert precision_at_k(["a", "a", "b"], ["a"], 3) == pytest.approx(0.5)

    def test_mrr_at_k_first_relevant(self) -> None:
        assert mrr_at_k(["x", "y", "z"], ["y"]) == 0.5

    def test_ndcg_perfect_order(self) -> None:
        assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == 1.0

    def test_url_recall_normalizes_trailing_slash(self) -> None:
        assert (
            url_recall_at_k(
                ["https://Docs.Stripe.com/payments/"],
                ["https://docs.stripe.com/payments"],
                1,
            )
            == 1.0
        )

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            hit_at_k(["a"], ["a"], 0)


@pytest.mark.unit
class TestRerankMetrics:
    def test_first_relevant_rank(self) -> None:
        assert first_relevant_rank(["x", "a", "b"], ["b"]) == 3

    def test_rank_delta_positive_for_improvement(self) -> None:
        assert rank_delta(["x", "y", "a"], ["a", "x", "y"], ["a"]) == 2

    def test_rank_delta_negative_for_worse(self) -> None:
        assert rank_delta(["a", "x"], ["x", "a"], ["a"]) == -1

    def test_mrr_delta(self) -> None:
        assert mrr_delta(["x", "a"], ["a", "x"], ["a"]) == pytest.approx(0.5)

    def test_expected_source_kept_rate(self) -> None:
        assert expected_source_kept_rate(["a", "c"], ["a", "b"]) == 0.5

    def test_cache_hit_rate_zero_denominator(self) -> None:
        assert cache_hit_rate(0, 0) == 0.0


@pytest.mark.unit
class TestContextMetrics:
    def test_context_chunk_recall(self) -> None:
        assert context_chunk_recall(["a", "b"], ["b", "c"]) == 0.5

    def test_context_document_recall(self) -> None:
        assert context_document_recall(["doc_1"], ["doc_1", "doc_2"]) == 0.5

    def test_expected_source_dropped_rate_detects_drop(self) -> None:
        assert expected_source_dropped_rate(["a", "b"], ["b"], ["a", "b"]) == 0.5

    def test_token_budget_violation(self) -> None:
        assert token_budget_violation(101, 100) == 1.0
        assert token_budget_violation(100, 100) == 0.0

    def test_empty_context_rate(self) -> None:
        assert empty_context_rate([0, 10, 0]) == pytest.approx(2 / 3)

    def test_dedup_rate(self) -> None:
        assert dedup_rate(["a", "a", "b", "c"], ["a", "b", "c"]) == 0.25


@pytest.mark.unit
class TestGenerationMetrics:
    def test_empty_answer_flag(self) -> None:
        assert empty_answer_flag("   ") == 1.0

    def test_no_answer_flag_for_none_confidence(self) -> None:
        assert no_answer_flag("none", "non-empty") == 1.0

    def test_reference_token_f1(self) -> None:
        score = reference_token_f1("stripe handles disputes", "stripe disputes")
        assert 0.0 < score <= 1.0

    def test_reference_completeness(self) -> None:
        assert reference_completeness("a b", "a b c d") == 0.5

    def test_valid_generation_output_rejects_invalid(self) -> None:
        assert (
            valid_generation_output_flag(
                answer="",
                confidence="high",
                parsed_successfully=True,
            )
            == 0.0
        )
        assert (
            valid_generation_output_flag(
                answer="ok",
                confidence="invalid",
                parsed_successfully=True,
            )
            == 0.0
        )


@pytest.mark.unit
class TestCitationMetrics:
    def test_valid_citation_rate(self) -> None:
        assert valid_citation_rate(["a", "b"], ["a", "x"]) == 0.5

    def test_invented_source_rate(self) -> None:
        assert invented_source_rate(["a", "z"], ["a", "b"]) == 0.5

    def test_citation_precision(self) -> None:
        assert citation_precision(["a", "b"], ["a", "x"]) == 0.5

    def test_citation_recall(self) -> None:
        assert citation_recall(["a"], ["a", "b"]) == 0.5

    def test_high_confidence_invalid_source_flag(self) -> None:
        assert high_confidence_invalid_source_flag("high", ["z"], ["a"]) == 1.0

    def test_answer_without_sources_flag(self) -> None:
        assert answer_without_sources_flag("There is an answer", []) == 1.0


@pytest.mark.unit
class TestConfidenceMetrics:
    def test_confidence_distribution(self) -> None:
        dist = confidence_distribution(["high", "medium", "none", "unknown"])
        assert dist["high"] == 0.25
        assert dist["low"] == 0.25

    def test_abstention_rate(self) -> None:
        assert abstention_rate(["none", "high", "none"]) == pytest.approx(2 / 3)

    def test_high_confidence_without_sources_rate(self) -> None:
        assert high_confidence_without_sources_rate(["high", "low"], [0, 0]) == 0.5

    def test_abstention_on_answerable_rate(self) -> None:
        assert abstention_on_answerable_rate(["none", "high"], ["answer", "answer"]) == 0.5

    def test_answer_on_unanswerable_rate(self) -> None:
        assert answer_on_unanswerable_rate(["high", "none"], ["abstain", "abstain"]) == 0.5


@pytest.mark.unit
class TestRobustnessMetrics:
    def test_ood_abstention_rate(self) -> None:
        assert ood_abstention_rate(["none", "high"], ["ood", "ood"]) == 0.5

    def test_typo_answer_rate(self) -> None:
        assert typo_answer_rate(["high", "none"], ["typo", "typo"]) == 0.5

    def test_ambiguous_safe_rate(self) -> None:
        assert (
            ambiguous_abstention_or_low_confidence_rate(
                ["low", "high"],
                ["ambiguous", "ambiguous"],
            )
            == 0.5
        )

    def test_adversarial_valid_source_rate(self) -> None:
        assert adversarial_valid_source_rate(["adversarial", "adversarial"], [True, False]) == 0.5


@pytest.mark.unit
class TestLatencyMetrics:
    def test_mean_latency(self) -> None:
        assert mean_latency([10, 20, 30]) == 20.0

    def test_percentiles_deterministic(self) -> None:
        values = [10, 20, 30, 40, 50]
        assert percentile_latency(values, 50) == 30.0
        assert percentile_latency(values, 95) == pytest.approx(48.0)
        assert percentile_latency(values, 99) == pytest.approx(49.6)

    def test_empty_latency_summary_returns_zeros(self) -> None:
        summary = latency_summary([], "retrieval")
        assert summary["retrieval_mean_ms"] == 0.0
        assert summary["retrieval_p95_ms"] == 0.0

    def test_negative_latency_raises(self) -> None:
        with pytest.raises(ValueError):
            mean_latency([-1])

    def test_budget_violation_rate(self) -> None:
        assert latency_budget_violation_rate([90, 110, 120], 100) == pytest.approx(2 / 3)

    def test_stage_latency_summary_includes_all_stages(self) -> None:
        summary = stage_latency_summary(
            retrieve_ms=[10],
            rerank_ms=[20],
            context_ms=[30],
            generation_ms=[40],
            total_ms=[100],
        )
        assert "retrieval_mean_ms" in summary
        assert "rerank_mean_ms" in summary
        assert "context_mean_ms" in summary
        assert "generation_mean_ms" in summary
        assert "total_mean_ms" in summary
