"""Unit tests for eval judge adapters."""

from __future__ import annotations

import pytest

from app.evaluation.judges import HeuristicJudge, LocalLLMJudge, create_judge
from app.evaluation.records import JudgeRecord


@pytest.mark.unit
class TestHeuristicJudge:
    def test_returns_judge_record(self) -> None:
        judge = HeuristicJudge()
        result = judge.judge(
            sample_id="sample_1",
            question="How to enable webhooks?",
            answer="Enable webhooks from dashboard settings.",
            context_texts=["Dashboard settings include webhook configuration and endpoint setup."],
            sources=[{"chunk_id": "chunk_1"}],
            reference_answer=None,
        )
        assert isinstance(result, JudgeRecord)
        assert result.judge_backend == "heuristic"

    def test_grounded_answer_scores_higher_than_unsupported(self) -> None:
        judge = HeuristicJudge()
        grounded = judge.judge(
            sample_id="s1",
            question="How to enable webhooks?",
            answer="Use dashboard settings to configure webhook endpoints.",
            context_texts=["Configure webhook endpoints in dashboard settings."],
            sources=[{"chunk_id": "chunk_1"}],
        )
        unsupported = judge.judge(
            sample_id="s2",
            question="How to enable webhooks?",
            answer="Stripe guarantees zero disputes for all businesses.",
            context_texts=["Configure webhook endpoints in dashboard settings."],
            sources=[{"chunk_id": "chunk_1"}],
        )
        assert grounded.groundedness_score > unsupported.groundedness_score

    def test_relevant_answer_scores_higher_relevance(self) -> None:
        judge = HeuristicJudge()
        relevant = judge.judge(
            sample_id="s1",
            question="How do I configure 3D Secure?",
            answer="Configure 3D Secure in payment settings.",
            context_texts=["Payment settings include 3D Secure configuration details."],
            sources=[{"chunk_id": "chunk_1"}],
        )
        irrelevant = judge.judge(
            sample_id="s2",
            question="How do I configure 3D Secure?",
            answer="Cats are mammals and have whiskers.",
            context_texts=["Payment settings include 3D Secure configuration details."],
            sources=[{"chunk_id": "chunk_1"}],
        )
        assert relevant.relevance_score > irrelevant.relevance_score

    def test_non_empty_answer_with_no_sources_has_low_source_support(self) -> None:
        judge = HeuristicJudge()
        result = judge.judge(
            sample_id="s1",
            question="What is Radar?",
            answer="Radar is a fraud prevention product.",
            context_texts=["Radar helps fraud prevention."],
            sources=[],
        )
        assert result.source_support_score == 0.0

    def test_abstain_no_answer_returns_safe_verdict(self) -> None:
        judge = HeuristicJudge()
        result = judge.judge(
            sample_id="s1",
            question="What is unknown product?",
            answer="I don't have enough information in the indexed Stripe Guides sources to answer this reliably.",
            context_texts=[],
            sources=[],
        )
        assert result.verdict == "abstain"
        assert result.hallucination_risk == 0.0

    def test_reference_answer_improves_completeness(self) -> None:
        judge = HeuristicJudge()
        with_reference = judge.judge(
            sample_id="s1",
            question="What is Stripe Radar?",
            answer="Radar blocks suspicious payments.",
            context_texts=["Radar blocks suspicious payments and flags risky attempts."],
            sources=[{"chunk_id": "chunk_1"}],
            reference_answer="Radar blocks suspicious payments and flags risky attempts.",
        )
        without_reference = judge.judge(
            sample_id="s2",
            question="What is Stripe Radar?",
            answer="Radar blocks suspicious payments.",
            context_texts=["Radar blocks suspicious payments and flags risky attempts."],
            sources=[{"chunk_id": "chunk_1"}],
            reference_answer=None,
        )
        assert with_reference.completeness_score >= without_reference.completeness_score


@pytest.mark.unit
class TestJudgeFactory:
    def test_create_heuristic(self) -> None:
        assert isinstance(create_judge(backend="heuristic"), HeuristicJudge)

    def test_create_none(self) -> None:
        assert create_judge(backend="none") is None

    def test_unsupported_backend_raises(self) -> None:
        with pytest.raises(ValueError):
            create_judge(backend="unknown")

    def test_local_llm_judge_raises_clear_not_implemented(self) -> None:
        judge = LocalLLMJudge()
        with pytest.raises(NotImplementedError, match="later task"):
            judge.judge(
                sample_id="s1",
                question="Q",
                answer="A",
                context_texts=[],
                sources=[],
            )
