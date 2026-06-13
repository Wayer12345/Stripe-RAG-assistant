"""Unit tests for eval core runner with fake online layers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.evaluation.judges import HeuristicJudge
from app.evaluation.records import EvalDifficulty, EvalExpectedBehavior, EvalQueryType, EvalRunnerOptions, EvalSample, EvalSubset
from app.evaluation.runner import (
    run_context_case,
    run_eval_batch,
    run_full_case,
    run_generation_case,
    run_rerank_case,
    run_retrieval_case,
)


def _sample(*, sample_id: str = "eval_1", sample_type: EvalQueryType = EvalQueryType.FACTOID) -> EvalSample:
    return EvalSample(
        id=sample_id,
        question="How does Stripe documentation explain webhooks?",
        subset=EvalSubset.SYNTHETIC_SOURCE_GROUNDED,
        type=sample_type,
        difficulty=EvalDifficulty.EASY,
        expected_behavior=EvalExpectedBehavior.ANSWER,
        expected_chunk_ids=["chunk_1"],
        expected_document_ids=["doc_1"],
        expected_urls=["https://docs.stripe.com/webhooks"],
        reference_answer="Stripe docs explain how to configure webhook endpoints.",
        metadata={},
    )


def _build_fake_layers(
    *,
    fail_retrieve: bool = False,
    no_answer: bool = False,
) -> tuple[type, type, type, type, dict[str, int]]:
    calls = {"retrieve": 0, "rerank": 0, "context": 0, "generation": 0}

    source = SimpleNamespace(
        title="Webhooks",
        url="https://docs.stripe.com/webhooks",
        chunk_id="chunk_1",
        document_id="doc_1",
    )
    retrieval_item = SimpleNamespace(
        chunk_id="chunk_1",
        document_id="doc_1",
        source=source,
        final_score=0.9,
        text="Stripe webhooks let you receive event notifications.",
    )

    class FakeRetrieveLayer:
        def __init__(self, *, question: str, config_path: str, top_k: int | None, write_trace: bool) -> None:
            self.question = question

        def run(self) -> SimpleNamespace:
            calls["retrieve"] += 1
            if fail_retrieve:
                raise RuntimeError("retrieve failed")
            return SimpleNamespace(
                results=[retrieval_item],
                results_total=1,
                top_k=5,
                strategy="dense",
                duration_ms=11,
                trace_path="trace_retrieve.json",
            )

    class FakeRerankLayer:
        def __init__(
            self,
            *,
            question: str,
            candidates: list[SimpleNamespace],
            config_path: str,
            top_k_before: int | None,
            top_k_after: int | None,
            write_trace: bool,
        ) -> None:
            self.candidates = candidates

        def run(self) -> SimpleNamespace:
            calls["rerank"] += 1
            return SimpleNamespace(
                results=self.candidates,
                input_results_total=len(self.candidates),
                reranked_results_total=len(self.candidates),
                model_name="fake-reranker",
                top_k_before=5,
                top_k_after=3,
                latency_budget_exceeded=False,
                cache_hits=1,
                cache_misses=0,
                duration_ms=7,
                trace_path="trace_rerank.json",
            )

    class FakeBuildContextLayer:
        def __init__(
            self,
            *,
            question: str,
            results: list[SimpleNamespace],
            config_path: str,
            token_budget: int | None,
            max_chunks: int | None,
            write_trace: bool,
        ) -> None:
            self.results = results

        def run(self) -> SimpleNamespace:
            calls["context"] += 1
            context_bundle = SimpleNamespace(
                chunks=self.results,
                token_count=80,
                token_budget=120,
                truncated=False,
                sources=[source],
                rendered_context="Stripe webhooks let you receive event notifications.",
            )
            return SimpleNamespace(
                context_bundle=context_bundle,
                included_chunks_total=1,
                dropped_chunks_total=0,
                token_budget=120,
                token_count=80,
                sources_total=1,
                duration_ms=5,
                trace_path="trace_context.json",
            )

    class FakeGenerateAnswerLayer:
        def __init__(
            self,
            *,
            question: str,
            context_bundle: SimpleNamespace,
            config_path: str,
            write_trace: bool,
        ) -> None:
            self.context_bundle = context_bundle

        def run(self) -> SimpleNamespace:
            calls["generation"] += 1
            generated_answer = SimpleNamespace(
                answer=(
                    "I don't have enough information in the indexed Stripe Guides sources to answer this reliably."
                    if no_answer
                    else "Stripe docs explain configuring webhook endpoints and events."
                ),
                confidence="none" if no_answer else "high",
                sources=[] if no_answer else [source],
                parsed_successfully=True,
            )
            return SimpleNamespace(
                generated_answer=generated_answer,
                confidence=generated_answer.confidence,
                sources_total=len(generated_answer.sources),
                parsed_successfully=True,
                provider="ollama",
                model_name="fake-model",
                duration_ms=9,
                trace_path="trace_generation.json",
            )

    return FakeRetrieveLayer, FakeRerankLayer, FakeBuildContextLayer, FakeGenerateAnswerLayer, calls


@pytest.mark.unit
class TestEvalRunner:
    def test_run_retrieval_case_calls_only_retrieve(self) -> None:
        retrieve_cls, rerank_cls, context_cls, generation_cls, calls = _build_fake_layers()
        result = run_retrieval_case(_sample(), retrieve_layer_cls=retrieve_cls)
        assert calls["retrieve"] == 1
        assert calls["rerank"] == 0
        assert result.retrieval is not None

    def test_run_rerank_case_calls_retrieve_then_rerank(self) -> None:
        retrieve_cls, rerank_cls, context_cls, generation_cls, calls = _build_fake_layers()
        result = run_rerank_case(
            _sample(),
            retrieve_layer_cls=retrieve_cls,
            rerank_layer_cls=rerank_cls,
        )
        assert calls["retrieve"] == 1
        assert calls["rerank"] == 1
        assert calls["context"] == 0
        assert result.rerank is not None

    def test_run_context_case_calls_first_three_layers(self) -> None:
        retrieve_cls, rerank_cls, context_cls, generation_cls, calls = _build_fake_layers()
        result = run_context_case(
            _sample(),
            retrieve_layer_cls=retrieve_cls,
            rerank_layer_cls=rerank_cls,
            build_context_layer_cls=context_cls,
        )
        assert calls["retrieve"] == 1
        assert calls["rerank"] == 1
        assert calls["context"] == 1
        assert calls["generation"] == 0
        assert result.context is not None

    def test_run_generation_case_calls_all_layers(self) -> None:
        retrieve_cls, rerank_cls, context_cls, generation_cls, calls = _build_fake_layers()
        result = run_generation_case(
            _sample(),
            retrieve_layer_cls=retrieve_cls,
            rerank_layer_cls=rerank_cls,
            build_context_layer_cls=context_cls,
            generate_answer_layer_cls=generation_cls,
        )
        assert calls == {"retrieve": 1, "rerank": 1, "context": 1, "generation": 1}
        assert result.generation is not None

    def test_run_full_case_returns_eval_case_result_with_metadata(self) -> None:
        retrieve_cls, rerank_cls, context_cls, generation_cls, calls = _build_fake_layers()
        sample = _sample(sample_id="sample_42")
        result = run_full_case(
            sample,
            retrieve_layer_cls=retrieve_cls,
            rerank_layer_cls=rerank_cls,
            build_context_layer_cls=context_cls,
            generate_answer_layer_cls=generation_cls,
        )
        assert result.sample_id == "sample_42"
        assert result.question == sample.question
        assert result.subset == sample.subset.value
        assert result.type == sample.type.value
        assert result.retrieval is not None
        assert result.rerank is not None
        assert result.context is not None
        assert result.generation is not None
        assert result.citation is not None

    def test_metrics_are_populated(self) -> None:
        retrieve_cls, rerank_cls, context_cls, generation_cls, calls = _build_fake_layers()
        result = run_full_case(
            _sample(sample_type=EvalQueryType.OOD),
            retrieve_layer_cls=retrieve_cls,
            rerank_layer_cls=rerank_cls,
            build_context_layer_cls=context_cls,
            generate_answer_layer_cls=generation_cls,
        )
        assert "retrieval.chunk_recall_at_10" in result.metrics
        assert "rerank.mrr_delta" in result.metrics
        assert "context.context_chunk_recall" in result.metrics
        assert "generation.parsed_successfully" in result.metrics
        assert "citation.valid_citation_rate" in result.metrics
        assert "confidence.abstained" in result.metrics
        assert "robustness.ood_abstained" in result.metrics

    def test_trace_paths_and_latency_maps_are_populated(self) -> None:
        retrieve_cls, rerank_cls, context_cls, generation_cls, calls = _build_fake_layers()
        result = run_full_case(
            _sample(),
            retrieve_layer_cls=retrieve_cls,
            rerank_layer_cls=rerank_cls,
            build_context_layer_cls=context_cls,
            generate_answer_layer_cls=generation_cls,
        )
        assert result.trace_paths["retrieve"] == "trace_retrieve.json"
        assert result.trace_paths["generation"] == "trace_generation.json"
        assert result.latency_ms["retrieve"] > 0
        assert result.latency_ms["total"] > 0

    def test_batch_runner_preserves_order(self) -> None:
        retrieve_cls, rerank_cls, context_cls, generation_cls, calls = _build_fake_layers()
        samples = [_sample(sample_id="s1"), _sample(sample_id="s2")]
        results, errors = run_eval_batch(
            samples,
            mode="full",
            retrieve_layer_cls=retrieve_cls,
            rerank_layer_cls=rerank_cls,
            build_context_layer_cls=context_cls,
            generate_answer_layer_cls=generation_cls,
        )
        assert [result.sample_id for result in results] == ["s1", "s2"]
        assert errors == []

    def test_batch_collects_errors_when_not_fail_fast(self) -> None:
        retrieve_cls, rerank_cls, context_cls, generation_cls, calls = _build_fake_layers(
            fail_retrieve=True
        )
        options = EvalRunnerOptions(fail_fast=False)
        results, errors = run_eval_batch(
            [_sample(sample_id="s1"), _sample(sample_id="s2")],
            mode="full",
            options=options,
            retrieve_layer_cls=retrieve_cls,
            rerank_layer_cls=rerank_cls,
            build_context_layer_cls=context_cls,
            generate_answer_layer_cls=generation_cls,
        )
        assert len(results) == 2
        assert len(errors) == 2
        assert results[0].error is not None

    def test_batch_raises_on_fail_fast(self) -> None:
        retrieve_cls, rerank_cls, context_cls, generation_cls, calls = _build_fake_layers(
            fail_retrieve=True
        )
        options = EvalRunnerOptions(fail_fast=True)
        with pytest.raises(Exception):
            run_eval_batch(
                [_sample(sample_id="s1")],
                mode="full",
                options=options,
                retrieve_layer_cls=retrieve_cls,
                rerank_layer_cls=rerank_cls,
                build_context_layer_cls=context_cls,
                generate_answer_layer_cls=generation_cls,
            )

    def test_mode_validation_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError):
            run_eval_batch([_sample()], mode="unknown")

    def test_mode_retrieval_stops_early(self) -> None:
        retrieve_cls, rerank_cls, context_cls, generation_cls, calls = _build_fake_layers()
        run_eval_batch([_sample()], mode="retrieval", retrieve_layer_cls=retrieve_cls)
        assert calls["retrieve"] == 1
        assert calls["rerank"] == 0
        assert calls["context"] == 0
        assert calls["generation"] == 0

    def test_mode_context_stops_after_context(self) -> None:
        retrieve_cls, rerank_cls, context_cls, generation_cls, calls = _build_fake_layers()
        run_eval_batch(
            [_sample()],
            mode="context",
            retrieve_layer_cls=retrieve_cls,
            rerank_layer_cls=rerank_cls,
            build_context_layer_cls=context_cls,
        )
        assert calls["retrieve"] == 1
        assert calls["rerank"] == 1
        assert calls["context"] == 1
        assert calls["generation"] == 0

    def test_no_answer_result_is_supported(self) -> None:
        retrieve_cls, rerank_cls, context_cls, generation_cls, calls = _build_fake_layers(
            no_answer=True
        )
        result = run_generation_case(
            _sample(sample_type=EvalQueryType.UNANSWERABLE),
            retrieve_layer_cls=retrieve_cls,
            rerank_layer_cls=rerank_cls,
            build_context_layer_cls=context_cls,
            generate_answer_layer_cls=generation_cls,
        )
        assert result.generation is not None
        assert result.generation.confidence == "none"
        assert result.metrics["generation.no_answer"] == 1.0

    def test_judge_integration_adds_judge_metrics(self) -> None:
        retrieve_cls, rerank_cls, context_cls, generation_cls, calls = _build_fake_layers()
        options = EvalRunnerOptions(judge_enabled=True, judge_backend="heuristic")
        result = run_full_case(
            _sample(),
            options=options,
            retrieve_layer_cls=retrieve_cls,
            rerank_layer_cls=rerank_cls,
            build_context_layer_cls=context_cls,
            generate_answer_layer_cls=generation_cls,
            judge=HeuristicJudge(),
        )
        assert result.judge is not None
        assert "judge.groundedness_score" in result.metrics
