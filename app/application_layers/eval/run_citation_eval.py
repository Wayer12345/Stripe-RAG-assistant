"""Application-layer wrapper for citation eval mode."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.evaluation.records import EvalCaseResult
from app.evaluation.utils import (
    EvalRunExecutionResult,
    build_base_eval_arg_parser,
    load_metrics_payload,
    log_eval_run_summary,
    log_metrics_summary,
    parse_cli_set,
    run_eval_suite,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class RunCitationEvalLayer:
    def __init__(
        self,
        *,
        dataset_path: Path | str,
        run_dir: Path | str | None = None,
        runs_dir: Path | str = Path("data/eval/runs"),
        run_id: str | None = None,
        config_path: Path | str = Path("configs/config.yaml"),
        limit: int | None = None,
        seed: int | None = None,
        subsets: set[str] | None = None,
        types: set[str] | None = None,
        difficulties: set[str] | None = None,
        expected_behaviors: set[str] | None = None,
        retrieve_top_k: int | None = None,
        rerank_top_k_before: int | None = None,
        rerank_top_k_after: int | None = None,
        context_token_budget: int | None = None,
        context_max_chunks: int | None = None,
        write_trace: bool | None = None,
        fail_fast: bool = False,
        judge_enabled: bool = False,
        judge_backend: str = "heuristic",
        runner_fn: Callable[..., tuple[list[EvalCaseResult], list[dict[str, Any]]]] | None = None,
    ) -> None:
        self._dataset_path = dataset_path
        self._run_dir = run_dir
        self._runs_dir = runs_dir
        self._run_id = run_id
        self._config_path = config_path
        self._limit = limit
        self._seed = seed
        self._subsets = subsets
        self._types = types
        self._difficulties = difficulties
        self._expected_behaviors = expected_behaviors
        self._retrieve_top_k = retrieve_top_k
        self._rerank_top_k_before = rerank_top_k_before
        self._rerank_top_k_after = rerank_top_k_after
        self._context_token_budget = context_token_budget
        self._context_max_chunks = context_max_chunks
        self._write_trace = write_trace
        self._fail_fast = fail_fast
        self._judge_enabled = judge_enabled
        self._judge_backend = judge_backend
        self._runner_fn = runner_fn

    def run(self) -> EvalRunExecutionResult:
        result = run_eval_suite(
            dataset_path=self._dataset_path,
            suite="citation",
            run_dir=self._run_dir,
            runs_dir=self._runs_dir,
            run_id=self._run_id,
            config_path=self._config_path,
            limit=self._limit,
            seed=self._seed,
            subsets=self._subsets,
            types=self._types,
            difficulties=self._difficulties,
            expected_behaviors=self._expected_behaviors,
            retrieve_top_k=self._retrieve_top_k,
            rerank_top_k_before=self._rerank_top_k_before,
            rerank_top_k_after=self._rerank_top_k_after,
            context_token_budget=self._context_token_budget,
            context_max_chunks=self._context_max_chunks,
            write_trace=self._write_trace,
            fail_fast=self._fail_fast,
            judge_enabled=self._judge_enabled,
            judge_backend=self._judge_backend,
            runner_fn=self._runner_fn,
        )
        metrics = load_metrics_payload(result.metrics_path)
        log_metrics_summary(
            logger_name=logger,
            title="Citation eval metrics summary:",
            metrics=metrics,
            metric_paths={
                "valid_citation_rate": "citation.valid_citation_rate_mean",
                "invented_source_rate": "citation.invented_source_rate_mean",
                "citation_precision": "citation.citation_precision_mean",
                "citation_recall": "citation.citation_recall_mean",
                "answer_without_sources": "citation.answer_without_sources_mean",
                "high_confidence_invalid_source": "citation.high_confidence_invalid_source_mean",
                "unsupported_citation_rate": "citation.unsupported_citation_rate_mean",
                "generation_valid_output": "generation.valid_generation_output_mean",
                "confidence_high": "confidence.confidence_high_mean",
                "abstained": "confidence.abstained_mean",
                "total_p95_ms": "latency.total_p95_ms",
            },
        )
        return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = build_base_eval_arg_parser("Run citation eval on a dataset.")
    parser.add_argument("--rerank-top-k-before", type=int, default=None)
    parser.add_argument("--rerank-top-k-after", type=int, default=None)
    parser.add_argument("--context-token-budget", type=int, default=None)
    parser.add_argument("--context-max-chunks", type=int, default=None)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = RunCitationEvalLayer(
        dataset_path=args.dataset,
        run_dir=args.run_dir,
        runs_dir=args.runs_dir,
        run_id=args.run_id,
        config_path=args.config,
        limit=args.limit,
        seed=args.seed,
        subsets=parse_cli_set(args.subset),
        types=parse_cli_set(args.types),
        difficulties=parse_cli_set(args.difficulty),
        expected_behaviors=parse_cli_set(args.expected_behavior),
        retrieve_top_k=args.retrieve_top_k,
        rerank_top_k_before=args.rerank_top_k_before,
        rerank_top_k_after=args.rerank_top_k_after,
        context_token_budget=args.context_token_budget,
        context_max_chunks=args.context_max_chunks,
        write_trace=not args.no_trace,
        fail_fast=args.fail_fast,
        judge_enabled=args.judge,
        judge_backend=args.judge_backend,
    ).run()
    log_eval_run_summary(result, logger)


if __name__ == "__main__":
    main()
