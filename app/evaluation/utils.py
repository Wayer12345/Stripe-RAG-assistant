"""Shared utilities for eval orchestration and reporting."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.evaluation.datasets import filter_eval_samples, load_eval_samples
from app.evaluation.records import EvalCaseResult, EvalRunnerOptions
from app.evaluation.reports import write_eval_run_artifacts
from app.evaluation.runner import run_eval_batch
from app.infrastructure.storage.manifest_store import read_manifest
from app.utils.config import load_settings, resolve_config_dir_and_path, to_optional_path
from app.utils.ids import make_run_id
from app.utils.logging import get_logger, setup_logging
from app.utils.timing import finish_timed_run, start_timed_run

logger = get_logger(__name__)


def parse_cli_set(values: list[str] | None) -> set[str] | None:
    """Normalize repeated CLI flags into a set."""
    if not values:
        return None
    parsed = {item.strip() for item in values if item.strip()}
    return parsed or None


def dataset_id_from_path(dataset_path: Path) -> str:
    """Derive dataset id from a dataset artifact path."""
    if dataset_path.name == "dataset.jsonl":
        return dataset_path.parent.name
    return dataset_path.stem


def count_values(values: list[str]) -> dict[str, int]:
    """Count occurrences of values for compact logging."""
    return dict(Counter(values))


def load_metrics_payload(metrics_path: Path) -> dict[str, Any]:
    """Read metrics artifact payload as a dictionary."""
    payload = read_manifest(metrics_path)
    return payload if isinstance(payload, dict) else {}


def _extract_metric(metrics: Mapping[str, Any], dotted_path: str) -> float | int | None:
    current: Any = metrics
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if isinstance(current, (int, float)) and not isinstance(current, bool):
        return current
    return None


def _get_first_metric(
    metrics: Mapping[str, Any], candidates: Sequence[str] | str
) -> float | int | None:
    paths = [candidates] if isinstance(candidates, str) else list(candidates)
    for path in paths:
        value = _extract_metric(metrics, path)
        if value is not None:
            return value
    return None


def _format_metric(value: float | int | None) -> str:
    if value is None:
        return "None"
    return f"{float(value):.4f}"


def log_metrics_summary(
    *,
    logger_name: Any,
    title: str,
    metrics: Mapping[str, Any],
    metric_paths: Mapping[str, Sequence[str] | str],
) -> None:
    """Log a compact one-line metrics summary from metric path mappings."""
    parts: list[str] = []
    found_any = False
    for metric_name, path_candidates in metric_paths.items():
        value = _get_first_metric(metrics, path_candidates)
        if value is not None:
            found_any = True
        parts.append(f"{metric_name}={_format_metric(value)}")
    if not found_any:
        logger_name.warning("%s missing metric group payload", title)
    logger_name.info("%s %s", title, " ".join(parts))


SUITE_TO_MODE = {
    "retrieval": "retrieval",
    "rerank": "rerank",
    "context": "context",
    "generation": "generation",
    "citation": "citation",
    "robustness": "robustness",
    "full": "full",
}


@dataclass(frozen=True)
class EvalRunExecutionResult:
    """Result summary for one eval suite execution."""

    run_id: str
    mode: str
    dataset_path: Path
    run_dir: Path
    cases_total: int
    cases_successful: int
    cases_failed: int
    failure_rate: float
    metrics_path: Path
    summary_path: Path
    report_path: Path
    duration_ms: int


def _resolve_run_target(
    *,
    run_dir: Path | None,
    runs_dir: Path,
    run_id: str | None,
    mode: str,
) -> tuple[Path, Path, str]:
    if run_dir is not None:
        final_run_dir = run_dir
        resolved_run_id = run_id or final_run_dir.name
        if run_id is not None and run_id != final_run_dir.name:
            raise ValueError("--run-id must match --run-dir name when both are provided.")
        return final_run_dir.parent, final_run_dir, resolved_run_id

    resolved_run_id = run_id or make_run_id(f"{mode}_eval")
    final_run_dir = runs_dir / resolved_run_id
    return runs_dir, final_run_dir, resolved_run_id


def run_eval_suite(
    *,
    dataset_path: Path | str,
    suite: str,
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
) -> EvalRunExecutionResult:
    """Run one eval suite and write standard run artifacts."""
    suite_normalized = suite.strip().lower()
    if suite_normalized not in SUITE_TO_MODE:
        raise ValueError(f"Unsupported suite: {suite!r}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be > 0 when provided.")

    resolved_dataset_path = Path(dataset_path)
    resolved_run_dir = to_optional_path(run_dir)
    resolved_runs_dir = Path(runs_dir)
    resolved_config_path = Path(config_path)
    mode = SUITE_TO_MODE[suite_normalized]

    config_dir, _ = resolve_config_dir_and_path(resolved_config_path)
    settings = load_settings(config_dir)
    setup_logging(settings)

    resolved_write_trace = write_trace if write_trace is not None else settings.eval.write_trace
    options = EvalRunnerOptions(
        config_path=str(resolved_config_path),
        retrieve_top_k=retrieve_top_k,
        rerank_top_k_before=rerank_top_k_before,
        rerank_top_k_after=rerank_top_k_after,
        context_token_budget=context_token_budget,
        context_max_chunks=context_max_chunks,
        write_trace=resolved_write_trace,
        fail_fast=fail_fast,
        judge_enabled=judge_enabled,
        judge_backend=judge_backend,
    )
    batch_runner = runner_fn or run_eval_batch

    if not resolved_dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {resolved_dataset_path}")

    timed_run = start_timed_run(f"{mode}_eval")
    logger.info(
        "Starting eval layer: suite=%s mode=%s dataset_path=%s run_dir=%s runs_dir=%s run_id=%s config_path=%s limit=%s seed=%s subsets=%s types=%s difficulties=%s expected_behaviors=%s retrieve_top_k=%s rerank_top_k_before=%s rerank_top_k_after=%s context_token_budget=%s context_max_chunks=%s write_trace=%s fail_fast=%s judge_enabled=%s judge_backend=%s",
        suite_normalized,
        mode,
        resolved_dataset_path,
        resolved_run_dir,
        resolved_runs_dir,
        run_id,
        resolved_config_path,
        limit,
        seed,
        sorted(subsets) if subsets else None,
        sorted(types) if types else None,
        sorted(difficulties) if difficulties else None,
        sorted(expected_behaviors) if expected_behaviors else None,
        options.retrieve_top_k,
        options.rerank_top_k_before,
        options.rerank_top_k_after,
        options.context_token_budget,
        options.context_max_chunks,
        options.write_trace,
        options.fail_fast,
        options.judge_enabled,
        options.judge_backend,
    )

    samples = load_eval_samples(resolved_dataset_path)
    filtered_samples = filter_eval_samples(
        samples,
        subsets=subsets,
        types=types,
        difficulties=difficulties,
        expected_behaviors=expected_behaviors,
        limit=limit,
        seed=seed,
    )
    logger.info(
        "Loaded and filtered eval samples: mode=%s samples_loaded=%s samples_after_filtering=%s subset_counts=%s type_counts=%s difficulty_counts=%s expected_behavior_counts=%s",
        mode,
        len(samples),
        len(filtered_samples),
        count_values([sample.subset.value for sample in filtered_samples]),
        count_values([sample.type.value for sample in filtered_samples]),
        count_values([sample.difficulty.value for sample in filtered_samples]),
        count_values([sample.expected_behavior.value for sample in filtered_samples]),
    )
    if not filtered_samples:
        logger.warning(
            "Dataset filtering produced zero samples: mode=%s dataset_path=%s",
            mode,
            resolved_dataset_path,
        )
        raise ValueError("Dataset filtering produced zero samples.")

    runs_root, final_run_dir, resolved_run_id = _resolve_run_target(
        run_dir=resolved_run_dir,
        runs_dir=resolved_runs_dir,
        run_id=run_id,
        mode=mode,
    )
    logger.info(
        "Running eval batch: mode=%s samples_total=%s",
        mode,
        len(filtered_samples),
    )
    results, errors = batch_runner(
        filtered_samples,
        mode=mode,
        options=options,
    )
    cases_failed = len(errors)
    cases_total = len(results)
    cases_successful = cases_total - cases_failed
    failure_rate = 0.0 if cases_total == 0 else cases_failed / float(cases_total)
    logger.info(
        "Eval batch completed: mode=%s cases_total=%s cases_successful=%s cases_failed=%s failure_rate=%.4f errors_total=%s",
        mode,
        cases_total,
        cases_successful,
        cases_failed,
        failure_rate,
        len(errors),
    )

    finished_at, duration_ms = finish_timed_run(timed_run)
    artifacts = write_eval_run_artifacts(
        run_dir=runs_root,
        run_id=resolved_run_id,
        results=results,
        dataset_id=dataset_id_from_path(resolved_dataset_path),
        dataset_path=resolved_dataset_path,
        config_path=resolved_config_path,
        mode=mode,
        started_at=timed_run.started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        build_config={
            "suite": suite_normalized,
            "limit": limit,
            "seed": seed,
            "subsets": sorted(subsets) if subsets else None,
            "types": sorted(types) if types else None,
            "difficulties": sorted(difficulties) if difficulties else None,
            "expected_behaviors": sorted(expected_behaviors) if expected_behaviors else None,
            "errors_total": len(errors),
        },
    )
    logger.info(
        "Eval batch finalized: mode=%s cases_total=%s cases_successful=%s cases_failed=%s failure_rate=%.4f errors_total=%s duration_ms=%s",
        mode,
        cases_total,
        cases_successful,
        cases_failed,
        failure_rate,
        len(errors),
        duration_ms,
    )
    logger.info(
        "Eval artifacts written: mode=%s run_id=%s artifacts_written=true manifest_path=%s cases_path=%s metrics_path=%s summary_path=%s failures_path=%s worst_cases_path=%s report_path=%s",
        mode,
        resolved_run_id,
        artifacts["manifest"],
        artifacts["cases"],
        artifacts["metrics"],
        artifacts["summary"],
        artifacts["failures"],
        artifacts["worst_cases"],
        artifacts["report"],
    )

    return EvalRunExecutionResult(
        run_id=resolved_run_id,
        mode=mode,
        dataset_path=resolved_dataset_path,
        run_dir=final_run_dir,
        cases_total=cases_total,
        cases_successful=cases_successful,
        cases_failed=cases_failed,
        failure_rate=failure_rate,
        metrics_path=Path(artifacts["metrics"]),
        summary_path=Path(artifacts["summary"]),
        report_path=Path(artifacts["report"]),
        duration_ms=duration_ms,
    )


def log_eval_run_summary(result: EvalRunExecutionResult, logger_name: Any) -> None:
    """Log a compact one-line eval run summary to the provided logger."""
    logger_name.info(
        "Eval run complete: run_id=%s cases_total=%s cases_failed=%s report=%s",
        result.run_id,
        result.cases_total,
        result.cases_failed,
        result.report_path,
    )


def build_base_eval_arg_parser(description: str) -> argparse.ArgumentParser:
    """Create an argument parser pre-populated with arguments common to all eval suite layers.

    Callers may extend the returned parser with suite-specific arguments before
    calling ``parse_args()``.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("data/eval/runs"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--subset", action="append", default=[])
    parser.add_argument("--type", dest="types", action="append", default=[])
    parser.add_argument("--difficulty", action="append", default=[])
    parser.add_argument("--expected-behavior", action="append", default=[])
    parser.add_argument("--retrieve-top-k", type=int, default=None)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-backend", type=str, default="heuristic")
    return parser
