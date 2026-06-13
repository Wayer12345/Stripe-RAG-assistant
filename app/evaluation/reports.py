"""Core eval run report and artifact builders."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.evaluation.latency_metrics import stage_latency_summary
from app.evaluation.records import EvalCaseResult, EvalRunManifest, EvalRunPaths, EvalRunSummary
from app.infrastructure.storage.jsonl_store import JsonlStore, write_jsonl
from app.infrastructure.storage.manifest_store import read_manifest, write_json_payload, write_manifest
from app.utils.constants import ARTIFACT_SCHEMA_VERSION


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _question_preview(question: str, max_chars: int = 120) -> str:
    normalized = " ".join(question.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


def _metric_from_case(result: EvalCaseResult, metric_key: str) -> float | None:
    value = result.metrics.get(metric_key)
    if _is_numeric(value):
        return float(value)
    return None


def _case_overall_score(result: EvalCaseResult) -> float:
    selected: list[float] = []
    preferred = [
        "retrieval.document_recall_at_10",
        "retrieval.chunk_recall_at_10",
        "context.context_document_recall",
        "context.context_chunk_recall",
        "generation.valid_generation_output",
        "citation.valid_citation_rate",
        "judge.groundedness_score",
        "judge.source_support_score",
    ]
    for key in preferred:
        value = _metric_from_case(result, key)
        if value is not None:
            selected.append(value)

    answer_on_unanswerable = _metric_from_case(result, "confidence.answer_on_unanswerable")
    if answer_on_unanswerable is not None:
        selected.append(1.0 - answer_on_unanswerable)

    if not selected:
        return 0.0
    return sum(selected) / float(len(selected))


def _aggregate_metric_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0.0}
    return {
        "mean": sum(values) / float(len(values)),
        "min": min(values),
        "max": max(values),
        "count": float(len(values)),
    }


def build_eval_run_paths(*, runs_dir: Path | str, run_id: str) -> EvalRunPaths:
    """Build canonical artifact paths for one eval run."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty.")
    base_dir = Path(runs_dir) / run_id
    return EvalRunPaths(
        run_dir=str(base_dir),
        manifest_path=str(base_dir / "manifest.json"),
        cases_path=str(base_dir / "cases.jsonl"),
        metrics_path=str(base_dir / "metrics.json"),
        summary_path=str(base_dir / "summary.json"),
        failures_path=str(base_dir / "failures.jsonl"),
        worst_cases_path=str(base_dir / "worst_cases.json"),
        report_path=str(base_dir / "report.md"),
    )


def serialize_eval_case_result(result: EvalCaseResult) -> dict[str, Any]:
    """Serialize EvalCaseResult into JSON-safe dictionary."""
    return result.model_dump(mode="json")


def build_failure_rows(results: list[EvalCaseResult]) -> list[dict[str, Any]]:
    """Extract compact failure rows for failed/error cases."""
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.error is None and result.passed:
            continue
        rows.append(
            {
                "sample_id": result.sample_id,
                "question": result.question,
                "subset": result.subset,
                "type": result.type,
                "difficulty": result.difficulty,
                "expected_behavior": result.expected_behavior,
                "error": result.error,
                "trace_paths": dict(result.trace_paths),
                "latency_ms": dict(result.latency_ms),
            }
        )
    return rows


def build_worst_cases(
    results: list[EvalCaseResult],
    *,
    worst_n: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Build worst-case buckets from per-case results."""
    if worst_n <= 0:
        raise ValueError("worst_n must be > 0.")

    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "sample_id": result.sample_id,
                "question_preview": _question_preview(result.question),
                "subset": result.subset,
                "type": result.type,
                "overall_score": _case_overall_score(result),
                "retrieval_score": _metric_from_case(result, "retrieval.chunk_recall_at_10"),
                "context_score": _metric_from_case(result, "context.context_chunk_recall"),
                "invented_source_rate": _metric_from_case(result, "citation.invented_source_rate"),
                "latency_total_ms": float(result.latency_ms.get("total", 0.0)),
                "error": result.error,
            }
        )

    lowest_overall_score = sorted(rows, key=lambda row: row["overall_score"])[:worst_n]
    lowest_retrieval_score = sorted(
        [row for row in rows if row["retrieval_score"] is not None],
        key=lambda row: float(row["retrieval_score"]),
    )[:worst_n]
    lowest_context_score = sorted(
        [row for row in rows if row["context_score"] is not None],
        key=lambda row: float(row["context_score"]),
    )[:worst_n]
    highest_invented_source_rate = sorted(
        [row for row in rows if row["invented_source_rate"] is not None],
        key=lambda row: float(row["invented_source_rate"]),
        reverse=True,
    )[:worst_n]
    highest_latency = sorted(rows, key=lambda row: float(row["latency_total_ms"]), reverse=True)[:worst_n]
    failed_cases = [row for row in rows if row["error"] is not None][:worst_n]

    return {
        "lowest_overall_score": lowest_overall_score,
        "lowest_retrieval_score": lowest_retrieval_score,
        "lowest_context_score": lowest_context_score,
        "highest_invented_source_rate": highest_invented_source_rate,
        "highest_latency": highest_latency,
        "failed_cases": failed_cases,
    }


def build_metrics_summary(results: list[EvalCaseResult]) -> dict[str, Any]:
    """Aggregate namespaced per-case metrics into grouped summary."""
    if not results:
        return {
            "latency": stage_latency_summary(),
        }

    grouped_values: dict[str, dict[str, list[float]]] = {}
    for result in results:
        for metric_key, value in result.metrics.items():
            if not _is_numeric(value) or "." not in metric_key:
                continue
            group, metric_name = metric_key.split(".", 1)
            group_bucket = grouped_values.setdefault(group, {})
            metric_values = group_bucket.setdefault(metric_name, [])
            metric_values.append(float(value))

    summary: dict[str, Any] = {}
    for group, metrics_payload in grouped_values.items():
        group_summary: dict[str, float] = {}
        for metric_name, values in metrics_payload.items():
            stats = _aggregate_metric_values(values)
            group_summary[f"{metric_name}_mean"] = stats["mean"]
            group_summary[f"{metric_name}_min"] = stats["min"]
            group_summary[f"{metric_name}_max"] = stats["max"]
            group_summary[f"{metric_name}_count"] = stats["count"]
        summary[group] = group_summary

    latency_inputs = {
        "retrieve_ms": [float(result.latency_ms.get("retrieve", 0.0)) for result in results],
        "rerank_ms": [float(result.latency_ms.get("rerank", 0.0)) for result in results],
        "context_ms": [float(result.latency_ms.get("context", 0.0)) for result in results],
        "generation_ms": [float(result.latency_ms.get("generation", 0.0)) for result in results],
        "total_ms": [float(result.latency_ms.get("total", 0.0)) for result in results],
    }
    summary["latency"] = stage_latency_summary(**latency_inputs)
    return summary


def build_eval_run_summary(
    *,
    run_id: str,
    results: list[EvalCaseResult],
    metrics: dict[str, Any],
    started_at: str | None = None,
    finished_at: str | None = None,
    duration_ms: int | float | None = None,
) -> EvalRunSummary:
    """Build compact eval run summary from per-case outputs."""
    cases_total = len(results)
    cases_failed = sum(1 for result in results if result.error is not None)
    cases_successful = cases_total - cases_failed
    failure_rate = _safe_divide(float(cases_failed), float(cases_total))
    overall_values = [_case_overall_score(result) for result in results]
    overall_score = sum(overall_values) / float(len(overall_values)) if overall_values else 0.0
    metric_groups = sorted(metrics.keys())
    status = "failed" if cases_failed > 0 else "success"
    return EvalRunSummary(
        run_id=run_id,
        status=status,
        cases_total=cases_total,
        cases_successful=cases_successful,
        cases_failed=cases_failed,
        failure_rate=failure_rate,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=float(duration_ms) if duration_ms is not None else None,
        metric_groups=metric_groups,
        overall_score=overall_score,
        latency=dict(metrics.get("latency", {})) if isinstance(metrics.get("latency"), dict) else {},
    )


def build_eval_run_manifest(
    *,
    run_id: str,
    dataset_id: str | None,
    dataset_path: Path | str | None,
    run_paths: EvalRunPaths,
    started_at: str | None = None,
    finished_at: str | None = None,
    config_path: Path | str | None = None,
    mode: str | None = None,
    build_config: dict[str, Any] | None = None,
    notes: str | None = None,
) -> EvalRunManifest:
    """Build run manifest metadata."""
    return EvalRunManifest(
        run_id=run_id,
        schema_version=ARTIFACT_SCHEMA_VERSION,
        dataset_id=dataset_id,
        dataset_path=str(dataset_path) if dataset_path is not None else None,
        mode=mode,
        created_at=_now_iso(),
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=None,
        config_path=str(config_path) if config_path is not None else None,
        artifact_paths=run_paths,
        build_config=build_config or {},
        notes=notes,
    )


def build_report_markdown(
    *,
    manifest: EvalRunManifest,
    summary: EvalRunSummary,
    metrics: dict[str, Any],
    worst_cases: dict[str, list[dict[str, Any]]],
    failures: list[dict[str, Any]],
) -> str:
    """Build compact markdown report for one eval run."""
    lines: list[str] = []
    lines.append(f"# Eval Report: {manifest.run_id}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Status: {summary.status}")
    lines.append(f"- Cases total: {summary.cases_total}")
    lines.append(f"- Successful: {summary.cases_successful}")
    lines.append(f"- Failed: {summary.cases_failed}")
    lines.append(f"- Failure rate: {summary.failure_rate:.3f}")
    lines.append(f"- Overall score: {summary.overall_score:.3f}")
    lines.append(f"- Duration (ms): {summary.duration_ms if summary.duration_ms is not None else 0.0}")
    lines.append("")
    lines.append("## Metric Groups")
    for group_name in sorted(metrics.keys()):
        if group_name == "latency":
            continue
        group_payload = metrics[group_name]
        if not isinstance(group_payload, dict):
            continue
        lines.append(f"### {group_name}")
        for key in sorted(group_payload.keys()):
            lines.append(f"- {key}: {group_payload[key]}")
    lines.append("")
    lines.append("## Latency")
    latency_payload = metrics.get("latency", {})
    if isinstance(latency_payload, dict):
        for key in sorted(latency_payload.keys()):
            lines.append(f"- {key}: {latency_payload[key]}")
    lines.append("")
    lines.append("## Worst Cases")
    for bucket_name in sorted(worst_cases.keys()):
        lines.append(f"### {bucket_name}")
        for row in worst_cases.get(bucket_name, [])[:10]:
            lines.append(
                f"- {row.get('sample_id')}: {row.get('question_preview')} "
                f"(subset={row.get('subset')}, type={row.get('type')})"
            )
    lines.append("")
    lines.append("## Failures")
    if not failures:
        lines.append("- None")
    else:
        for row in failures[:20]:
            lines.append(f"- {row.get('sample_id')}: {row.get('error')}")
    lines.append("")
    return "\n".join(lines)


def write_eval_run_artifacts(
    *,
    run_dir: Path | str,
    run_id: str,
    results: list[EvalCaseResult],
    dataset_id: str | None = None,
    dataset_path: Path | str | None = None,
    config_path: Path | str | None = None,
    mode: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    duration_ms: int | float | None = None,
    build_config: dict[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Path]:
    """Write full eval artifact set under run directory."""
    run_paths = build_eval_run_paths(runs_dir=run_dir, run_id=run_id)
    target_dir = Path(run_paths.run_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    serialized_cases = [serialize_eval_case_result(result) for result in results]
    failures = build_failure_rows(results)
    worst_cases = build_worst_cases(results)
    metrics = build_metrics_summary(results)
    summary = build_eval_run_summary(
        run_id=run_id,
        results=results,
        metrics=metrics,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
    )
    manifest = build_eval_run_manifest(
        run_id=run_id,
        dataset_id=dataset_id,
        dataset_path=dataset_path,
        run_paths=run_paths,
        started_at=started_at,
        finished_at=finished_at,
        config_path=config_path,
        mode=mode,
        build_config=build_config,
        notes=notes,
    )
    manifest.duration_ms = summary.duration_ms
    report_md = build_report_markdown(
        manifest=manifest,
        summary=summary,
        metrics=metrics,
        worst_cases=worst_cases,
        failures=failures,
    )

    manifest_path = Path(run_paths.manifest_path)
    cases_path = Path(run_paths.cases_path or target_dir / "cases.jsonl")
    metrics_path = Path(run_paths.metrics_path or target_dir / "metrics.json")
    summary_path = Path(run_paths.summary_path or target_dir / "summary.json")
    failures_path = Path(run_paths.failures_path or target_dir / "failures.jsonl")
    worst_cases_path = Path(run_paths.worst_cases_path or target_dir / "worst_cases.json")
    report_path = Path(run_paths.report_path or target_dir / "report.md")

    write_manifest(manifest_path, manifest)
    write_jsonl(cases_path, serialized_cases)
    write_json_payload(metrics_path, metrics)
    write_json_payload(summary_path, summary)
    write_jsonl(failures_path, failures)
    write_json_payload(worst_cases_path, worst_cases)
    report_path.write_text(report_md, encoding="utf-8")

    return {
        "run_dir": target_dir,
        "manifest": manifest_path,
        "cases": cases_path,
        "metrics": metrics_path,
        "summary": summary_path,
        "failures": failures_path,
        "worst_cases": worst_cases_path,
        "report": report_path,
    }


def load_eval_run_artifacts(run_dir: Path | str) -> dict[str, Any]:
    """Load all standard eval run artifacts from run directory."""
    base_dir = Path(run_dir)
    manifest_path = base_dir / "manifest.json"
    cases_path = base_dir / "cases.jsonl"
    metrics_path = base_dir / "metrics.json"
    summary_path = base_dir / "summary.json"
    failures_path = base_dir / "failures.jsonl"
    worst_cases_path = base_dir / "worst_cases.json"
    report_path = base_dir / "report.md"

    store = JsonlStore()
    return {
        "manifest": read_manifest(manifest_path),
        "cases": store.read(cases_path),
        "metrics": read_manifest(metrics_path),
        "summary": read_manifest(summary_path),
        "failures": store.read(failures_path),
        "worst_cases": read_manifest(worst_cases_path),
        "report_markdown": report_path.read_text(encoding="utf-8"),
        "paths": {
            "manifest": manifest_path,
            "cases": cases_path,
            "metrics": metrics_path,
            "summary": summary_path,
            "failures": failures_path,
            "worst_cases": worst_cases_path,
            "report": report_path,
        },
    }
