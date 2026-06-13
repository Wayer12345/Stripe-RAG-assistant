"""Regression comparison helpers for eval run artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.evaluation.reports import load_eval_run_artifacts


class RegressionGate(BaseModel):
    """Quality gate definition for one metric path."""

    model_config = ConfigDict(extra="forbid")

    metric_path: str
    min_value: float | None = None
    max_value: float | None = None
    max_drop: float | None = None
    max_increase: float | None = None
    max_relative_increase: float | None = None
    required: bool = True
    description: str | None = None


class RegressionGateResult(BaseModel):
    """Gate evaluation result."""

    model_config = ConfigDict(extra="forbid")

    metric_path: str
    baseline_value: float | None = None
    candidate_value: float | None = None
    delta: float | None = None
    passed: bool
    reason: str


class RegressionComparisonResult(BaseModel):
    """Comparison output for baseline and candidate eval runs."""

    model_config = ConfigDict(extra="forbid")

    baseline_run_id: str
    candidate_run_id: str
    passed: bool
    gate_results: list[RegressionGateResult]
    metric_deltas: dict[str, float]
    summary: dict[str, Any]


def get_nested_metric(payload: dict[str, Any], metric_path: str) -> float | None:
    """Read numeric metric from nested dict using dotted path."""
    if not metric_path.strip():
        return None
    current: Any = payload
    for part in metric_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if isinstance(current, (int, float)) and not isinstance(current, bool):
        return float(current)
    return None


def flatten_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Flatten nested numeric metrics into dotted-path map."""
    flattened: dict[str, float] = {}

    def _walk(current: Any, prefix: str) -> None:
        if isinstance(current, dict):
            for key, value in current.items():
                next_prefix = f"{prefix}.{key}" if prefix else key
                _walk(value, next_prefix)
            return
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            flattened[prefix] = float(current)

    _walk(payload, "")
    return flattened


def compute_metric_deltas(
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
) -> dict[str, float]:
    """Compute candidate-baseline deltas for shared flattened metrics."""
    baseline_flat = flatten_metrics(baseline_metrics)
    candidate_flat = flatten_metrics(candidate_metrics)
    deltas: dict[str, float] = {}
    for key in sorted(set(baseline_flat).intersection(candidate_flat)):
        deltas[key] = candidate_flat[key] - baseline_flat[key]
    return deltas


def default_regression_gates() -> list[RegressionGate]:
    """Return conservative default quality gates."""
    return [
        RegressionGate(
            metric_path="retrieval.document_recall_at_10_mean",
            max_drop=0.05,
            required=False,
        ),
        RegressionGate(
            metric_path="retrieval.chunk_recall_at_10_mean",
            max_drop=0.05,
            required=False,
        ),
        RegressionGate(
            metric_path="context.context_document_recall_mean",
            max_drop=0.05,
            required=False,
        ),
        RegressionGate(
            metric_path="context.context_chunk_recall_mean",
            max_drop=0.05,
            required=False,
        ),
        RegressionGate(
            metric_path="generation.parsed_successfully_mean",
            max_drop=0.02,
            required=False,
        ),
        RegressionGate(
            metric_path="citation.valid_citation_rate_mean",
            max_drop=0.05,
            required=False,
        ),
        RegressionGate(
            metric_path="citation.invented_source_rate_mean",
            max_increase=0.02,
            required=False,
        ),
        RegressionGate(
            metric_path="latency.total_p95_ms",
            max_relative_increase=0.25,
            required=False,
        ),
    ]


def apply_regression_gates(
    *,
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    gates: list[RegressionGate] | None = None,
) -> list[RegressionGateResult]:
    """Apply quality gates to baseline/candidate metrics."""
    gate_list = gates or default_regression_gates()
    results: list[RegressionGateResult] = []

    for gate in gate_list:
        baseline_value = get_nested_metric(baseline_metrics, gate.metric_path)
        candidate_value = get_nested_metric(candidate_metrics, gate.metric_path)
        delta: float | None = None
        if baseline_value is not None and candidate_value is not None:
            delta = candidate_value - baseline_value

        if baseline_value is None or candidate_value is None:
            if gate.required:
                results.append(
                    RegressionGateResult(
                        metric_path=gate.metric_path,
                        baseline_value=baseline_value,
                        candidate_value=candidate_value,
                        delta=delta,
                        passed=False,
                        reason="Required metric missing in baseline or candidate.",
                    )
                )
            else:
                results.append(
                    RegressionGateResult(
                        metric_path=gate.metric_path,
                        baseline_value=baseline_value,
                        candidate_value=candidate_value,
                        delta=delta,
                        passed=True,
                        reason="Metric missing; gate skipped because required=False.",
                    )
                )
            continue

        passed = True
        reasons: list[str] = []
        if gate.min_value is not None and candidate_value < gate.min_value:
            passed = False
            reasons.append(f"candidate {candidate_value:.6f} < min_value {gate.min_value:.6f}")
        if gate.max_value is not None and candidate_value > gate.max_value:
            passed = False
            reasons.append(f"candidate {candidate_value:.6f} > max_value {gate.max_value:.6f}")
        if gate.max_drop is not None:
            drop = baseline_value - candidate_value
            if drop > gate.max_drop:
                passed = False
                reasons.append(f"drop {drop:.6f} > max_drop {gate.max_drop:.6f}")
        if gate.max_increase is not None:
            increase = candidate_value - baseline_value
            if increase > gate.max_increase:
                passed = False
                reasons.append(f"increase {increase:.6f} > max_increase {gate.max_increase:.6f}")
        if gate.max_relative_increase is not None:
            if baseline_value <= 0:
                if candidate_value > baseline_value:
                    passed = False
                    reasons.append(
                        "relative increase undefined for non-positive baseline, candidate is larger."
                    )
            else:
                relative_increase = (candidate_value - baseline_value) / baseline_value
                if relative_increase > gate.max_relative_increase:
                    passed = False
                    reasons.append(
                        "relative increase "
                        f"{relative_increase:.6f} > max_relative_increase {gate.max_relative_increase:.6f}"
                    )

        results.append(
            RegressionGateResult(
                metric_path=gate.metric_path,
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                delta=delta,
                passed=passed,
                reason="; ".join(reasons) if reasons else "Passed.",
            )
        )

    return results


def compare_eval_runs(
    *,
    baseline_run_dir: Path | str,
    candidate_run_dir: Path | str,
    gates: list[RegressionGate] | None = None,
) -> RegressionComparisonResult:
    """Compare baseline and candidate eval runs from artifact directories."""
    baseline_payload = load_eval_run_artifacts(baseline_run_dir)
    candidate_payload = load_eval_run_artifacts(candidate_run_dir)

    baseline_manifest = baseline_payload.get("manifest", {})
    candidate_manifest = candidate_payload.get("manifest", {})
    baseline_summary = baseline_payload.get("summary", {})
    candidate_summary = candidate_payload.get("summary", {})
    baseline_metrics = baseline_payload.get("metrics", {})
    candidate_metrics = candidate_payload.get("metrics", {})

    baseline_run_id = str(
        baseline_manifest.get("run_id") or baseline_summary.get("run_id") or "baseline"
    )
    candidate_run_id = str(
        candidate_manifest.get("run_id") or candidate_summary.get("run_id") or "candidate"
    )

    gate_results = apply_regression_gates(
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        gates=gates,
    )
    metric_deltas = compute_metric_deltas(baseline_metrics, candidate_metrics)
    passed = all(item.passed for item in gate_results)
    summary = {
        "gates_total": len(gate_results),
        "gates_passed": sum(1 for item in gate_results if item.passed),
        "gates_failed": sum(1 for item in gate_results if not item.passed),
        "baseline_overall_score": baseline_summary.get("overall_score"),
        "candidate_overall_score": candidate_summary.get("overall_score"),
        "overall_score_delta": (
            float(candidate_summary.get("overall_score", 0.0))
            - float(baseline_summary.get("overall_score", 0.0))
        ),
    }
    return RegressionComparisonResult(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        passed=passed,
        gate_results=gate_results,
        metric_deltas=metric_deltas,
        summary=summary,
    )


def build_regression_report_markdown(result: RegressionComparisonResult) -> str:
    """Build human-readable markdown for regression comparison."""
    lines: list[str] = []
    lines.append("# Eval Regression Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Baseline: {result.baseline_run_id}")
    lines.append(f"- Candidate: {result.candidate_run_id}")
    lines.append(f"- Passed: {result.passed}")
    lines.append("")
    lines.append("## Gate Results")
    lines.append("| Metric | Baseline | Candidate | Delta | Passed | Reason |")
    lines.append("| --- | ---: | ---: | ---: | :---: | --- |")
    for gate in result.gate_results:
        baseline_value = "" if gate.baseline_value is None else f"{gate.baseline_value:.6f}"
        candidate_value = "" if gate.candidate_value is None else f"{gate.candidate_value:.6f}"
        delta = "" if gate.delta is None else f"{gate.delta:.6f}"
        lines.append(
            f"| {gate.metric_path} | {baseline_value} | {candidate_value} | "
            f"{delta} | {'yes' if gate.passed else 'no'} | {gate.reason} |"
        )
    lines.append("")
    lines.append("## Metric Deltas")
    for key in sorted(result.metric_deltas.keys()):
        lines.append(f"- {key}: {result.metric_deltas[key]:.6f}")
    lines.append("")
    return "\n".join(lines)


def write_regression_report(
    *,
    result: RegressionComparisonResult,
    output_path: Path | str,
) -> Path:
    """Write regression markdown report and return output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown = build_regression_report_markdown(result)
    path.write_text(markdown, encoding="utf-8")
    return path
