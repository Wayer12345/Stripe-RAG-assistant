"""Deterministic latency helpers for eval."""

from __future__ import annotations


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _validated(values_ms: list[float | int]) -> list[float]:
    values = [float(value) for value in values_ms]
    if any(value < 0 for value in values):
        raise ValueError("Latency values must be >= 0.")
    return values


def mean_latency(values_ms: list[float | int]) -> float:
    values = _validated(values_ms)
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def percentile_latency(values_ms: list[float | int], percentile: float) -> float:
    """Deterministic percentile using linear interpolation."""
    if percentile < 0 or percentile > 100:
        raise ValueError("percentile must be in [0, 100].")
    values = sorted(_validated(values_ms))
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (percentile / 100.0) * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def latency_summary(values_ms: list[float | int], prefix: str) -> dict[str, float]:
    values = _validated(values_ms)
    if not values:
        return {
            f"{prefix}_mean_ms": 0.0,
            f"{prefix}_p50_ms": 0.0,
            f"{prefix}_p90_ms": 0.0,
            f"{prefix}_p95_ms": 0.0,
            f"{prefix}_p99_ms": 0.0,
            f"{prefix}_max_ms": 0.0,
        }
    return {
        f"{prefix}_mean_ms": mean_latency(values),
        f"{prefix}_p50_ms": percentile_latency(values, 50),
        f"{prefix}_p90_ms": percentile_latency(values, 90),
        f"{prefix}_p95_ms": percentile_latency(values, 95),
        f"{prefix}_p99_ms": percentile_latency(values, 99),
        f"{prefix}_max_ms": max(values),
    }


def stage_latency_summary(
    *,
    retrieve_ms: list[float | int] | None = None,
    rerank_ms: list[float | int] | None = None,
    context_ms: list[float | int] | None = None,
    generation_ms: list[float | int] | None = None,
    total_ms: list[float | int] | None = None,
) -> dict[str, float]:
    """Build stage-wise latency summaries."""
    summary: dict[str, float] = {}
    summary.update(latency_summary(retrieve_ms or [], "retrieval"))
    summary.update(latency_summary(rerank_ms or [], "rerank"))
    summary.update(latency_summary(context_ms or [], "context"))
    summary.update(latency_summary(generation_ms or [], "generation"))
    summary.update(latency_summary(total_ms or [], "total"))
    return summary


def latency_budget_violation_rate(values_ms: list[float | int], budget_ms: float | int) -> float:
    if budget_ms < 0:
        raise ValueError("budget_ms must be >= 0.")
    values = _validated(values_ms)
    if not values:
        return 0.0
    violations = sum(1 for value in values if value > float(budget_ms))
    return _safe_divide(float(violations), float(len(values)))
