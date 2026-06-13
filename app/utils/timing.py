"""Small timing utilities for scripts and pipeline orchestration."""

from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

from app.utils.ids import make_run_id


@dataclass(frozen=True)
class TimedRun:
    """Lifecycle timing record for a single stage run."""

    run_id: str
    stage: str
    started_at: str
    started_perf_seconds: float


def now_perf_seconds() -> float:
    """Return a high-resolution monotonic timer value in seconds."""
    return perf_counter()


def elapsed_ms(start_perf_seconds: float, end_perf_seconds: float | None = None) -> int:
    """Return elapsed milliseconds from a monotonic start time."""
    end = perf_counter() if end_perf_seconds is None else end_perf_seconds
    return max(0, int((end - start_perf_seconds) * 1000))


def start_timed_run(stage: str) -> TimedRun:
    """Create and return a started timed run for *stage*."""
    return TimedRun(
        run_id=make_run_id(stage),
        stage=stage,
        started_at=datetime.now(UTC).isoformat(),
        started_perf_seconds=now_perf_seconds(),
    )


def finish_timed_run(run: TimedRun) -> tuple[str, int]:
    """Return finished timestamp and elapsed milliseconds for *run*."""
    finished_at = datetime.now(UTC).isoformat()
    duration_ms = elapsed_ms(run.started_perf_seconds)
    return finished_at, duration_ms
