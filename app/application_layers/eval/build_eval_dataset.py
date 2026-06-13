"""Application-layer builder for eval datasets from chunks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from app.evaluation.dataset_builder import build_eval_dataset_from_chunks
from app.evaluation.records import EvalSubset
from app.utils.config import load_settings, resolve_config_dir_and_path
from app.utils.ids import make_run_id
from app.utils.logging import get_logger, setup_logging
from app.utils.timing import finish_timed_run, start_timed_run

logger = get_logger(__name__)


@dataclass(frozen=True)
class BuildEvalDatasetResult:
    dataset_id: str
    dataset_dir: Path
    dataset_path: Path
    manifest_path: Path
    audit_path: Path | None
    samples_total: int
    synthetic_samples_total: int
    negative_samples_total: int
    robustness_samples_total: int
    audit_samples_total: int
    duration_ms: int


class BuildEvalDatasetLayer:
    def __init__(
        self,
        *,
        chunks_path: Path | str,
        output_dir: Path | str = Path("data/eval/datasets"),
        dataset_id: str | None = None,
        synthetic_target_size: int | None = None,
        negative_target_size: int | None = None,
        robustness_target_size: int | None = None,
        audit_target_size: int | None = None,
        seed: int | None = None,
        min_chunk_chars: int | None = None,
        config_path: Path | str = Path("configs/config.yaml"),
    ) -> None:
        self._chunks_path = Path(chunks_path)
        self._output_dir = Path(output_dir)
        self._dataset_id = dataset_id
        self._synthetic_target_size = synthetic_target_size
        self._negative_target_size = negative_target_size
        self._robustness_target_size = robustness_target_size
        self._audit_target_size = audit_target_size
        self._seed = seed
        self._min_chunk_chars = min_chunk_chars
        self._config_path = Path(config_path)

    def run(self) -> BuildEvalDatasetResult:
        if not self._chunks_path.exists():
            logger.error("Eval dataset build failed: chunks path not found: %s", self._chunks_path)
            raise FileNotFoundError(f"Chunks path not found: {self._chunks_path}")

        config_dir, _ = resolve_config_dir_and_path(self._config_path)
        settings = load_settings(config_dir)
        setup_logging(settings)

        dataset_id = self._dataset_id or make_run_id("eval_dataset")
        synthetic_target_size = (
            self._synthetic_target_size
            if self._synthetic_target_size is not None
            else settings.eval.default_synthetic_target_size
        )
        negative_target_size = (
            self._negative_target_size
            if self._negative_target_size is not None
            else settings.eval.default_negative_target_size
        )
        robustness_target_size = (
            self._robustness_target_size
            if self._robustness_target_size is not None
            else settings.eval.default_robustness_target_size
        )
        audit_target_size = (
            self._audit_target_size
            if self._audit_target_size is not None
            else settings.eval.default_audit_target_size
        )
        seed = self._seed if self._seed is not None else settings.eval.default_seed
        min_chunk_chars = (
            self._min_chunk_chars
            if self._min_chunk_chars is not None
            else settings.eval.min_chunk_chars
        )

        logger.info(
            "Starting eval dataset build: stage=build_eval_dataset chunks_path=%s output_dir=%s dataset_id=%s synthetic_target_size=%s negative_target_size=%s robustness_target_size=%s audit_target_size=%s seed=%s min_chunk_chars=%s config_path=%s",
            self._chunks_path,
            self._output_dir,
            dataset_id,
            synthetic_target_size,
            negative_target_size,
            robustness_target_size,
            audit_target_size,
            seed,
            min_chunk_chars,
            self._config_path,
        )
        timed_run = start_timed_run("build_eval_dataset")
        dataset = build_eval_dataset_from_chunks(
            chunks_path=self._chunks_path,
            dataset_id=dataset_id,
            output_dir=self._output_dir,
            synthetic_target_size=synthetic_target_size,
            negative_target_size=negative_target_size,
            robustness_target_size=robustness_target_size,
            audit_target_size=audit_target_size,
            seed=seed,
            min_chunk_chars=min_chunk_chars,
        )
        _, duration_ms = finish_timed_run(timed_run)

        dataset_dir = self._output_dir / dataset_id
        subset_counts = dataset.subset_counts()
        build_stats = {}
        if dataset.manifest is not None and dataset.manifest.build_config is not None:
            build_stats = dataset.manifest.build_config.get("build_stats", {})
        eligible_chunks = build_stats.get("eligible_chunks_total")
        dropped_chunks = build_stats.get("dropped_chunks_total")
        dropped_reasons = build_stats.get("dropped_reasons")
        if eligible_chunks is not None or dropped_chunks is not None:
            logger.info(
                "Eval dataset build quality stats: eligible_chunks=%s dropped_chunks=%s dropped_reasons=%s",
                eligible_chunks,
                dropped_chunks,
                dropped_reasons,
            )
        audit_path = dataset_dir / "audit.jsonl"
        result = BuildEvalDatasetResult(
            dataset_id=dataset_id,
            dataset_dir=dataset_dir,
            dataset_path=dataset_dir / "dataset.jsonl",
            manifest_path=dataset_dir / "manifest.json",
            audit_path=audit_path if audit_path.exists() else None,
            samples_total=len(dataset),
            synthetic_samples_total=subset_counts.get(EvalSubset.SYNTHETIC_SOURCE_GROUNDED.value, 0),
            negative_samples_total=subset_counts.get(EvalSubset.NEGATIVE.value, 0),
            robustness_samples_total=subset_counts.get(EvalSubset.ROBUSTNESS.value, 0),
            audit_samples_total=subset_counts.get(EvalSubset.AUDIT.value, 0),
            duration_ms=duration_ms,
        )
        logger.info(
            "Eval dataset build result: dataset_id=%s samples_total=%s synthetic_samples_total=%s negative_samples_total=%s robustness_samples_total=%s audit_samples_total=%s eligible_chunks_total=%s dropped_chunks_total=%s dropped_reasons=%s dataset_path=%s manifest_path=%s audit_path=%s duration_ms=%s",
            result.dataset_id,
            result.samples_total,
            result.synthetic_samples_total,
            result.negative_samples_total,
            result.robustness_samples_total,
            result.audit_samples_total,
            eligible_chunks,
            dropped_chunks,
            dropped_reasons if isinstance(dropped_reasons, dict) else None,
            result.dataset_path,
            result.manifest_path,
            result.audit_path,
            result.duration_ms,
        )
        return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build eval dataset from chunk artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval/datasets"))
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--synthetic-target-size", type=int, default=None)
    parser.add_argument("--negative-target-size", type=int, default=None)
    parser.add_argument("--robustness-target-size", type=int, default=None)
    parser.add_argument("--audit-target-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-chunk-chars", type=int, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    return parser


def main() -> None:
    setup_logging()
    args = _build_arg_parser().parse_args()
    result = BuildEvalDatasetLayer(
        chunks_path=args.chunks,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        synthetic_target_size=args.synthetic_target_size,
        negative_target_size=args.negative_target_size,
        robustness_target_size=args.robustness_target_size,
        audit_target_size=args.audit_target_size,
        seed=args.seed,
        min_chunk_chars=args.min_chunk_chars,
        config_path=args.config,
    ).run()
    logger.info(
        "Build eval dataset complete: dataset_id=%s dataset_path=%s samples_total=%s duration_ms=%s",
        result.dataset_id,
        result.dataset_path,
        result.samples_total,
        result.duration_ms,
    )


if __name__ == "__main__":
    main()
