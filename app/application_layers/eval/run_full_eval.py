"""CLI entrypoint layer for full eval orchestration.

Exposes ``main()`` so that ``scripts/eval/smoke_run_eval.py`` can import from
``app.application_layers`` without violating the rule that forbids scripts from
importing directly from ``app.application``.

``EvalService`` itself lives in ``app/application/`` because it is a high-level
multi-suite orchestrator; this thin shim is the only bridge between scripts and
that service.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.application.eval_service import EvalService
from app.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full eval orchestration.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--datasets-dir", type=Path, default=Path("data/eval/datasets"))
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("data/eval/runs"))
    parser.add_argument("--run-id-prefix", type=str, default="eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--retrieve-top-k", type=int, default=None)
    parser.add_argument("--rerank-top-k-before", type=int, default=None)
    parser.add_argument("--rerank-top-k-after", type=int, default=None)
    parser.add_argument("--context-token-budget", type=int, default=None)
    parser.add_argument("--context-max-chunks", type=int, default=None)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-backend", type=str, default="heuristic")
    return parser


def main() -> None:
    """Parse CLI arguments and run the full eval suite."""
    setup_logging()
    args = _build_parser().parse_args()
    try:
        EvalService(config_path=args.config).run(
            dataset_path=args.dataset,
            chunks_path=args.chunks,
            datasets_dir=args.datasets_dir,
            dataset_id=args.dataset_id,
            runs_dir=args.runs_dir,
            run_id_prefix=args.run_id_prefix,
            limit=args.limit,
            seed=args.seed,
            retrieve_top_k=args.retrieve_top_k,
            rerank_top_k_before=args.rerank_top_k_before,
            rerank_top_k_after=args.rerank_top_k_after,
            context_token_budget=args.context_token_budget,
            context_max_chunks=args.context_max_chunks,
            write_trace=not args.no_trace,
            fail_fast=args.fail_fast,
            judge_enabled=args.judge,
            judge_backend=args.judge_backend,
        )
    except Exception:
        logger.exception("Full eval run failed")
        sys.exit(1)

    logger.info("Full eval run finished successfully")


if __name__ == "__main__":
    main()
