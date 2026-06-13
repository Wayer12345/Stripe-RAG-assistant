"""Minimal online query service orchestration."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.application_layers.online.build_context import BuildContextLayer
from app.application_layers.online.generate_answer import GenerateAnswerLayer
from app.application_layers.online.rerank import RerankLayer
from app.application_layers.online.retrieve import RetrieveLayer
from app.infrastructure.retrieval.retriever_factory import shutdown_retriever_cache
from app.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@dataclass(frozen=True)
class QueryServiceResult:
    """Compact result returned by QueryService."""

    answer: str
    confidence: str
    sources_total: int


class QueryService:
    """Orchestrates online query flow from existing layers."""

    def __init__(self, *, config_path: Path | str = Path("configs/config.yaml")) -> None:
        self._config_path = Path(config_path)

    def run(
        self,
        *,
        question: str,
        filters: dict[str, Any] | None = None,
        retrieve_top_k: int | None = None,
        rerank_top_k_before: int | None = None,
        rerank_top_k_after: int | None = None,
        context_token_budget: int | None = None,
        context_max_chunks: int | None = None,
        write_trace: bool | None = None,
    ) -> QueryServiceResult:
        retrieve = RetrieveLayer(
            question=question,
            config_path=self._config_path,
            top_k=retrieve_top_k,
            filters=filters,
            write_trace=write_trace,
        ).run()
        rerank = RerankLayer(
            question=question,
            candidates=retrieve.results,
            config_path=self._config_path,
            top_k_before=rerank_top_k_before,
            top_k_after=rerank_top_k_after,
            write_trace=write_trace,
        ).run()
        context = BuildContextLayer(
            question=question,
            results=rerank.results,
            config_path=self._config_path,
            token_budget=context_token_budget,
            max_chunks=context_max_chunks,
            write_trace=write_trace,
        ).run()
        generation = GenerateAnswerLayer(
            question=question,
            context_bundle=context.context_bundle,
            config_path=self._config_path,
            write_trace=write_trace,
        ).run()
        return QueryServiceResult(
            answer=generation.generated_answer.answer,
            confidence=generation.confidence,
            sources_total=generation.sources_total,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run online QueryService orchestration.")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    return parser


def main() -> None:
    setup_logging()

    args = _build_parser().parse_args()

    try:
        result = QueryService(config_path=args.config).run(question=args.question)
    except Exception:
        logger.exception("QueryService failed")
        sys.exit(1)
    finally:
        shutdown_retriever_cache()

    logger.info("Confidence: %s | sources_total: %s", result.confidence, result.sources_total)
    logger.info("Answer:\n%s", result.answer)


if __name__ == "__main__":
    main()
