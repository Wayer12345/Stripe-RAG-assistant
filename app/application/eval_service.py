"""Minimal eval service orchestration."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.application_layers.eval.build_eval_dataset import BuildEvalDatasetLayer
from app.application_layers.eval.run_citation_eval import RunCitationEvalLayer
from app.application_layers.eval.run_context_eval import RunContextEvalLayer
from app.application_layers.eval.run_generation_eval import RunGenerationEvalLayer
from app.application_layers.eval.run_rerank_eval import RunRerankEvalLayer
from app.application_layers.eval.run_retrieval_eval import RunRetrievalEvalLayer
from app.application_layers.eval.run_robustness_eval import RunRobustnessEvalLayer
from app.evaluation.datasets import ensure_unique_eval_samples_file
from app.infrastructure.embeddings.embedder_factory import create_embedder
from app.infrastructure.generation.ollama_client import OllamaClient
from app.infrastructure.retrieval.dense_retriever import DenseRetriever
from app.infrastructure.vector_stores.qdrant_store import QdrantStore
from app.utils.config import Settings, load_settings, resolve_config_dir_and_path
from app.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class EvalService:
    """Run eval layers sequentially from one service method."""

    def __init__(self, *, config_path: Path | str = Path("configs/config.yaml")) -> None:
        self._config_path = Path(config_path)

    def run(
        self,
        *,
        dataset_path: Path | str | None = None,
        chunks_path: Path | str | None = None,
        datasets_dir: Path | str | None = None,
        dataset_id: str | None = None,
        runs_dir: Path | str | None = None,
        run_id_prefix: str = "eval",
        limit: int | None = None,
        seed: int | None = None,
        retrieve_top_k: int | None = None,
        rerank_top_k_before: int | None = None,
        rerank_top_k_after: int | None = None,
        context_token_budget: int | None = None,
        context_max_chunks: int | None = None,
        write_trace: bool | None = None,
        fail_fast: bool | None = None,
        judge_enabled: bool | None = None,
        judge_backend: str | None = None,
    ) -> None:
        settings = self._load_settings()

        # Resolve defaults from config when not explicitly provided.
        resolved_datasets_dir = Path(datasets_dir) if datasets_dir is not None else settings.eval.datasets_dir
        resolved_runs_dir = Path(runs_dir) if runs_dir is not None else settings.eval.runs_dir
        resolved_fail_fast = bool(fail_fast) if fail_fast is not None else settings.eval.fail_fast
        resolved_judge_enabled = judge_enabled if judge_enabled is not None else settings.eval.judge_enabled
        resolved_judge_backend = judge_backend if judge_backend is not None else settings.eval.judge_backend
        resolved_write_trace = write_trace if write_trace is not None else settings.eval.write_trace
        resolved_seed = seed if seed is not None else settings.eval.default_seed
        resolved_limit = limit if limit is not None else settings.eval.default_limit

        self._run_preflight_checks(settings)

        resolved_dataset_path: Path
        if dataset_path is not None:
            resolved_dataset_path = Path(dataset_path)
        elif chunks_path is not None:
            dataset_result = BuildEvalDatasetLayer(
                config_path=self._config_path,
                chunks_path=chunks_path,
                output_dir=resolved_datasets_dir,
                dataset_id=dataset_id,
                seed=resolved_seed,
            ).run()
            resolved_dataset_path = dataset_result.dataset_path
        else:
            raise ValueError("Either dataset_path or chunks_path must be provided.")

        normalized_dataset_path, renamed_total = ensure_unique_eval_samples_file(
            resolved_dataset_path
        )
        resolved_dataset_path = normalized_dataset_path
        if renamed_total > 0:
            logger.warning(
                "Dataset contained duplicate sample ids; rewritten with unique ids: dataset_path=%s renamed_total=%s",
                resolved_dataset_path,
                renamed_total,
            )

        RunRetrievalEvalLayer(
            config_path=self._config_path,
            dataset_path=resolved_dataset_path,
            runs_dir=resolved_runs_dir,
            run_id=f"{run_id_prefix}_retrieval",
            limit=resolved_limit,
            seed=resolved_seed,
            retrieve_top_k=retrieve_top_k,
            write_trace=resolved_write_trace,
            fail_fast=resolved_fail_fast,
            judge_enabled=resolved_judge_enabled,
            judge_backend=resolved_judge_backend,
        ).run()
        RunRerankEvalLayer(
            config_path=self._config_path,
            dataset_path=resolved_dataset_path,
            runs_dir=resolved_runs_dir,
            run_id=f"{run_id_prefix}_rerank",
            limit=resolved_limit,
            seed=resolved_seed,
            retrieve_top_k=retrieve_top_k,
            rerank_top_k_before=rerank_top_k_before,
            rerank_top_k_after=rerank_top_k_after,
            write_trace=resolved_write_trace,
            fail_fast=resolved_fail_fast,
            judge_enabled=resolved_judge_enabled,
            judge_backend=resolved_judge_backend,
        ).run()
        RunContextEvalLayer(
            config_path=self._config_path,
            dataset_path=resolved_dataset_path,
            runs_dir=resolved_runs_dir,
            run_id=f"{run_id_prefix}_context",
            limit=resolved_limit,
            seed=resolved_seed,
            retrieve_top_k=retrieve_top_k,
            rerank_top_k_before=rerank_top_k_before,
            rerank_top_k_after=rerank_top_k_after,
            context_token_budget=context_token_budget,
            context_max_chunks=context_max_chunks,
            write_trace=resolved_write_trace,
            fail_fast=resolved_fail_fast,
            judge_enabled=resolved_judge_enabled,
            judge_backend=resolved_judge_backend,
        ).run()
        RunGenerationEvalLayer(
            config_path=self._config_path,
            dataset_path=resolved_dataset_path,
            runs_dir=resolved_runs_dir,
            run_id=f"{run_id_prefix}_generation",
            limit=resolved_limit,
            seed=resolved_seed,
            retrieve_top_k=retrieve_top_k,
            rerank_top_k_before=rerank_top_k_before,
            rerank_top_k_after=rerank_top_k_after,
            context_token_budget=context_token_budget,
            context_max_chunks=context_max_chunks,
            write_trace=resolved_write_trace,
            fail_fast=resolved_fail_fast,
            judge_enabled=resolved_judge_enabled,
            judge_backend=resolved_judge_backend,
        ).run()
        RunCitationEvalLayer(
            config_path=self._config_path,
            dataset_path=resolved_dataset_path,
            runs_dir=resolved_runs_dir,
            run_id=f"{run_id_prefix}_citation",
            limit=resolved_limit,
            seed=resolved_seed,
            retrieve_top_k=retrieve_top_k,
            rerank_top_k_before=rerank_top_k_before,
            rerank_top_k_after=rerank_top_k_after,
            context_token_budget=context_token_budget,
            context_max_chunks=context_max_chunks,
            write_trace=resolved_write_trace,
            fail_fast=resolved_fail_fast,
            judge_enabled=resolved_judge_enabled,
            judge_backend=resolved_judge_backend,
        ).run()
        RunRobustnessEvalLayer(
            config_path=self._config_path,
            dataset_path=resolved_dataset_path,
            runs_dir=resolved_runs_dir,
            run_id=f"{run_id_prefix}_robustness",
            limit=resolved_limit,
            seed=resolved_seed,
            retrieve_top_k=retrieve_top_k,
            rerank_top_k_before=rerank_top_k_before,
            rerank_top_k_after=rerank_top_k_after,
            context_token_budget=context_token_budget,
            context_max_chunks=context_max_chunks,
            write_trace=resolved_write_trace,
            fail_fast=resolved_fail_fast,
            judge_enabled=resolved_judge_enabled,
            judge_backend=resolved_judge_backend,
        ).run()

    def _load_settings(self) -> Settings:
        config_dir, _ = resolve_config_dir_and_path(self._config_path)
        return load_settings(config_dir)

    def _run_preflight_checks(self, settings: Settings) -> None:
        preflight = settings.eval.preflight
        if not preflight.enabled:
            logger.info("Eval preflight checks are disabled via config.")
            return

        logger.info("Starting eval preflight checks.")

        retrieval_warmup: dict[str, Any] = {}
        if (
            preflight.require_qdrant_healthcheck
            or preflight.require_embed_query_warmup
            or preflight.require_tiny_search_warmup
        ):
            retriever = self._build_preflight_retriever(settings)
            try:
                retrieval_warmup = retriever.warmup()
            finally:
                self._try_close_vector_store(retriever)

        generation_warmup: dict[str, Any] = {}
        if preflight.require_ollama_healthcheck or preflight.require_ollama_generate_warmup:
            ollama_client = OllamaClient(
                base_url=settings.generation.base_url,
                model_name=settings.generation.model_name,
                timeout_seconds=settings.generation.timeout_seconds,
                temperature=settings.generation.temperature,
                max_tokens=settings.generation.max_tokens,
                top_p=settings.generation.top_p,
                keep_alive=settings.generation.keep_alive,
            )
            if preflight.require_ollama_healthcheck:
                generation_warmup["ollama_healthcheck_ok"] = bool(ollama_client.healthcheck())
            if preflight.require_ollama_generate_warmup:
                health_ok = generation_warmup.get("ollama_healthcheck_ok")
                if health_ok is not False:
                    generation_warmup["ollama_generate_warmup_ok"] = bool(
                        ollama_client.warmup_generate(
                            prompt=preflight.ollama_generate_prompt,
                            max_tokens=preflight.ollama_generate_max_tokens,
                        )
                    )
                else:
                    generation_warmup["ollama_generate_warmup_ok"] = False

        failures: list[str] = []
        if (
            preflight.require_qdrant_healthcheck
            and retrieval_warmup.get("qdrant_healthcheck_ok") is not True
        ):
            failures.append("qdrant_healthcheck")
        if (
            preflight.require_embed_query_warmup
            and retrieval_warmup.get("embed_query_warmup_ok") is not True
        ):
            failures.append("embed_query_warmup")
        if (
            preflight.require_tiny_search_warmup
            and retrieval_warmup.get("tiny_search_warmup_ok") is not True
        ):
            failures.append("tiny_search_warmup")
        if (
            preflight.require_ollama_healthcheck
            and generation_warmup.get("ollama_healthcheck_ok") is not True
        ):
            failures.append("ollama_healthcheck")
        if (
            preflight.require_ollama_generate_warmup
            and generation_warmup.get("ollama_generate_warmup_ok") is not True
        ):
            failures.append("ollama_generate_warmup")

        if failures:
            failed_checks = ", ".join(failures)
            logger.error(
                "Eval preflight failed: failed_checks=%s retrieval=%s generation=%s",
                failed_checks,
                retrieval_warmup,
                generation_warmup,
            )
            raise RuntimeError(
                f"Eval preflight failed: {failed_checks}. Fix local dependencies before running eval."
            )

        logger.info(
            "Eval preflight checks passed: retrieval=%s generation=%s",
            retrieval_warmup,
            generation_warmup,
        )

    @staticmethod
    def _build_preflight_retriever(settings: Settings) -> DenseRetriever:
        embedder = create_embedder(settings)
        vector_store = QdrantStore(
            mode=settings.vector_store.mode,
            local_path=settings.vector_store.local_path,
            host=settings.vector_store.host,
            port=settings.vector_store.port,
            url=settings.vector_store.url,
            api_key=settings.vector_store.api_key,
            timeout=settings.vector_store.timeout,
            prefer_grpc=settings.vector_store.prefer_grpc,
            collection_name=settings.vector_store.collection_name,
            distance=settings.vector_store.distance,
            upsert_batch_size=settings.vector_store.upsert_batch_size,
            wait=settings.vector_store.wait,
            payload_indexes=settings.vector_store.payload_indexes,
        )
        preflight = settings.eval.preflight
        return DenseRetriever(
            embedder=embedder,
            vector_store=vector_store,
            default_top_k=settings.retrieval.dense_top_k,
            warmup_qdrant_healthcheck_enabled=preflight.require_qdrant_healthcheck,
            warmup_embed_query_enabled=preflight.require_embed_query_warmup,
            warmup_embed_query_text=preflight.retrieval_warmup_query,
            warmup_tiny_search_enabled=preflight.require_tiny_search_warmup,
            warmup_tiny_search_top_k=preflight.tiny_search_top_k,
        )

    @staticmethod
    def _try_close_vector_store(retriever: object) -> None:
        close_method: Callable[[], None] | None = getattr(retriever, "close", None)
        if callable(close_method):
            close_method()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EvalService orchestration.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--datasets-dir", type=Path, default=None)
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
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
    parser.add_argument("--judge-backend", type=str, default=None)
    return parser


def main() -> None:
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
            write_trace=None if not args.no_trace else False,
            fail_fast=args.fail_fast or None,
            judge_enabled=args.judge or None,
            judge_backend=args.judge_backend,
        )
    except Exception:
        logger.exception("EvalService failed")
        sys.exit(1)

    logger.info("EvalService finished successfully")


if __name__ == "__main__":
    main()
