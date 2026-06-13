"""Cross-encoder reranker core implementation with optional file cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.domain.models.retrieval_result import RetrievalResult
from app.utils.hashing import sha256_text
from app.utils.logging import get_logger
from app.utils.timing import elapsed_ms, now_perf_seconds

logger = get_logger(__name__)


class RerankerCache:
    """Simple local JSON-per-key cache for reranker scores."""

    def __init__(self, *, cache_path: Path | str) -> None:
        self._cache_path = Path(cache_path)
        self._cache_path.mkdir(parents=True, exist_ok=True)

    def _entry_path(self, key: str) -> Path:
        key_hash = sha256_text(key)
        return self._cache_path / f"{key_hash}.json"

    def get(self, key: str) -> float | None:
        entry_path = self._entry_path(key)
        if not entry_path.exists():
            return None
        try:
            payload = json.loads(entry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt cache entry ignored: path=%s", entry_path)
            return None

        score = payload.get("score")
        if not isinstance(score, (int, float)):
            logger.warning("Corrupt cache entry ignored (invalid score): path=%s", entry_path)
            return None
        return float(score)

    def set(self, key: str, score: float, metadata: dict[str, Any] | None = None) -> None:
        entry_path = self._entry_path(key)
        payload: dict[str, Any] = {"score": float(score)}
        if metadata:
            payload["metadata"] = metadata
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        entry_path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")


class CrossEncoderReranker:
    """Reranks retrieval candidates using sentence-transformers CrossEncoder."""

    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int,
        top_k_before: int,
        top_k_after: int,
        max_query_chars: int,
        max_pair_chars: int,
        warmup_enabled: bool,
        cache_enabled: bool,
        cache_path: Path | str | None = None,
        latency_budget_ms: int = 50,
        model: Any | None = None,
    ) -> None:
        self._configured_model_name = model_name.strip()
        if not self._configured_model_name:
            raise ValueError("model_name must not be empty.")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if top_k_before <= 0:
            raise ValueError("top_k_before must be > 0.")
        if top_k_after <= 0:
            raise ValueError("top_k_after must be > 0.")
        if top_k_after > top_k_before:
            raise ValueError("top_k_after must be <= top_k_before.")
        if max_query_chars <= 0:
            raise ValueError("max_query_chars must be > 0.")
        if max_pair_chars <= 0:
            raise ValueError("max_pair_chars must be > 0.")
        if latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be > 0.")

        self._batch_size = batch_size
        self._top_k_before = top_k_before
        self._top_k_after = top_k_after
        self._max_query_chars = max_query_chars
        self._max_pair_chars = max_pair_chars
        self._warmup_enabled = warmup_enabled
        self._cache_enabled = cache_enabled
        self._latency_budget_ms = latency_budget_ms
        self._model = model
        self._warmup_duration_ms: int | None = None
        self._last_stats: dict[str, Any] = {
            "scored_pairs": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "model_scored_pairs": 0,
            "duration_ms": 0,
            "latency_budget_ms": self._latency_budget_ms,
            "latency_budget_exceeded": False,
            "model_name": self._configured_model_name,
            "top_k_before": self._top_k_before,
            "top_k_after": self._top_k_after,
        }
        self._cache: RerankerCache | None = None
        if self._cache_enabled:
            if cache_path is None:
                raise ValueError("cache_path must be provided when cache_enabled is True.")
            self._cache = RerankerCache(cache_path=cache_path)
        else:
            logger.warning("Reranker cache disabled.")

    def model_name(self) -> str:
        """Return configured model name."""
        return self._configured_model_name

    def last_stats(self) -> dict[str, Any]:
        """Return stats from the most recent rerank run."""
        return dict(self._last_stats)

    def warmup(self) -> dict[str, bool | int | str | None]:
        """Warm up model with a tiny dummy prediction."""
        if not self._warmup_enabled:
            return {
                "status": "skipped",
                "reranker_warmup_ok": None,
                "reranker_warmup_duration_ms": self._warmup_duration_ms,
            }

        logger.info("Reranker warmup started: model_name=%s", self._configured_model_name)
        warmup_start = now_perf_seconds()
        _ = self._score_pairs([("warmup query", "warmup text")])
        self._warmup_duration_ms = elapsed_ms(warmup_start)
        logger.info(
            "Reranker warmup finished: model_name=%s duration_ms=%s",
            self._configured_model_name,
            self._warmup_duration_ms,
        )
        return {
            "status": "success",
            "reranker_warmup_ok": True,
            "reranker_warmup_duration_ms": self._warmup_duration_ms,
        }

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        *,
        top_k_before: int | None = None,
        top_k_after: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank retrieval candidates by cross-encoder relevance scores."""
        if not query.strip():
            raise ValueError("query must not be empty.")

        resolved_top_k_before = self._top_k_before if top_k_before is None else top_k_before
        resolved_top_k_after = self._top_k_after if top_k_after is None else top_k_after
        if resolved_top_k_before <= 0:
            raise ValueError("top_k_before must be > 0.")
        if resolved_top_k_after <= 0:
            raise ValueError("top_k_after must be > 0.")
        if resolved_top_k_after > resolved_top_k_before:
            raise ValueError("top_k_after must be <= top_k_before.")
        if not candidates:
            self._last_stats = {
                "scored_pairs": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "model_scored_pairs": 0,
                "duration_ms": 0,
                "latency_budget_ms": self._latency_budget_ms,
                "latency_budget_exceeded": False,
                "model_name": self._configured_model_name,
                "top_k_before": resolved_top_k_before,
                "top_k_after": resolved_top_k_after,
                "warmup_duration_ms": self._warmup_duration_ms,
            }
            return []

        rerank_start = now_perf_seconds()
        if resolved_top_k_before > len(candidates):
            logger.warning(
                "top_k_before larger than candidates: top_k_before=%s candidates=%s",
                resolved_top_k_before,
                len(candidates),
            )

        considered_candidates = candidates[:resolved_top_k_before]
        prepared_query = self._prepare_query(query)
        prepared_pairs = self._prepare_pairs(prepared_query, considered_candidates)
        logger.info(
            "Rerank started: model_name=%s candidates=%s scored_pairs=%s latency_budget_ms=%s",
            self._configured_model_name,
            len(considered_candidates),
            len(prepared_pairs),
            self._latency_budget_ms,
        )

        cache_hits = 0
        cache_misses = 0
        model_scored_pairs = 0
        scores: list[float | None] = [None] * len(considered_candidates)
        miss_pairs: list[tuple[str, str]] = []
        miss_indexes: list[int] = []
        miss_keys: list[str] = []

        for index, (candidate, pair) in enumerate(zip(considered_candidates, prepared_pairs, strict=False)):
            cache_key = self._build_cache_key(prepared_query, candidate.chunk_id, pair[1])
            cached_score = self._cache.get(cache_key) if self._cache is not None else None
            if cached_score is not None:
                scores[index] = cached_score
                cache_hits += 1
                continue
            cache_misses += 1
            miss_pairs.append(pair)
            miss_indexes.append(index)
            miss_keys.append(cache_key)

        if miss_pairs:
            model_scores = self._score_pairs(miss_pairs)
            model_scored_pairs = len(model_scores)
            for idx, score, cache_key in zip(miss_indexes, model_scores, miss_keys, strict=False):
                scores[idx] = score
                if self._cache is not None:
                    self._cache.set(
                        cache_key,
                        score,
                        metadata={
                            "model_name": self._configured_model_name,
                            "max_query_chars": self._max_query_chars,
                            "max_pair_chars": self._max_pair_chars,
                        },
                    )

        typed_scores = [float(score) for score in scores if score is not None]
        if len(typed_scores) != len(considered_candidates):
            raise ValueError(
                "Reranker score count mismatch after cache/model scoring: "
                f"expected={len(considered_candidates)} actual={len(typed_scores)}"
            )

        reranked_with_index: list[tuple[int, RetrievalResult]] = []
        for index, (candidate, score) in enumerate(
            zip(considered_candidates, typed_scores, strict=False), start=1
        ):
            updated = candidate.model_copy(
                update={
                    "reranker_score": float(score),
                    "final_score": float(score),
                    "rank": index,
                }
            )
            reranked_with_index.append((index, updated))

        reranked_with_index.sort(
            key=lambda item: (-item[1].final_score, item[0]),
        )
        reranked = [result for _, result in reranked_with_index[:resolved_top_k_after]]

        duration_ms = elapsed_ms(rerank_start)
        latency_budget_exceeded = duration_ms > self._latency_budget_ms
        if latency_budget_exceeded:
            logger.warning(
                "Rerank latency budget exceeded: duration_ms=%s latency_budget_ms=%s",
                duration_ms,
                self._latency_budget_ms,
            )

        logger.info(
            "Rerank finished: scored_pairs=%s cache_hits=%s cache_misses=%s model_scored_pairs=%s duration_ms=%s latency_budget_ms=%s",
            len(prepared_pairs),
            cache_hits,
            cache_misses,
            model_scored_pairs,
            duration_ms,
            self._latency_budget_ms,
        )
        self._last_stats = {
            "scored_pairs": len(prepared_pairs),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "model_scored_pairs": model_scored_pairs,
            "duration_ms": duration_ms,
            "latency_budget_ms": self._latency_budget_ms,
            "latency_budget_exceeded": latency_budget_exceeded,
            "warmup_duration_ms": self._warmup_duration_ms,
            "model_name": self._configured_model_name,
            "top_k_before": resolved_top_k_before,
            "top_k_after": resolved_top_k_after,
        }
        return reranked

    def _prepare_query(self, query: str) -> str:
        prepared = query.strip()[: self._max_query_chars]
        if not prepared:
            raise ValueError("query must not be empty after normalization.")
        return prepared

    def _prepare_text(self, text: str) -> str:
        return text.strip()[: self._max_pair_chars]

    def _prepare_pairs(
        self, query: str, candidates: list[RetrievalResult]
    ) -> list[tuple[str, str]]:
        return [(query, self._prepare_text(candidate.text)) for candidate in candidates]

    def _build_cache_key(self, query: str, chunk_id: str, prepared_text: str) -> str:
        query_hash = sha256_text(query)
        text_hash = sha256_text(prepared_text if prepared_text else "<EMPTY_TEXT>")
        return (
            f"q_{query_hash}"
            f"__chunk_{chunk_id}"
            f"__t_{text_hash}"
            f"__m_{sha256_text(self._configured_model_name)}"
            f"__mq_{self._max_query_chars}"
            f"__mp_{self._max_pair_chars}"
        )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        logger.info("Lazy loading CrossEncoder model: model_name=%s", self._configured_model_name)
        try:
            from sentence_transformers import CrossEncoder
        except Exception as err:  # pragma: no cover - import error is env-specific
            logger.exception("Failed to import sentence-transformers CrossEncoder.")
            raise RuntimeError("Failed to import sentence-transformers CrossEncoder.") from err
        try:
            self._model = CrossEncoder(self._configured_model_name)
        except Exception as err:  # pragma: no cover - load error is env-specific
            logger.exception("Failed to load cross-encoder model: model_name=%s", self._configured_model_name)
            raise RuntimeError(
                f"Failed to load cross-encoder model: {self._configured_model_name}"
            ) from err
        return self._model

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        model = self._load_model()
        try:
            predictions: Any
            try:
                predictions = model.predict(pairs, batch_size=self._batch_size)
            except TypeError:
                predictions = model.predict(pairs)
        except Exception as err:
            logger.exception("Cross-encoder scoring failed.")
            raise RuntimeError("Cross-encoder scoring failed.") from err

        raw_scores: list[Any]
        if isinstance(predictions, list):
            raw_scores = predictions
        elif hasattr(predictions, "tolist"):
            raw = predictions.tolist()
            raw_scores = raw if isinstance(raw, list) else [raw]
        else:
            raw_scores = [predictions]

        scores: list[float] = [float(score) for score in raw_scores]
        if len(scores) != len(pairs):
            logger.error(
                "Invalid score count from reranker model: expected=%s actual=%s",
                len(pairs),
                len(scores),
            )
            raise ValueError(
                "Invalid score count from reranker model: "
                f"expected={len(pairs)} actual={len(scores)}"
            )
        return scores
