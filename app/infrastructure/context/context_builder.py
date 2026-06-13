"""Core context packing logic for online answer generation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from app.domain.models.context import ContextBundle
from app.domain.models.retrieval_result import RetrievalResult
from app.domain.models.source import Source
from app.infrastructure.context.context_formatter import ContextFormatter
from app.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_DEDUPLICATE_BY = ["chunk_id", "text_hash", "url"]
_DEFAULT_CONTEXT_FORMAT_VERSION = "context_v1"


@dataclass(frozen=True)
class _PackOutcome:
    included_results: list[RetrievalResult]
    dropped_chunk_ids: list[str]
    token_count: int
    truncated: bool


class ContextBuilder:
    """Builds deterministic `ContextBundle` from reranked retrieval results."""

    def __init__(
        self,
        *,
        token_budget: int,
        max_chunks: int,
        max_sources: int,
        min_chunk_tokens: int,
        deduplicate_by: list[str],
        include_scores: bool,
        include_metadata: bool,
        context_format_version: str,
        truncate_long_chunks: bool = True,
        max_chunk_tokens: int = 700,
        formatter: ContextFormatter | None = None,
    ) -> None:
        if token_budget <= 0:
            raise ValueError("token_budget must be > 0.")
        if max_chunks <= 0:
            raise ValueError("max_chunks must be > 0.")
        if max_sources <= 0:
            raise ValueError("max_sources must be > 0.")
        if min_chunk_tokens <= 0:
            raise ValueError("min_chunk_tokens must be > 0.")
        if max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be > 0.")
        if not context_format_version.strip():
            raise ValueError("context_format_version must not be empty.")

        normalized_deduplicate_by = [key.strip().lower() for key in deduplicate_by if key.strip()]
        if not normalized_deduplicate_by:
            normalized_deduplicate_by = list(_DEFAULT_DEDUPLICATE_BY)

        self._token_budget = token_budget
        self._max_chunks = max_chunks
        self._max_sources = max_sources
        self._min_chunk_tokens = min_chunk_tokens
        self._max_chunk_tokens = max_chunk_tokens
        self._deduplicate_by = normalized_deduplicate_by
        self._include_scores = include_scores
        self._include_metadata = include_metadata
        self._context_format_version = context_format_version.strip()
        self._truncate_long_chunks = truncate_long_chunks
        self._formatter = formatter or ContextFormatter()

    def warmup(self) -> dict[str, bool | str]:
        """Warm lightweight context components and formatter path."""
        _ = self._formatter.format(
            query="warmup query",
            results=[],
            include_scores=self._include_scores,
            include_metadata=self._include_metadata,
        )
        return {"status": "success", "context_builder_warmup_ok": True}

    def build(
        self,
        *,
        query: str,
        results: list[RetrievalResult],
        token_budget: int | None = None,
        max_chunks: int | None = None,
    ) -> ContextBundle:
        """Build a bounded context bundle while preserving relevance order."""
        if not query.strip():
            logger.error("Context build failed: invalid query.")
            raise ValueError("query must not be empty.")

        resolved_token_budget = token_budget if token_budget is not None else self._token_budget
        resolved_max_chunks = max_chunks if max_chunks is not None else self._max_chunks
        if resolved_token_budget <= 0:
            raise ValueError("token_budget must be > 0.")
        if resolved_max_chunks <= 0:
            raise ValueError("max_chunks must be > 0.")

        logger.info(
            "Starting context build: input_results=%s token_budget=%s max_chunks=%s",
            len(results),
            resolved_token_budget,
            resolved_max_chunks,
        )

        if not results:
            logger.warning("Context builder received zero input results.")
            return self._empty_bundle(
                query=query.strip(),
                token_budget=resolved_token_budget,
                input_results_total=0,
                dropped_chunk_ids=[],
                deduplicated_chunks_total=0,
            )

        kept_results, deduplicated_drop_ids = self._deduplicate_results(results)
        if not kept_results:
            logger.warning("Context builder produced no candidates after deduplication.")
            return self._empty_bundle(
                query=query.strip(),
                token_budget=resolved_token_budget,
                input_results_total=len(results),
                dropped_chunk_ids=deduplicated_drop_ids,
                deduplicated_chunks_total=len(deduplicated_drop_ids),
            )

        outcome = self._pack_results(
            query=query.strip(),
            candidates=kept_results,
            token_budget=resolved_token_budget,
            max_chunks=resolved_max_chunks,
        )
        dropped_chunk_ids = deduplicated_drop_ids + outcome.dropped_chunk_ids
        included_chunk_ids = {item.chunk_id for item in outcome.included_results}
        dropped_chunk_ids = [
            chunk_id for chunk_id in dropped_chunk_ids if chunk_id not in included_chunk_ids
        ]
        rendered_context = self._formatter.format(
            query=query.strip(),
            results=outcome.included_results,
            include_scores=self._include_scores,
            include_metadata=self._include_metadata,
        )
        token_count = self._token_count(rendered_context)
        sources = self._build_sources(outcome.included_results)

        if not outcome.included_results:
            logger.warning("Empty context produced after budget packing.")

        metadata = {
            "builder_name": self.__class__.__name__,
            "input_results_total": len(results),
            "included_chunks_total": len(outcome.included_results),
            "dropped_chunks_total": len(dropped_chunk_ids),
            "deduplicated_chunks_total": len(deduplicated_drop_ids),
            "max_chunks": resolved_max_chunks,
            "max_sources": self._max_sources,
            "deduplicate_by": list(self._deduplicate_by),
            "include_scores": self._include_scores,
            "include_metadata": self._include_metadata,
            "format_version": self._context_format_version,
        }

        logger.info(
            "Finished context build: deduped=%s included=%s dropped=%s token_count=%s truncated=%s",
            len(deduplicated_drop_ids),
            len(outcome.included_results),
            len(dropped_chunk_ids),
            token_count,
            outcome.truncated,
        )
        return ContextBundle(
            query=query.strip(),
            chunks=outcome.included_results,
            rendered_context=rendered_context,
            token_count=token_count,
            sources=sources,
            token_budget=resolved_token_budget,
            truncated=outcome.truncated,
            dropped_chunk_ids=dropped_chunk_ids,
            context_format_version=self._context_format_version,
            metadata=metadata,
        )

    def _pack_results(
        self,
        *,
        query: str,
        candidates: list[RetrievalResult],
        token_budget: int,
        max_chunks: int,
    ) -> _PackOutcome:
        included_results: list[RetrievalResult] = []
        dropped_chunk_ids: list[str] = []
        token_count = 0
        truncated = False

        for result in candidates:
            if len(included_results) >= max_chunks:
                dropped_chunk_ids.append(result.chunk_id)
                truncated = True
                continue

            prepared = self._prepare_candidate_for_budget(
                result, remaining_tokens=token_budget - token_count
            )
            if prepared is None:
                dropped_chunk_ids.append(result.chunk_id)
                truncated = True
                continue

            block = self._formatter.format_source_block(
                index=len(included_results) + 1,
                result=prepared,
                text=prepared.text,
                include_scores=self._include_scores,
                include_metadata=self._include_metadata,
            )
            if not block.strip():
                dropped_chunk_ids.append(result.chunk_id)
                continue

            block_tokens = self._token_count(block)
            if block_tokens <= token_budget - token_count:
                included_results.append(prepared)
                token_count += block_tokens
                continue

            remaining = token_budget - token_count
            if remaining <= 0:
                logger.warning("Token budget reached while packing context.")
                dropped_chunk_ids.append(result.chunk_id)
                truncated = True
                continue

            truncated_candidate = self._truncate_candidate_for_remaining(
                prepared,
                remaining_tokens=remaining,
                query=query,
                source_index=len(included_results) + 1,
            )
            if truncated_candidate is None:
                if not included_results:
                    logger.warning(
                        "Top chunk too large for context budget; dropping chunk_id=%s",
                        result.chunk_id,
                    )
                dropped_chunk_ids.append(result.chunk_id)
                truncated = True
                continue

            truncated_block = self._formatter.format_source_block(
                index=len(included_results) + 1,
                result=truncated_candidate,
                text=truncated_candidate.text,
                include_scores=self._include_scores,
                include_metadata=self._include_metadata,
            )
            truncated_tokens = self._token_count(truncated_block)
            if truncated_tokens <= remaining:
                logger.warning(
                    "Long chunk truncated to fit token budget: chunk_id=%s", result.chunk_id
                )
                included_results.append(truncated_candidate)
                token_count += truncated_tokens
                truncated = True
                continue

            dropped_chunk_ids.append(result.chunk_id)
            truncated = True

        return _PackOutcome(
            included_results=included_results,
            dropped_chunk_ids=dropped_chunk_ids,
            token_count=token_count,
            truncated=truncated,
        )

    def _prepare_candidate_for_budget(
        self, result: RetrievalResult, *, remaining_tokens: int
    ) -> RetrievalResult | None:
        if remaining_tokens <= 0:
            return None
        text = result.text.strip()
        if not text:
            return None
        token_count = self._token_count(text)
        if token_count < self._min_chunk_tokens:
            return None

        if token_count > self._max_chunk_tokens and self._truncate_long_chunks:
            truncated_text = self._truncate_to_tokens(text, self._max_chunk_tokens)
            metadata = dict(result.metadata)
            metadata["chunk_truncated"] = True
            metadata["chunk_truncated_reason"] = "max_chunk_tokens"
            return result.model_copy(update={"text": truncated_text, "metadata": metadata})

        if token_count > self._max_chunk_tokens and not self._truncate_long_chunks:
            return None

        return result

    def _truncate_candidate_for_remaining(
        self,
        result: RetrievalResult,
        *,
        remaining_tokens: int,
        query: str,
        source_index: int,
    ) -> RetrievalResult | None:
        _ = query
        if not self._truncate_long_chunks:
            return None
        if remaining_tokens < self._min_chunk_tokens:
            return None

        max_text_tokens = min(remaining_tokens, self._max_chunk_tokens)
        if max_text_tokens < self._min_chunk_tokens:
            return None
        text = self._truncate_to_tokens(result.text, max_text_tokens)
        if not text.strip():
            return None

        metadata = dict(result.metadata)
        metadata["chunk_truncated"] = True
        metadata["chunk_truncated_reason"] = "token_budget"
        candidate = result.model_copy(update={"text": text, "metadata": metadata})
        block = self._formatter.format_source_block(
            index=source_index,
            result=candidate,
            text=text,
            include_scores=self._include_scores,
            include_metadata=self._include_metadata,
        )
        if self._token_count(block) > remaining_tokens:
            overhead = self._token_count(
                self._formatter.format_source_block(
                    index=source_index,
                    result=candidate,
                    text="",
                    include_scores=self._include_scores,
                    include_metadata=self._include_metadata,
                )
            )
            allowed_tokens = remaining_tokens - overhead
            if allowed_tokens < self._min_chunk_tokens:
                return None
            adjusted = self._truncate_to_tokens(result.text, allowed_tokens)
            if not adjusted.strip():
                return None
            metadata["chunk_truncated"] = True
            metadata["chunk_truncated_reason"] = "token_budget"
            candidate = result.model_copy(update={"text": adjusted, "metadata": metadata})
        return candidate

    def _build_sources(self, results: list[RetrievalResult]) -> list[Source]:
        sources: list[Source] = []
        seen_url: set[str] = set()
        seen_document_id: set[str] = set()

        for result in results:
            source = result.source
            deduped_key = source.url.strip() if source.url else source.document_id
            if source.url and deduped_key in seen_url:
                continue
            if not source.url and deduped_key in seen_document_id:
                continue

            support_score = self._source_support_score(result)
            source_with_score = source.model_copy(update={"support_score": support_score})
            sources.append(source_with_score)

            if source.url:
                seen_url.add(deduped_key)
            else:
                seen_document_id.add(deduped_key)

            if len(sources) >= self._max_sources:
                break

        if sources and all(not item.url and not item.title for item in sources):
            logger.warning("Sources are missing URL/title fields for all entries.")
        return sources

    def _source_support_score(self, result: RetrievalResult) -> float | None:
        for score in (result.final_score, result.reranker_score, result.retrieval_score):
            if isinstance(score, (int, float)) and 0.0 <= float(score) <= 1.0:
                return float(score)
        return None

    def _deduplicate_results(
        self, results: list[RetrievalResult]
    ) -> tuple[list[RetrievalResult], list[str]]:
        kept: list[RetrievalResult] = []
        dropped_ids: list[str] = []
        seen_chunk_ids: set[str] = set()
        seen_text_hashes: set[str] = set()
        seen_url_text_hash: set[tuple[str, str]] = set()
        seen_document_text_hash: set[tuple[str, str]] = set()

        dedupe_text_hash = "text_hash" in self._deduplicate_by
        dedupe_url = "url" in self._deduplicate_by
        dedupe_document = "document_id" in self._deduplicate_by

        for result in results:
            text_hash = self._text_hash(result.text)
            if result.chunk_id in seen_chunk_ids:
                dropped_ids.append(result.chunk_id)
                continue

            if dedupe_text_hash and text_hash in seen_text_hashes:
                dropped_ids.append(result.chunk_id)
                continue

            if dedupe_url and result.source.url:
                key = (result.source.url.strip(), text_hash)
                if key in seen_url_text_hash:
                    dropped_ids.append(result.chunk_id)
                    continue

            if dedupe_document:
                key = (result.document_id, text_hash)
                if key in seen_document_text_hash:
                    dropped_ids.append(result.chunk_id)
                    continue

            kept.append(result)
            seen_chunk_ids.add(result.chunk_id)
            if dedupe_text_hash:
                seen_text_hashes.add(text_hash)
            if dedupe_url and result.source.url:
                seen_url_text_hash.add((result.source.url.strip(), text_hash))
            if dedupe_document:
                seen_document_text_hash.add((result.document_id, text_hash))

        return kept, dropped_ids

    def _empty_bundle(
        self,
        *,
        query: str,
        token_budget: int,
        input_results_total: int,
        dropped_chunk_ids: list[str],
        deduplicated_chunks_total: int,
    ) -> ContextBundle:
        metadata = {
            "builder_name": self.__class__.__name__,
            "input_results_total": input_results_total,
            "included_chunks_total": 0,
            "dropped_chunks_total": len(dropped_chunk_ids),
            "deduplicated_chunks_total": deduplicated_chunks_total,
            "max_chunks": self._max_chunks,
            "max_sources": self._max_sources,
            "deduplicate_by": list(self._deduplicate_by),
            "include_scores": self._include_scores,
            "include_metadata": self._include_metadata,
            "format_version": self._context_format_version,
        }
        return ContextBundle(
            query=query,
            chunks=[],
            rendered_context="",
            token_count=0,
            sources=[],
            token_budget=token_budget,
            truncated=False,
            dropped_chunk_ids=dropped_chunk_ids,
            context_format_version=self._context_format_version,
            metadata=metadata,
        )

    @staticmethod
    def _truncate_to_tokens(text: str, token_limit: int) -> str:
        if token_limit <= 0:
            return ""
        tokens = text.split()
        return " ".join(tokens[:token_limit]).strip()

    @staticmethod
    def _token_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _text_hash(text: str) -> str:
        normalized = " ".join(text.split()).lower()
        return sha256(normalized.encode("utf-8")).hexdigest()
