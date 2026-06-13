"""Unit tests for reranking infrastructure."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from app.domain.models.retrieval_result import RetrievalResult
from app.domain.models.source import Source
from app.infrastructure.reranking.cross_encoder_reranker import CrossEncoderReranker, RerankerCache
from app.infrastructure.reranking.reranker_factory import create_reranker


class FakeCrossEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[tuple[str, str]], int | None]] = []

    def predict(
        self, pairs: list[tuple[str, str]], batch_size: int | None = None
    ) -> list[float]:
        self.calls.append((pairs, batch_size))
        return [float(len(text)) for _, text in pairs]


def _candidate(
    *,
    index: int,
    text: str,
    dense_score: float = 0.9,
    retrieval_score: float = 0.8,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"chunk-{index}",
        document_id=f"doc-{index}",
        text=text,
        source=Source(
            title=f"title-{index}",
            url=f"https://docs.example/{index}",
            section="section",
            chunk_id=f"chunk-{index}",
            document_id=f"doc-{index}",
            support_score=0.9,
            source_type="markdown",
        ),
        retrieval_score=retrieval_score,
        lexical_score=None,
        dense_score=dense_score,
        reranker_score=None,
        final_score=retrieval_score,
        metadata={"k": f"v-{index}"},
    )


def _reranker(
    tmp_path: Path,
    *,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k_before: int = 5,
    top_k_after: int = 3,
    cache_enabled: bool = True,
) -> tuple[CrossEncoderReranker, FakeCrossEncoder]:
    fake_model = FakeCrossEncoder()
    reranker = CrossEncoderReranker(
        model_name=model_name,
        batch_size=4,
        top_k_before=top_k_before,
        top_k_after=top_k_after,
        max_query_chars=10,
        max_pair_chars=12,
        warmup_enabled=True,
        cache_enabled=cache_enabled,
        cache_path=tmp_path / "cache",
        latency_budget_ms=50,
        model=fake_model,
    )
    return reranker, fake_model


@pytest.mark.unit
def test_injected_model_is_used_without_real_load(tmp_path: Path) -> None:
    reranker, fake_model = _reranker(tmp_path)
    reranker.rerank("query", [_candidate(index=1, text="abc")])
    assert len(fake_model.calls) == 1


@pytest.mark.unit
def test_empty_candidates_return_empty_list(tmp_path: Path) -> None:
    reranker, _ = _reranker(tmp_path)
    assert reranker.rerank("query", []) == []


@pytest.mark.unit
def test_empty_query_raises_clear_error(tmp_path: Path) -> None:
    reranker, _ = _reranker(tmp_path)
    with pytest.raises(ValueError, match="query must not be empty"):
        reranker.rerank("   ", [_candidate(index=1, text="abc")])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("top_k_before", "top_k_after"),
    [(0, 1), (1, 0), (2, 3)],
)
def test_invalid_top_k_values_raise(tmp_path: Path, top_k_before: int, top_k_after: int) -> None:
    reranker, _ = _reranker(tmp_path)
    with pytest.raises(ValueError):
        reranker.rerank(
            "query",
            [_candidate(index=1, text="abc")],
            top_k_before=top_k_before,
            top_k_after=top_k_after,
        )


@pytest.mark.unit
def test_applies_top_k_before_before_scoring(tmp_path: Path) -> None:
    reranker, fake_model = _reranker(tmp_path, top_k_before=2, top_k_after=2)
    reranker.rerank("query", [_candidate(index=1, text="a"), _candidate(index=2, text="bb"), _candidate(index=3, text="ccc")])
    assert len(fake_model.calls[0][0]) == 2


@pytest.mark.unit
def test_returns_at_most_top_k_after(tmp_path: Path) -> None:
    reranker, _ = _reranker(tmp_path, top_k_before=5, top_k_after=2)
    output = reranker.rerank(
        "query",
        [
            _candidate(index=1, text="a"),
            _candidate(index=2, text="bbbb"),
            _candidate(index=3, text="ccc"),
        ],
    )
    assert len(output) == 2


@pytest.mark.unit
def test_model_called_in_batch_not_per_candidate(tmp_path: Path) -> None:
    reranker, fake_model = _reranker(tmp_path, top_k_before=4, top_k_after=4)
    reranker.rerank("query", [_candidate(index=1, text="a"), _candidate(index=2, text="bb")])
    assert len(fake_model.calls) == 1
    assert len(fake_model.calls[0][0]) == 2


@pytest.mark.unit
def test_passes_batch_size_to_model(tmp_path: Path) -> None:
    reranker, fake_model = _reranker(tmp_path)
    reranker.rerank("query", [_candidate(index=1, text="abc")])
    assert fake_model.calls[0][1] == 4


@pytest.mark.unit
def test_truncates_query_and_candidate_text(tmp_path: Path) -> None:
    reranker, fake_model = _reranker(tmp_path)
    reranker.rerank(
        "query that is long",
        [_candidate(index=1, text="text that is definitely longer than max")],
    )
    pair = fake_model.calls[0][0][0]
    assert pair[0] == "query that"
    assert len(pair[1]) == 12


@pytest.mark.unit
def test_sets_reranker_and_final_scores_and_preserves_fields(tmp_path: Path) -> None:
    reranker, _ = _reranker(tmp_path, top_k_before=3, top_k_after=3)
    candidate = _candidate(index=1, text="abcdef", dense_score=0.123, retrieval_score=0.456)
    output = reranker.rerank("query", [candidate])[0]
    assert output.reranker_score is not None
    assert output.final_score == output.reranker_score
    assert output.retrieval_score == pytest.approx(0.456)
    assert output.dense_score == pytest.approx(0.123)
    assert output.source.title == candidate.source.title
    assert output.metadata == candidate.metadata


@pytest.mark.unit
def test_sorts_by_score_desc_and_preserves_input_tie_order(tmp_path: Path) -> None:
    reranker, _ = _reranker(tmp_path, top_k_before=3, top_k_after=3)
    # scores by text length -> second and third tie
    output = reranker.rerank(
        "query",
        [
            _candidate(index=1, text="aa"),
            _candidate(index=2, text="bbb"),
            _candidate(index=3, text="ccc"),
        ],
    )
    assert [item.chunk_id for item in output] == ["chunk-2", "chunk-3", "chunk-1"]


@pytest.mark.unit
def test_last_stats_contains_expected_fields(tmp_path: Path) -> None:
    reranker, _ = _reranker(tmp_path)
    reranker.rerank("query", [_candidate(index=1, text="abc")])
    stats = reranker.last_stats()
    assert "cache_hits" in stats
    assert "cache_misses" in stats
    assert "duration_ms" in stats
    assert "latency_budget_ms" in stats
    assert "latency_budget_exceeded" in stats


@pytest.mark.unit
def test_warmup_calls_model_once(tmp_path: Path) -> None:
    reranker, fake_model = _reranker(tmp_path)
    reranker.warmup()
    assert len(fake_model.calls) == 1
    assert len(fake_model.calls[0][0]) == 1


@pytest.mark.unit
def test_cache_hit_avoids_model_scoring(tmp_path: Path) -> None:
    reranker, fake_model = _reranker(tmp_path)
    candidates = [_candidate(index=1, text="cache me")]
    reranker.rerank("query", candidates)
    first_call_count = len(fake_model.calls)
    reranker.rerank("query", candidates)
    assert len(fake_model.calls) == first_call_count
    assert reranker.last_stats()["cache_hits"] >= 1


@pytest.mark.unit
def test_cache_miss_triggers_model_scoring(tmp_path: Path) -> None:
    reranker, fake_model = _reranker(tmp_path)
    reranker.rerank("query", [_candidate(index=1, text="cache miss")])
    assert len(fake_model.calls) == 1
    assert reranker.last_stats()["cache_misses"] >= 1


@pytest.mark.unit
def test_cache_key_changes_for_query_text_and_model(tmp_path: Path) -> None:
    reranker_a, _ = _reranker(tmp_path, model_name="model-a")
    reranker_b, _ = _reranker(tmp_path, model_name="model-b")
    key_a = reranker_a._build_cache_key("query-a", "chunk-1", "text")
    key_b_query = reranker_a._build_cache_key("query-b", "chunk-1", "text")
    key_b_model = reranker_b._build_cache_key("query-a", "chunk-1", "text")
    assert key_a != key_b_query
    assert key_a != key_b_model


@pytest.mark.unit
def test_cache_key_changes_for_chunk_text(tmp_path: Path) -> None:
    reranker, _ = _reranker(tmp_path)
    key_a = reranker._build_cache_key("query", "chunk-1", "text-a")
    key_b = reranker._build_cache_key("query", "chunk-1", "text-b")
    assert key_a != key_b


@pytest.mark.unit
def test_corrupt_cache_entry_is_handled_as_miss(tmp_path: Path) -> None:
    cache = RerankerCache(cache_path=tmp_path / "cache")
    key = "broken-entry"
    cache._entry_path(key).write_text("{not-json", encoding="utf-8")
    assert cache.get(key) is None


@pytest.mark.unit
def test_cache_uses_short_hashed_filename_for_long_keys(tmp_path: Path) -> None:
    cache = RerankerCache(cache_path=tmp_path / "cache")
    long_key = "x" * 2000
    cache.set(long_key, 0.42)
    assert cache.get(long_key) == pytest.approx(0.42)
    entry_path = cache._entry_path(long_key)
    assert len(entry_path.name) < 255


@pytest.mark.unit
def test_factory_creates_cross_encoder_reranker_with_config_values(tmp_path: Path) -> None:
    settings = SimpleNamespace(
        reranking=SimpleNamespace(
            enabled=True,
            provider="cross_encoder",
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k_before=12,
            top_k_after=8,
            batch_size=10,
            max_query_chars=400,
            max_pair_chars=1200,
            warmup_enabled=True,
            cache_enabled=True,
            cache_path=tmp_path / "cache",
            latency_budget_ms=50,
        )
    )
    reranker = create_reranker(settings)  # type: ignore[arg-type]
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_name() == "cross-encoder/ms-marco-MiniLM-L-6-v2"


@pytest.mark.unit
def test_factory_raises_for_unsupported_provider(tmp_path: Path) -> None:
    settings = SimpleNamespace(
        reranking=SimpleNamespace(
            enabled=True,
            provider="unsupported",
            model_name="x",
            top_k_before=12,
            top_k_after=8,
            batch_size=10,
            max_query_chars=400,
            max_pair_chars=1200,
            warmup_enabled=True,
            cache_enabled=True,
            cache_path=tmp_path / "cache",
            latency_budget_ms=50,
        )
    )
    with pytest.raises(ValueError, match="Unsupported reranker provider"):
        create_reranker(settings)  # type: ignore[arg-type]


@pytest.mark.unit
def test_factory_validates_invalid_top_k(tmp_path: Path) -> None:
    settings = SimpleNamespace(
        reranking=SimpleNamespace(
            enabled=True,
            provider="cross_encoder",
            model_name="x",
            top_k_before=4,
            top_k_after=10,
            batch_size=10,
            max_query_chars=400,
            max_pair_chars=1200,
            warmup_enabled=True,
            cache_enabled=True,
            cache_path=tmp_path / "cache",
            latency_budget_ms=50,
        )
    )
    with pytest.raises(ValueError, match="top_k_after must be <="):
        create_reranker(settings)  # type: ignore[arg-type]


@pytest.mark.unit
def test_factory_validates_invalid_batch_size(tmp_path: Path) -> None:
    settings = SimpleNamespace(
        reranking=SimpleNamespace(
            enabled=True,
            provider="cross_encoder",
            model_name="x",
            top_k_before=12,
            top_k_after=8,
            batch_size=0,
            max_query_chars=400,
            max_pair_chars=1200,
            warmup_enabled=True,
            cache_enabled=True,
            cache_path=tmp_path / "cache",
            latency_budget_ms=50,
        )
    )
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        create_reranker(settings)  # type: ignore[arg-type]


@pytest.mark.unit
def test_factory_validates_invalid_latency_budget(tmp_path: Path) -> None:
    settings = SimpleNamespace(
        reranking=SimpleNamespace(
            enabled=True,
            provider="cross_encoder",
            model_name="x",
            top_k_before=12,
            top_k_after=8,
            batch_size=10,
            max_query_chars=400,
            max_pair_chars=1200,
            warmup_enabled=True,
            cache_enabled=True,
            cache_path=tmp_path / "cache",
            latency_budget_ms=0,
        )
    )
    with pytest.raises(ValueError, match="latency_budget_ms must be > 0"):
        create_reranker(settings)  # type: ignore[arg-type]
