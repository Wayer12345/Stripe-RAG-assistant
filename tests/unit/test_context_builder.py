"""Unit tests for context formatting and packing infrastructure."""

from __future__ import annotations

from app.domain.models.context import ContextBundle
from app.domain.models.retrieval_result import RetrievalResult
from app.domain.models.source import Source
from app.infrastructure.context.context_builder import ContextBuilder
from app.infrastructure.context.context_formatter import ContextFormatter
import pytest


def _result(
    idx: int,
    *,
    text: str,
    chunk_id: str | None = None,
    document_id: str | None = None,
    url: str | None = None,
    use_default_url: bool = True,
    title: str | None = None,
    section: str | None = "section",
    final_score: float = 0.8,
    reranker_score: float | None = 0.7,
    retrieval_score: float | None = 0.6,
    metadata: dict[str, object] | None = None,
) -> RetrievalResult:
    resolved_chunk_id = chunk_id or f"chunk-{idx}"
    resolved_document_id = document_id or f"doc-{idx}"
    resolved_title = title or f"title-{idx}"
    source_url = f"https://docs.example/{idx}" if use_default_url and url is None else url
    return RetrievalResult(
        chunk_id=resolved_chunk_id,
        document_id=resolved_document_id,
        text=text,
        source=Source(
            title=resolved_title,
            url=source_url,
            section=section,
            chunk_id=resolved_chunk_id,
            document_id=resolved_document_id,
            support_score=0.8,
            source_type="markdown",
            heading_path=["Payments", "Cards"],
        ),
        retrieval_score=retrieval_score,
        lexical_score=None,
        dense_score=0.5,
        reranker_score=reranker_score,
        final_score=final_score,
        metadata=metadata or {"category": "payments", "vector": [0.1, 0.2], "token_count": 42},
    )


def _builder(**overrides: object) -> ContextBuilder:
    options: dict[str, object] = {
        "token_budget": 120,
        "max_chunks": 8,
        "max_sources": 5,
        "min_chunk_tokens": 1,
        "max_chunk_tokens": 60,
        "truncate_long_chunks": True,
        "deduplicate_by": ["chunk_id", "text_hash", "url"],
        "include_scores": True,
        "include_metadata": True,
        "context_format_version": "context_v1",
    }
    options.update(overrides)
    return ContextBuilder(**options)


@pytest.mark.unit
def test_formatter_formats_source_block_with_required_fields() -> None:
    formatter = ContextFormatter()
    block = formatter.format_source_block(
        index=1,
        result=_result(1, text="hello world"),
        text="hello world",
        include_scores=True,
        include_metadata=False,
    )
    assert "[Source 1]" in block
    assert "Title:" in block
    assert "URL:" in block
    assert "Section:" in block
    assert "Chunk ID: chunk-1" in block
    assert "Document ID: doc-1" in block


@pytest.mark.unit
def test_formatter_omits_missing_optional_fields_consistently() -> None:
    formatter = ContextFormatter()
    item = _result(1, text="hello", url=None, section=None, use_default_url=False)
    block = formatter.format_source_block(
        index=1,
        result=item,
        text=item.text,
        include_scores=False,
        include_metadata=False,
    )
    assert "URL:" not in block
    assert "Section:" not in block


@pytest.mark.unit
def test_formatter_includes_score_when_enabled() -> None:
    formatter = ContextFormatter()
    block = formatter.format_source_block(
        index=1,
        result=_result(1, text="x"),
        text="x",
        include_scores=True,
        include_metadata=False,
    )
    assert "Score:" in block


@pytest.mark.unit
def test_formatter_excludes_score_when_disabled() -> None:
    formatter = ContextFormatter()
    block = formatter.format_source_block(
        index=1,
        result=_result(1, text="x"),
        text="x",
        include_scores=False,
        include_metadata=False,
    )
    assert "Score:" not in block


@pytest.mark.unit
def test_formatter_includes_only_safe_metadata_when_enabled() -> None:
    formatter = ContextFormatter()
    block = formatter.format_source_block(
        index=1,
        result=_result(1, text="x"),
        text="x",
        include_scores=False,
        include_metadata=True,
    )
    assert "Metadata source_type:" in block
    assert "Metadata category:" in block
    assert "Metadata token_count:" in block
    assert "vector" not in block


@pytest.mark.unit
def test_empty_query_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="query must not be empty"):
        _builder().build(query=" ", results=[_result(1, text="text")])


@pytest.mark.unit
def test_empty_results_returns_valid_empty_context_bundle() -> None:
    bundle = _builder().build(query="question", results=[])
    assert isinstance(bundle, ContextBundle)
    assert bundle.rendered_context == ""
    assert bundle.token_count == 0
    assert bundle.chunks == []
    assert bundle.sources == []


@pytest.mark.unit
def test_one_result_produces_one_source_block() -> None:
    bundle = _builder().build(query="question", results=[_result(1, text="single block")])
    assert len(bundle.chunks) == 1
    assert "[Source 1]" in bundle.rendered_context


@pytest.mark.unit
def test_multiple_results_preserve_relevance_order() -> None:
    items = [_result(1, text="first"), _result(2, text="second"), _result(3, text="third")]
    bundle = _builder().build(query="question", results=items)
    assert [item.chunk_id for item in bundle.chunks] == ["chunk-1", "chunk-2", "chunk-3"]


@pytest.mark.unit
def test_duplicate_chunk_id_is_removed() -> None:
    items = [_result(1, text="a", chunk_id="dup"), _result(2, text="b", chunk_id="dup")]
    bundle = _builder().build(query="question", results=items)
    assert len(bundle.chunks) == 1
    assert bundle.metadata["deduplicated_chunks_total"] == 1


@pytest.mark.unit
def test_duplicate_text_hash_is_removed() -> None:
    items = [_result(1, text="same"), _result(2, text="same")]
    bundle = _builder().build(query="question", results=items)
    assert len(bundle.chunks) == 1
    assert "chunk-2" in bundle.dropped_chunk_ids


@pytest.mark.unit
def test_duplicate_url_with_same_text_is_removed() -> None:
    items = [
        _result(1, text="same text", url="https://docs.example/shared"),
        _result(2, text="same text", url="https://docs.example/shared"),
    ]
    bundle = _builder(deduplicate_by=["chunk_id", "url"]).build(query="question", results=items)
    assert len(bundle.chunks) == 1
    assert "chunk-2" in bundle.dropped_chunk_ids


@pytest.mark.unit
def test_token_budget_is_respected() -> None:
    items = [_result(1, text=" ".join(["a"] * 40)), _result(2, text=" ".join(["b"] * 40))]
    bundle = _builder(token_budget=55, max_chunk_tokens=55).build(query="question", results=items)
    assert bundle.token_count <= 55


@pytest.mark.unit
def test_max_chunks_is_respected() -> None:
    items = [_result(1, text="a"), _result(2, text="b"), _result(3, text="c")]
    bundle = _builder(max_chunks=2).build(query="question", results=items)
    assert len(bundle.chunks) == 2
    assert "chunk-3" in bundle.dropped_chunk_ids


@pytest.mark.unit
def test_dropped_chunk_ids_recorded_and_truncated_when_budget_exceeded() -> None:
    items = [_result(1, text=" ".join(["a"] * 30)), _result(2, text=" ".join(["b"] * 30))]
    bundle = _builder(token_budget=45, max_chunk_tokens=45).build(query="question", results=items)
    assert bundle.truncated is True
    assert bundle.dropped_chunk_ids


@pytest.mark.unit
def test_long_chunk_is_truncated_when_enabled() -> None:
    item = _result(1, text=" ".join(["token"] * 120))
    bundle = _builder(token_budget=120, max_chunk_tokens=20).build(query="question", results=[item])
    assert bundle.chunks
    assert len(bundle.chunks[0].text.split()) <= 20
    assert bundle.chunks[0].metadata.get("chunk_truncated") is True


@pytest.mark.unit
def test_source_list_is_deduplicated_and_respects_max_sources() -> None:
    items = [
        _result(1, text="a", url="https://docs.example/shared"),
        _result(2, text="b", url="https://docs.example/shared"),
        _result(3, text="c", url="https://docs.example/unique"),
    ]
    bundle = _builder(max_sources=1, deduplicate_by=["chunk_id"]).build(query="question", results=items)
    assert len(bundle.sources) == 1


@pytest.mark.unit
def test_context_bundle_metadata_includes_expected_stats() -> None:
    bundle = _builder().build(query="question", results=[_result(1, text="x"), _result(2, text="x")])
    assert bundle.metadata["builder_name"] == "ContextBuilder"
    assert bundle.metadata["input_results_total"] == 2
    assert bundle.metadata["included_chunks_total"] == len(bundle.chunks)
    assert bundle.metadata["dropped_chunks_total"] == len(bundle.dropped_chunk_ids)
    assert bundle.metadata["format_version"] == "context_v1"


@pytest.mark.unit
def test_rendered_context_contains_source_markers() -> None:
    bundle = _builder().build(query="question", results=[_result(1, text="x"), _result(2, text="y")])
    assert "[Source 1]" in bundle.rendered_context
    assert "[Source 2]" in bundle.rendered_context


@pytest.mark.unit
def test_token_count_is_deterministic() -> None:
    builder = _builder()
    inputs = [_result(1, text="hello world"), _result(2, text="again and again")]
    first = builder.build(query="question", results=inputs)
    second = builder.build(query="question", results=inputs)
    assert first.token_count == second.token_count
    assert first.rendered_context == second.rendered_context
