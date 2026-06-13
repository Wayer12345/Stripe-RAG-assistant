"""Unit tests for API request/response schemas."""

from __future__ import annotations

import pytest
from app.schemas.query import QueryDebugOptions, QueryFilters, QueryRequest
from app.schemas.response import QueryResponse, SourceResponse
from pydantic import ValidationError


@pytest.mark.unit
def test_query_request_accepts_valid_question() -> None:
    payload = QueryRequest(question="What is 3D Secure 2?")
    assert payload.question == "What is 3D Secure 2?"


@pytest.mark.unit
def test_query_request_rejects_empty_question() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="   ")


@pytest.mark.unit
def test_query_request_validates_positive_top_k() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="hello", retrieve_top_k=0)


@pytest.mark.unit
def test_query_request_rejects_invalid_rerank_bounds() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="hello", rerank_top_k_before=3, rerank_top_k_after=4)


@pytest.mark.unit
def test_query_filters_trim_empty_fields_to_none() -> None:
    filters = QueryFilters(document_id="  ", source_type=" docs ")
    assert filters.document_id is None
    assert filters.source_type == "docs"


@pytest.mark.unit
def test_query_response_serializes_sources() -> None:
    response = QueryResponse(
        query_id="run_1",
        answer="Answer",
        confidence="high",
        sources=[SourceResponse(title="Stripe", chunk_id="chunk_1", document_id="doc_1")],
    )
    serialized = response.model_dump(mode="json")
    assert serialized["sources"][0]["title"] == "Stripe"


@pytest.mark.unit
def test_debug_options_defaults() -> None:
    options = QueryDebugOptions()
    assert options.include_trace is True
    assert options.include_stage_counts is True
    assert options.include_latency is True
    assert options.include_sources is True
    assert options.include_retrieved_preview is False
    assert options.include_context_preview is False
