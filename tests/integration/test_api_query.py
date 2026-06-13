"""Integration tests for query API route."""

from __future__ import annotations

from typing import Any

import pytest
from app.api.dependencies import get_api_query_service
from app.api.routes_query import router as query_router
from app.domain.models.source import Source
from fastapi import FastAPI
from fastapi.testclient import TestClient


class _FakeService:
    def __init__(self) -> None:
        self.calls = 0

    def query(self, **kwargs: Any) -> Any:
        self.calls += 1
        return type(
            "Result",
            (),
            {
                "run_id": "run_1",
                "answer": "Answer",
                "confidence": "high",
                "sources_total": 1,
                "sources": [Source(title="Stripe", chunk_id="chunk_1", document_id="doc_1")],
                "debug": {
                    "stage_counts": {"retrieve_results_total": 1, "context_truncated": False},
                    "latency_ms": {"total": 10},
                    "trace_paths": {"api_query": "trace.json"},
                },
            },
        )()


@pytest.mark.integration
def test_post_query_returns_200_for_valid_request() -> None:
    app = FastAPI()
    app.include_router(query_router)
    service = _FakeService()
    app.dependency_overrides[get_api_query_service] = lambda: service
    client = TestClient(app)

    response = client.post("/query", json={"question": "What is 3DS?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Answer"
    assert payload["confidence"] == "high"
    assert len(payload["sources"]) == 1


@pytest.mark.integration
def test_debug_fields_included_when_requested() -> None:
    app = FastAPI()
    app.include_router(query_router)
    app.dependency_overrides[get_api_query_service] = lambda: _FakeService()
    client = TestClient(app)

    response = client.post("/query", json={"question": "Hello", "debug": {"include_trace": True}})
    assert response.status_code == 200
    assert response.json()["debug"]["trace_paths"]["api_query"] == "trace.json"


@pytest.mark.integration
def test_empty_question_returns_422() -> None:
    app = FastAPI()
    app.include_router(query_router)
    app.dependency_overrides[get_api_query_service] = lambda: _FakeService()
    client = TestClient(app)

    response = client.post("/query", json={"question": " "})
    assert response.status_code == 422


@pytest.mark.integration
def test_service_unavailable_returns_503() -> None:
    app = FastAPI()
    app.include_router(query_router)
    client = TestClient(app)

    response = client.post("/query", json={"question": "Hello"})
    assert response.status_code == 503


@pytest.mark.integration
def test_service_failure_returns_500() -> None:
    app = FastAPI()
    app.include_router(query_router)

    class _FailService:
        def query(self, **kwargs: Any) -> Any:
            _ = kwargs
            raise RuntimeError("boom")

    app.dependency_overrides[get_api_query_service] = lambda: _FailService()
    client = TestClient(app)

    response = client.post("/query", json={"question": "Hello"})
    assert response.status_code == 500


@pytest.mark.integration
def test_route_uses_injected_service_instance() -> None:
    app = FastAPI()
    app.include_router(query_router)
    service = _FakeService()
    app.dependency_overrides[get_api_query_service] = lambda: service
    client = TestClient(app)

    client.post("/query", json={"question": "Hello"})
    client.post("/query", json={"question": "Hello again"})
    assert service.calls == 2
