"""Query route definitions."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_api_query_service
from app.application.api_query_service import ApiQueryService
from app.schemas.api import ErrorDetail, ErrorResponse
from app.schemas.query import QueryRequest
from app.schemas.response import QueryDebugResponse, QueryResponse, SourceResponse
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["query"])
ApiQueryServiceDep = Annotated[ApiQueryService, Depends(get_api_query_service)]


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def query_endpoint(
    payload: QueryRequest,
    service: ApiQueryServiceDep,
) -> QueryResponse:
    """Execute full online query flow and return answer payload."""
    include_debug = payload.debug is not None
    logger.info("Query request started: question_len=%s", len(payload.question))
    try:
        result = service.query(
            question=payload.question,
            filters=payload.filters.as_dict() if payload.filters is not None else None,
            retrieve_top_k=payload.retrieve_top_k,
            rerank_top_k_before=payload.rerank_top_k_before,
            rerank_top_k_after=payload.rerank_top_k_after,
            context_token_budget=payload.context_token_budget,
            context_max_chunks=payload.context_max_chunks,
            write_trace=payload.write_trace,
            include_debug=include_debug,
        )
    except HTTPException:
        raise
    except Exception as err:
        logger.exception("Query request failed unexpectedly.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error=ErrorDetail(code="internal_error", message="Query execution failed.")
            ).model_dump(mode="json"),
        ) from err

    debug_response: QueryDebugResponse | None = None
    if include_debug and result.debug is not None:
        stage_counts = result.debug.get("stage_counts")
        latency_ms = result.debug.get("latency_ms")
        trace_paths = result.debug.get("trace_paths")
        debug_response = QueryDebugResponse(
            stage_counts=stage_counts if isinstance(stage_counts, dict) else None,
            latency_ms=latency_ms if isinstance(latency_ms, dict) else None,
            trace_paths=trace_paths if isinstance(trace_paths, dict) else None,
            context=None,
        )

    response = QueryResponse(
        query_id=result.run_id,
        answer=result.answer,
        confidence=result.confidence,
        sources=[
            SourceResponse(
                title=source.title,
                url=source.url,
                section=source.section,
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                support_score=source.support_score,
                source_type=source.source_type,
                source_name=source.source_name,
                source_path=source.source_path,
            )
            for source in result.sources
        ],
        debug=debug_response,
    )
    logger.info(
        "Query request finished: run_id=%s confidence=%s sources_total=%s",
        result.run_id,
        result.confidence,
        result.sources_total,
    )
    return response
