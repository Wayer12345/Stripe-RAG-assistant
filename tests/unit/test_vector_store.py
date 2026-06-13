"""Unit tests for Qdrant vector store mapping and validation logic."""

from __future__ import annotations

import uuid

import pytest
from app.domain.models.chunk import Chunk
from app.domain.models.embedded_chunk import EmbeddedChunk
from app.infrastructure.vector_stores.qdrant_collections import map_distance
from app.infrastructure.vector_stores.qdrant_filters import build_qdrant_filter
from app.infrastructure.vector_stores.qdrant_store import QdrantStore
from app.utils.hashing import sha256_text
from qdrant_client import models


def _embedded_chunk_with_optional_none() -> EmbeddedChunk:
    chunk = Chunk(
        id="chunk-test",
        document_id="doc-test",
        text="chunk text",
        chunk_index=0,
        token_count=3,
        content_hash=sha256_text("chunk text"),
        chunking_strategy="semantic",
        heading_path=["H1"],
        section=None,
        metadata={
            "title": "Doc title",
            "url": None,
            "source_path": "docs/test.md",
            "source_name": "test.md",
            "source_type": "markdown",
            "category": "guides",
            "document_content_hash": None,
            "chunker_name": "semantic_chunker",
        },
    )
    return EmbeddedChunk(
        chunk=chunk,
        vector=[0.1, 0.2, 0.3],
        embedding_model="fake-model",
        embedding_dim=3,
        normalized=True,
    )


def test_embedded_chunk_maps_to_qdrant_point() -> None:
    embedded = _embedded_chunk_with_optional_none()
    point = QdrantStore.point_from_embedded_chunk(embedded)

    assert isinstance(point, models.PointStruct)
    assert point.id is not None
    assert point.vector == [0.1, 0.2, 0.3]
    assert point.payload is not None
    assert point.payload["chunk_id"] == "chunk-test"


def test_payload_includes_required_fields_and_omits_none_values() -> None:
    payload = QdrantStore.point_payload_from_embedded_chunk(_embedded_chunk_with_optional_none())

    required_fields = {
        "chunk_id",
        "document_id",
        "text",
        "title",
        "source_path",
        "source_name",
        "source_type",
        "category",
        "token_count",
        "content_hash",
        "embedding_model",
        "embedding_dim",
        "normalized",
    }
    assert required_fields.issubset(set(payload))
    assert "url" not in payload
    assert "document_content_hash" not in payload


def test_distance_mapping_supports_expected_values() -> None:
    assert map_distance("cosine") == models.Distance.COSINE
    assert map_distance("dot") == models.Distance.DOT
    assert map_distance("euclid") == models.Distance.EUCLID
    assert map_distance("manhattan") == models.Distance.MANHATTAN


def test_unsupported_distance_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unsupported distance metric"):
        map_distance("hamming")


def test_app_filters_map_to_qdrant_filters() -> None:
    qdrant_filter = build_qdrant_filter(
        {
            "document_id": "doc-1",
            "category": ["guides", "faq"],
            "token_count": {"gte": 10, "lte": 20},
        }
    )

    assert qdrant_filter is not None
    assert isinstance(qdrant_filter, models.Filter)
    assert len(qdrant_filter.must or []) == 3


def test_deterministic_point_id_generation_is_stable() -> None:
    point_id_1 = QdrantStore._deterministic_point_id("chunk-1")
    point_id_2 = QdrantStore._deterministic_point_id("chunk-1")

    assert point_id_1 == point_id_2
    assert isinstance(uuid.UUID(point_id_1), uuid.UUID)


def test_vector_dimension_validation_catches_bad_vectors() -> None:
    good = _embedded_chunk_with_optional_none()
    bad = EmbeddedChunk.model_construct(
        chunk=good.chunk,
        vector=[0.1, 0.2],
        embedding_model="fake-model",
        embedding_dim=3,
        normalized=True,
    )
    with pytest.raises(ValueError, match="embedding_dim=3"):
        QdrantStore.validate_embedded_chunks([good, bad])

