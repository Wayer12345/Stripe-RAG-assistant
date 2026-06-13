"""Qdrant-backed vector store implementation."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from app.domain.models.embedded_chunk import EmbeddedChunk
from app.domain.models.retrieval_result import RetrievalMethod, RetrievalResult
from app.domain.models.source import Source
from app.infrastructure.vector_stores.qdrant_collections import (
    collection_exists,
    create_collection,
    create_payload_indexes,
    recreate_collection,
    validate_collection_config,
)
from app.infrastructure.vector_stores.qdrant_filters import build_qdrant_filter

_POINT_ID_NAMESPACE = uuid.UUID("ff69fd97-0de7-47ae-85b4-c44ed77eb02f")


class QdrantStore:
    """Qdrant vector store with collection lifecycle + dense upsert/search."""

    def __init__(
        self,
        *,
        collection_name: str,
        mode: str = "embedded",
        local_path: Path | str = Path("data/indexes/qdrant"),
        distance: str = "cosine",
        host: str = "localhost",
        port: int = 6333,
        url: str | None = None,
        api_key: str | None = None,
        timeout: int = 30,
        prefer_grpc: bool = False,
        wait: bool = True,
        upsert_batch_size: int = 64,
        payload_indexes: dict[str, str] | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty.")
        if upsert_batch_size <= 0:
            raise ValueError("upsert_batch_size must be > 0.")

        self.collection_name = collection_name.strip()
        self.mode = mode.strip().lower()
        self.local_path = Path(local_path)
        self.distance = distance.strip().lower()
        self.host = host.strip()
        self.port = port
        self.url = url
        self.timeout = timeout
        self.prefer_grpc = prefer_grpc
        self.wait = wait
        self.upsert_batch_size = upsert_batch_size
        self.payload_indexes = payload_indexes or {}
        if self.mode != "embedded":
            raise ValueError(
                f"QdrantStore supports only embedded mode for this project. Got mode={self.mode!r}."
            )
        self.local_path.mkdir(parents=True, exist_ok=True)
        self._client = client or QdrantClient(path=str(self.local_path))
        self._vector_dim: int | None = None
        self._last_healthcheck_error: str | None = None

    @staticmethod
    def _deterministic_point_id(chunk_id: str) -> str:
        if not chunk_id.strip():
            raise ValueError("chunk_id must not be empty for point ID generation.")
        return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id.strip()))

    @staticmethod
    def _nonnull_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in payload.items() if value is not None}

    @classmethod
    def point_payload_from_embedded_chunk(cls, embedded_chunk: EmbeddedChunk) -> dict[str, Any]:
        """Build a payload dict preserving source and embedding metadata."""
        chunk = embedded_chunk.chunk
        metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        payload = {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "text": chunk.text,
            "title": metadata.get("title"),
            "url": metadata.get("url"),
            "source_path": metadata.get("source_path"),
            "source_name": metadata.get("source_name"),
            "source_type": metadata.get("source_type"),
            "category": metadata.get("category"),
            "section": chunk.section or metadata.get("section"),
            "heading_path": chunk.heading_path,
            "token_count": chunk.token_count,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "content_hash": chunk.content_hash,
            "document_content_hash": metadata.get("document_content_hash"),
            "chunking_strategy": chunk.chunking_strategy or metadata.get("chunking_strategy"),
            "chunker_name": metadata.get("chunker_name"),
            "embedding_model": embedded_chunk.embedding_model,
            "embedding_dim": embedded_chunk.embedding_dim,
            "normalized": embedded_chunk.normalized,
        }
        return cls._nonnull_payload(payload)

    @classmethod
    def point_from_embedded_chunk(cls, embedded_chunk: EmbeddedChunk) -> models.PointStruct:
        """Map an EmbeddedChunk to Qdrant PointStruct."""
        return models.PointStruct(
            id=cls._deterministic_point_id(embedded_chunk.chunk.id),
            vector=[float(value) for value in embedded_chunk.vector],
            payload=cls.point_payload_from_embedded_chunk(embedded_chunk),
        )

    @staticmethod
    def validate_embedded_chunks(embedded_chunks: Sequence[EmbeddedChunk]) -> tuple[int, str, bool]:
        """Validate vector/model consistency and return inferred metadata."""
        if not embedded_chunks:
            raise ValueError("No embedded chunks available for validation.")

        inferred_dim = len(embedded_chunks[0].vector)
        inferred_model = embedded_chunks[0].embedding_model
        inferred_normalized = embedded_chunks[0].normalized

        for embedded in embedded_chunks:
            vector_dim = len(embedded.vector)
            if embedded.embedding_dim != vector_dim:
                raise ValueError(
                    f"Chunk {embedded.chunk.id!r} has embedding_dim={embedded.embedding_dim} "
                    f"but vector length={vector_dim}."
                )
            if vector_dim != inferred_dim:
                raise ValueError(
                    f"Chunk {embedded.chunk.id!r} has vector length={vector_dim}; "
                    f"expected {inferred_dim}."
                )
            if embedded.embedding_model != inferred_model:
                raise ValueError(
                    f"Chunk {embedded.chunk.id!r} uses embedding model "
                    f"{embedded.embedding_model!r}; expected {inferred_model!r}."
                )
            if embedded.normalized != inferred_normalized:
                raise ValueError(
                    f"Chunk {embedded.chunk.id!r} has normalized={embedded.normalized}; "
                    f"expected {inferred_normalized}."
                )

        return inferred_dim, inferred_model, inferred_normalized

    def healthcheck(self) -> bool:
        try:
            self._client.get_collections()
            self._last_healthcheck_error = None
            return True
        except Exception as exc:
            self._last_healthcheck_error = str(exc)
            return False

    def healthcheck_error(self) -> str | None:
        """Return the last healthcheck backend error message, if any."""
        return self._last_healthcheck_error

    def close(self) -> None:
        """Close underlying Qdrant client resources when supported."""
        close_method = getattr(self._client, "close", None)
        if callable(close_method):
            close_method()

    def collection_exists(self, collection_name: str | None = None) -> bool:
        target_name = collection_name or self.collection_name
        return collection_exists(self._client, target_name)

    def create_collection(self, *, recreate: bool = False, vector_dim: int | None = None) -> None:
        target_dim = vector_dim or self._vector_dim
        if target_dim is None:
            raise ValueError("vector_dim must be provided before creating collection.")
        if recreate:
            recreate_collection(
                self._client,
                collection_name=self.collection_name,
                vector_dim=target_dim,
                distance=self.distance,
            )
            return
        if not self.collection_exists():
            create_collection(
                self._client,
                collection_name=self.collection_name,
                vector_dim=target_dim,
                distance=self.distance,
            )

    def recreate_collection(self, vector_dim: int) -> None:
        self._vector_dim = vector_dim
        recreate_collection(
            self._client,
            collection_name=self.collection_name,
            vector_dim=vector_dim,
            distance=self.distance,
        )

    def validate_collection(self, embedding_dim: int) -> None:
        self._vector_dim = embedding_dim
        validate_collection_config(
            self._client,
            collection_name=self.collection_name,
            expected_vector_dim=embedding_dim,
            expected_distance=self.distance,
        )

    def validate_collection_config(self, expected_vector_dim: int) -> None:
        self.validate_collection(expected_vector_dim)

    def create_payload_indexes(self) -> None:
        if not self.payload_indexes:
            return
        if self.mode == "embedded":
            # Local in-process Qdrant currently ignores payload indexes.
            return
        create_payload_indexes(
            self._client,
            collection_name=self.collection_name,
            payload_indexes=self.payload_indexes,
        )

    def upsert(self, chunks: list[EmbeddedChunk]) -> int:
        if not chunks:
            return 0
        inferred_dim, _, _ = self.validate_embedded_chunks(chunks)
        self._vector_dim = inferred_dim
        self.validate_collection(inferred_dim)

        upserted_total = 0
        for start in range(0, len(chunks), self.upsert_batch_size):
            batch = chunks[start : start + self.upsert_batch_size]
            points = [self.point_from_embedded_chunk(item) for item in batch]
            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=self.wait,
            )
            upserted_total += len(batch)
        return upserted_total

    def upsert_embedded_chunks(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        return self.upsert(embedded_chunks)

    def count(self) -> int:
        response = self._client.count(
            collection_name=self.collection_name,
            count_filter=None,
            exact=True,
        )
        return int(response.count)

    def scroll(
        self,
        limit: int = 100,
        offset: models.PointId | None = None,
    ) -> tuple[list[models.Record], models.PointId | None]:
        return self._client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0.")
        if not query_vector:
            raise ValueError("query_vector must not be empty.")

        query_filter = build_qdrant_filter(filters)
        hits = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        ).points

        results: list[RetrievalResult] = []
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            chunk_id = payload.get("chunk_id")
            document_id = payload.get("document_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                continue
            if not isinstance(document_id, str) or not document_id.strip():
                continue
            score = float(hit.score or 0.0)
            title = payload.get("title") or payload.get("source_name") or document_id
            source = Source(
                title=str(title),
                url=payload.get("url"),
                section=payload.get("section"),
                chunk_id=chunk_id,
                document_id=document_id,
                support_score=score,
                source_path=payload.get("source_path"),
                source_type=payload.get("source_type"),
                source_name=payload.get("source_name"),
                heading_path=payload.get("heading_path") or [],
            )
            results.append(
                RetrievalResult(
                    chunk_id=source.chunk_id,
                    document_id=source.document_id,
                    text=str(payload.get("text") or ""),
                    source=source,
                    final_score=score,
                    retrieval_score=score,
                    dense_score=score,
                    retrieval_method=RetrievalMethod.DENSE,
                    rank=rank,
                    metadata={
                        "point_id": str(hit.id),
                        "token_count": payload.get("token_count"),
                        "source_type": payload.get("source_type"),
                        "title": payload.get("title"),
                        "url": payload.get("url"),
                        "embedding_model": payload.get("embedding_model"),
                        "embedding_dim": payload.get("embedding_dim"),
                    },
                )
            )
        return results

    def delete_by_document_id(self, document_id: str) -> int:
        if not document_id.strip():
            raise ValueError("document_id must not be empty.")
        before = self.count()
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id.strip()),
                        )
                    ]
                )
            ),
            wait=self.wait,
        )
        after = self.count()
        return max(0, before - after)
