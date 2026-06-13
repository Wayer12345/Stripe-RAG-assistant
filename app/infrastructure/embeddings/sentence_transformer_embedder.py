"""Sentence-transformers based local embedder implementation."""

from __future__ import annotations

import math
from typing import Literal

from app.domain.interfaces.embedder import Embedder
from app.domain.models.chunk import Chunk
from app.domain.models.embedded_chunk import EmbeddedChunk

_QUERY_PROMPT_BY_MODE: dict[str, str] = {
    "none": "",
    "bge": "Represent this sentence for searching relevant passages: ",
    "e5": "query: ",
}
_DOCUMENT_PROMPT_BY_MODE: dict[str, str] = {
    "none": "",
    "bge": "",
    "e5": "passage: ",
}


class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by local sentence-transformers models."""

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        prefix_mode: Literal["none", "bge", "e5"] = "bge",
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty.")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if prefix_mode not in _QUERY_PROMPT_BY_MODE:
            allowed = ", ".join(sorted(_QUERY_PROMPT_BY_MODE))
            raise ValueError(f"Unsupported prefix_mode={prefix_mode!r}. Allowed: {allowed}.")

        self._model_name = model_name.strip()
        self._batch_size = batch_size
        self._normalize_embeddings = normalize_embeddings
        self._prefix_mode = prefix_mode
        self._model: object | None = None
        self._embedding_dim_cache: int | None = None

    @property
    def normalize_embeddings(self) -> bool:
        return self._normalize_embeddings

    @property
    def prefix_mode(self) -> str:
        return self._prefix_mode

    @staticmethod
    def query_prefix_for_mode(prefix_mode: str) -> str:
        return _QUERY_PROMPT_BY_MODE[prefix_mode]

    @staticmethod
    def document_prefix_for_mode(prefix_mode: str) -> str:
        return _DOCUMENT_PROMPT_BY_MODE[prefix_mode]

    def _get_model(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _prefix_for(self, input_type: Literal["query", "document"]) -> str:
        if input_type == "query":
            return self.query_prefix_for_mode(self._prefix_mode)
        return self.document_prefix_for_mode(self._prefix_mode)

    def _validate_vectors(self, vectors: list[list[float]], expected_count: int) -> list[list[float]]:
        if len(vectors) != expected_count:
            raise ValueError(
                f"Embedding output count mismatch: expected {expected_count}, got {len(vectors)}."
            )
        if not vectors:
            return vectors

        dim = len(vectors[0])
        if dim <= 0:
            raise ValueError("Embedding vectors must have positive dimension.")

        for vector in vectors:
            if len(vector) != dim:
                raise ValueError("Inconsistent embedding dimensions returned by model.")
            for value in vector:
                if not math.isfinite(value):
                    raise ValueError("Non-finite embedding value returned by model.")

        self._embedding_dim_cache = dim
        return vectors

    def embed_texts(
        self,
        texts: list[str],
        *,
        input_type: Literal["query", "document"],
    ) -> list[list[float]]:
        """Embed texts in batch and return plain Python vectors."""
        if not texts:
            return []

        prefix = self._prefix_for(input_type)
        prefixed_texts = [f"{prefix}{text}" for text in texts]
        model = self._get_model()
        raw_vectors = model.encode(
            prefixed_texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        vectors = raw_vectors.tolist()
        if vectors and isinstance(vectors[0], float):
            vectors = [vectors]
        typed_vectors = [[float(value) for value in vector] for vector in vectors]
        return self._validate_vectors(typed_vectors, expected_count=len(texts))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document/passages for retrieval indexing."""
        return self.embed_texts(texts, input_type="document")

    def embed_query(self, query: str) -> list[float]:
        """Embed one query string."""
        vectors = self.embed_texts([query], input_type="query")
        return vectors[0]

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed domain chunks into EmbeddedChunk records preserving order."""
        if not chunks:
            return []
        vectors = self.embed_documents([chunk.text for chunk in chunks])
        dim = len(vectors[0]) if vectors else self.embedding_dim()
        return [
            EmbeddedChunk(
                chunk=chunk,
                vector=vector,
                embedding_model=self._model_name,
                embedding_dim=dim,
                normalized=self._normalize_embeddings,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    def embedding_dim(self) -> int:
        """Return embedding dimensionality from model metadata."""
        if self._embedding_dim_cache is not None:
            return self._embedding_dim_cache

        model = self._get_model()
        dim = int(model.get_sentence_embedding_dimension())
        if dim <= 0:
            raise ValueError(
                f"Model {self._model_name!r} returned invalid embedding dimension: {dim}"
            )
        self._embedding_dim_cache = dim
        return dim

    def model_name(self) -> str:
        return self._model_name
