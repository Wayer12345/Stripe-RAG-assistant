"""Unit tests for domain interface contracts and RawDocument validation."""

import inspect
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from app.domain.interfaces import (
    Chunker,
    Cleaner,
    DocumentLoader,
    Embedder,
    Evaluator,
    Generator,
    LexicalIndex,
    OutputParser,
    Parser,
    RawDocument,
    Reranker,
    Retriever,
    VectorStore,
)
from app.domain.models.answer import Confidence, GeneratedAnswer
from app.domain.models.chunk import Chunk
from app.domain.models.context import ContextBundle
from app.domain.models.document import Document
from app.domain.models.embedded_chunk import EmbeddedChunk
from app.domain.models.eval_case import Difficulty, EvalCase
from app.domain.models.retrieval_result import RetrievalResult
from app.domain.models.source import Source
from pydantic import ValidationError

_NOW = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)


def _raw_document(**kwargs: object) -> RawDocument:
    defaults: dict[str, object] = {
        "source_type": "txt",
        "content": b"Stripe docs raw bytes",
    }
    defaults.update(kwargs)
    return RawDocument(**defaults)


def _document(**kwargs: object) -> Document:
    defaults: dict[str, object] = {
        "id": "doc-1",
        "source_type": "txt",
        "source_path": "docs/guide.txt",
        "source_name": "guide.txt",
        "source_mime_type": "text/plain",
        "text": "Stripe supports online payments.",
        "content_hash": "dochash1",
        "created_at": _NOW,
    }
    defaults.update(kwargs)
    return Document(**defaults)


def _chunk(**kwargs: object) -> Chunk:
    defaults: dict[str, object] = {
        "id": "chunk-1",
        "document_id": "doc-1",
        "text": "Stripe supports online payments.",
        "chunk_index": 0,
        "token_count": 4,
        "content_hash": "chunkhash1",
    }
    defaults.update(kwargs)
    return Chunk(**defaults)


def _source(**kwargs: object) -> Source:
    defaults: dict[str, object] = {
        "title": "Stripe guide",
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "source_path": "docs/guide.txt",
        "source_type": "txt",
        "source_name": "guide.txt",
    }
    defaults.update(kwargs)
    return Source(**defaults)


def _retrieval_result(**kwargs: object) -> RetrievalResult:
    defaults: dict[str, object] = {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "text": "Stripe supports online payments.",
        "source": _source(),
        "final_score": 0.9,
    }
    defaults.update(kwargs)
    return RetrievalResult(**defaults)


def _embedded_chunk(**kwargs: object) -> EmbeddedChunk:
    defaults: dict[str, object] = {
        "chunk": _chunk(),
        "vector": [0.1, 0.2, 0.3],
        "embedding_model": "local-embedder",
        "embedding_dim": 3,
        "normalized": True,
    }
    defaults.update(kwargs)
    return EmbeddedChunk(**defaults)


def _context_bundle(**kwargs: object) -> ContextBundle:
    result = _retrieval_result()
    defaults: dict[str, object] = {
        "query": "What does Stripe support?",
        "chunks": [result],
        "rendered_context": "[1] Stripe supports online payments.",
        "token_count": 8,
        "sources": [result.source],
    }
    defaults.update(kwargs)
    return ContextBundle(**defaults)


def _generated_answer(**kwargs: object) -> GeneratedAnswer:
    defaults: dict[str, object] = {
        "answer": "Stripe supports online payments.",
        "confidence": Confidence.HIGH,
        "sources": [_source()],
        "raw_output": '{"answer":"Stripe supports online payments."}',
        "parsed_successfully": True,
    }
    defaults.update(kwargs)
    return GeneratedAnswer(**defaults)


def _eval_case(**kwargs: object) -> EvalCase:
    defaults: dict[str, object] = {
        "id": "case-1",
        "question": "What does Stripe support?",
        "difficulty": Difficulty.EASY,
        "is_answerable": True,
    }
    defaults.update(kwargs)
    return EvalCase(**defaults)


class TestInterfaceImportability:
    def test_module_imports(self) -> None:
        module_names = [
            "app.domain.interfaces.document_loader",
            "app.domain.interfaces.parser",
            "app.domain.interfaces.cleaner",
            "app.domain.interfaces.chunker",
            "app.domain.interfaces.embedder",
            "app.domain.interfaces.vector_store",
            "app.domain.interfaces.lexical_index",
            "app.domain.interfaces.retriever",
            "app.domain.interfaces.reranker",
            "app.domain.interfaces.generator",
            "app.domain.interfaces.output_parser",
            "app.domain.interfaces.evaluator",
        ]
        for module_name in module_names:
            module = import_module(module_name)
            assert module is not None

    def test_package_exports(self) -> None:
        from app.domain.interfaces import (  # noqa: F401
            Chunker,
            Cleaner,
            DocumentLoader,
            Embedder,
            Evaluator,
            Generator,
            LexicalIndex,
            OutputParser,
            Parser,
            RawDocument,
            Reranker,
            Retriever,
            VectorStore,
        )


class TestRawDocumentValidation:
    def test_valid_minimal(self) -> None:
        raw_document = _raw_document()
        assert raw_document.source_type == "txt"
        assert raw_document.content == b"Stripe docs raw bytes"
        assert raw_document.metadata == {}

    def test_valid_with_optional_fields(self) -> None:
        raw_document = _raw_document(
            source_type="pdf",
            source_path="data/raw/guide.pdf",
            source_name="guide.pdf",
            mime_type="application/pdf",
            metadata={"origin": "test"},
        )
        assert raw_document.source_path == "data/raw/guide.pdf"
        assert raw_document.source_name == "guide.pdf"
        assert raw_document.mime_type == "application/pdf"
        assert raw_document.metadata == {"origin": "test"}

    def test_empty_source_type_fails(self) -> None:
        with pytest.raises(ValidationError):
            _raw_document(source_type="")

    def test_whitespace_source_type_fails(self) -> None:
        with pytest.raises(ValidationError):
            _raw_document(source_type="   ")

    def test_empty_bytes_content_fails(self) -> None:
        with pytest.raises(ValidationError):
            _raw_document(content=b"")

    def test_empty_source_path_fails_when_provided(self) -> None:
        with pytest.raises(ValidationError):
            _raw_document(source_path="")

    def test_empty_source_name_fails_when_provided(self) -> None:
        with pytest.raises(ValidationError):
            _raw_document(source_name="")

    def test_empty_mime_type_fails_when_provided(self) -> None:
        with pytest.raises(ValidationError):
            _raw_document(mime_type="")

    def test_metadata_defaults_to_empty_dict(self) -> None:
        raw_document = _raw_document()
        assert raw_document.metadata == {}

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RawDocument(
                source_type="txt",
                content=b"bytes",
                unknown_field="not-allowed",
            )

    def test_model_dump_works(self) -> None:
        dumped = _raw_document().model_dump()
        assert dumped["source_type"] == "txt"
        assert dumped["content"] == b"Stripe docs raw bytes"

    def test_model_dump_json_mode_works(self) -> None:
        dumped = _raw_document().model_dump(mode="json")
        assert dumped["source_type"] == "txt"
        assert dumped["content"] == "Stripe docs raw bytes"

    def test_model_validate_from_dumped_data(self) -> None:
        dumped = _raw_document(
            source_path="data/raw/guide.txt",
            source_name="guide.txt",
            mime_type="text/plain",
            metadata={"lang": "en"},
        ).model_dump(mode="json")
        restored = RawDocument.model_validate(dumped)
        assert restored.source_name == "guide.txt"
        assert restored.mime_type == "text/plain"
        assert restored.metadata["lang"] == "en"


class FakeDocumentLoader:
    def load(self) -> list[RawDocument]:
        return [_raw_document()]


class FakeParser:
    def supports(self, source_type: str, mime_type: str | None = None) -> bool:
        return source_type == "txt" and (mime_type in {None, "text/plain"})

    def parse(self, raw_document: RawDocument) -> list[Document]:
        return [_document(source_type=raw_document.source_type, source_mime_type=raw_document.mime_type)]


class FakeCleaner:
    def clean(self, document: Document) -> Document:
        cleaned_text = " ".join(document.text.split())
        return _document(
            id=document.id,
            source_type=document.source_type,
            source_path=document.source_path,
            source_name=document.source_name,
            source_mime_type=document.source_mime_type,
            text=cleaned_text,
            content_hash=document.content_hash,
            metadata=document.metadata,
            created_at=document.created_at,
        )


class FakeChunker:
    def chunk(self, document: Document) -> list[Chunk]:
        return [_chunk(document_id=document.id, text=document.text)]


class FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return [float(len(query))]

    def embedding_dim(self) -> int:
        return 1

    def model_name(self) -> str:
        return "fake-embedder"


class FakeVectorStore:
    def __init__(self) -> None:
        self._exists = False
        self._count = 0

    def create_collection(self, *, recreate: bool = False) -> None:
        if recreate:
            self._count = 0
        self._exists = True

    def collection_exists(self) -> bool:
        return self._exists

    def validate_collection(self, embedding_dim: int) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

    def upsert(self, chunks: list[EmbeddedChunk]) -> int:
        self._count += len(chunks)
        return len(chunks)

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        if not query_vector or top_k <= 0:
            return []
        return [_retrieval_result()]

    def delete_by_document_id(self, document_id: str) -> int:
        if document_id == "doc-1" and self._count > 0:
            self._count -= 1
            return 1
        return 0

    def count(self) -> int:
        return self._count

    def healthcheck(self) -> bool:
        return True


class FakeLexicalIndex:
    def __init__(self) -> None:
        self._count = 0
        self._saved_path: Path | None = None
        self._loaded_path: Path | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self._count = len(chunks)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        if not query or top_k <= 0:
            return []
        return [_retrieval_result()]

    def save(self, path: Path) -> None:
        self._saved_path = path

    def load(self, path: Path) -> None:
        self._loaded_path = path

    def count(self) -> int:
        return self._count


class FakeRetriever:
    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        if not query or top_k <= 0:
            return []
        return [_retrieval_result()]


class FakeReranker:
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        _ = query
        return candidates[:top_k]


class FakeGenerator:
    def generate(self, query: str, context: ContextBundle) -> GeneratedAnswer:
        _ = query
        return _generated_answer(sources=context.sources)


class FakeOutputParser:
    def parse(self, raw_output: str, *, available_sources: list[Source]) -> GeneratedAnswer:
        return _generated_answer(raw_output=raw_output, sources=available_sources)


class FakeEvaluator:
    def evaluate(
        self,
        cases: list[EvalCase],
        predictions: list[GeneratedAnswer],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "cases_total": len(cases),
            "predictions_total": len(predictions),
            "metadata": metadata or {},
        }


class TestFakeImplementations:
    def test_fake_document_loader(self) -> None:
        loader: DocumentLoader = FakeDocumentLoader()
        loaded = loader.load()
        assert len(loaded) == 1
        assert isinstance(loaded[0], RawDocument)

    def test_fake_parser(self) -> None:
        parser: Parser = FakeParser()
        assert parser.supports("txt", "text/plain") is True
        parsed = parser.parse(_raw_document(mime_type="text/plain"))
        assert len(parsed) == 1
        assert isinstance(parsed[0], Document)

    def test_fake_cleaner(self) -> None:
        cleaner: Cleaner = FakeCleaner()
        cleaned = cleaner.clean(_document(text="Stripe   supports   payments."))
        assert isinstance(cleaned, Document)
        assert cleaned.text == "Stripe supports payments."

    def test_fake_chunker(self) -> None:
        chunker: Chunker = FakeChunker()
        chunks = chunker.chunk(_document())
        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)

    def test_fake_embedder(self) -> None:
        embedder: Embedder = FakeEmbedder()
        doc_vectors = embedder.embed_documents(["a", "ab"])
        query_vector = embedder.embed_query("abc")
        assert doc_vectors == [[1.0], [2.0]]
        assert query_vector == [3.0]
        assert embedder.embedding_dim() == 1
        assert embedder.model_name() == "fake-embedder"

    def test_fake_vector_store(self) -> None:
        vector_store: VectorStore = FakeVectorStore()
        vector_store.create_collection()
        assert vector_store.collection_exists() is True
        vector_store.validate_collection(embedding_dim=3)
        upserted = vector_store.upsert([_embedded_chunk()])
        assert upserted == 1
        results = vector_store.search([0.1], top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)
        deleted = vector_store.delete_by_document_id("doc-1")
        assert deleted == 1
        assert vector_store.count() == 0
        assert vector_store.healthcheck() is True

    def test_fake_lexical_index(self) -> None:
        lexical_index: LexicalIndex = FakeLexicalIndex()
        lexical_index.build([_chunk()])
        assert lexical_index.count() == 1
        results = lexical_index.search("stripe", top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)
        lexical_index.save(Path("unused/path.pkl"))
        lexical_index.load(Path("unused/path.pkl"))

    def test_fake_retriever(self) -> None:
        retriever: Retriever = FakeRetriever()
        results = retriever.retrieve("stripe", top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)

    def test_fake_reranker(self) -> None:
        reranker: Reranker = FakeReranker()
        candidates = [_retrieval_result()]
        reranked = reranker.rerank("stripe", candidates, top_k=1)
        assert len(reranked) == 1
        assert isinstance(reranked[0], RetrievalResult)

    def test_fake_generator(self) -> None:
        generator: Generator = FakeGenerator()
        answer = generator.generate("stripe", _context_bundle())
        assert isinstance(answer, GeneratedAnswer)
        assert answer.answer

    def test_fake_output_parser(self) -> None:
        output_parser: OutputParser = FakeOutputParser()
        answer = output_parser.parse("raw output", available_sources=[_source()])
        assert isinstance(answer, GeneratedAnswer)
        assert answer.raw_output == "raw output"

    def test_fake_evaluator(self) -> None:
        evaluator: Evaluator = FakeEvaluator()
        report = evaluator.evaluate(
            cases=[_eval_case()],
            predictions=[_generated_answer()],
            metadata={"suite": "unit"},
        )
        assert report["cases_total"] == 1
        assert report["predictions_total"] == 1
        assert report["metadata"]["suite"] == "unit"


# ---------------------------------------------------------------------------
# Protocol method body hardening: every Protocol method must have an explicit
# ellipsis body, not just a docstring.
# ---------------------------------------------------------------------------

_PROTOCOL_METHODS: dict[type, list[str]] = {
    DocumentLoader: ["load"],
    Parser: ["supports", "parse"],
    Cleaner: ["clean"],
    Chunker: ["chunk"],
    Embedder: ["embed_documents", "embed_query", "embedding_dim", "model_name"],
    VectorStore: [
        "create_collection",
        "collection_exists",
        "validate_collection",
        "upsert",
        "search",
        "delete_by_document_id",
        "count",
        "healthcheck",
    ],
    LexicalIndex: ["build", "search", "save", "load", "count"],
    Retriever: ["retrieve"],
    Reranker: ["rerank"],
    Generator: ["generate"],
    OutputParser: ["parse"],
    Evaluator: ["evaluate"],
}


def _has_ellipsis_body(method: object) -> bool:
    """Return True when the method source contains an explicit ``...`` body line."""
    try:
        source = inspect.getsource(method)  # type: ignore[arg-type]
    except (OSError, TypeError):
        return False
    # Strip away the docstring block and check the remaining lines for `...`.
    # We look for a line whose stripped content is exactly `...`.
    lines = source.splitlines()
    in_docstring = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('"""', "'''")):
            # Toggle docstring state; single-line docstrings start and end on the same line.
            quote = '"""' if stripped.startswith('"""') else "'''"
            count = stripped.count(quote)
            if count >= 2:
                # Single-line docstring — already closed.
                continue
            in_docstring = not in_docstring
            continue
        if in_docstring:
            if '"""' in stripped or "'''" in stripped:
                in_docstring = False
            continue
        if stripped == "...":
            return True
    return False


class TestProtocolMethodBodies:
    """Every Protocol method must have an explicit ``...`` body, not just a docstring."""

    @pytest.mark.parametrize(
        ("protocol_cls", "method_name"),
        [
            (cls, method)
            for cls, methods in _PROTOCOL_METHODS.items()
            for method in methods
        ],
        ids=lambda x: x if isinstance(x, str) else x.__name__,
    )
    def test_method_has_ellipsis_body(self, protocol_cls: type, method_name: str) -> None:
        method = getattr(protocol_cls, method_name, None)
        assert method is not None, (
            f"{protocol_cls.__name__}.{method_name} not found"
        )
        assert _has_ellipsis_body(method), (
            f"{protocol_cls.__name__}.{method_name} is missing an explicit "
            f"'...' body — docstring-only method bodies are not allowed."
        )

