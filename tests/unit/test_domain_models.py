"""Unit tests for core domain models.

Covers valid construction, validation failures, and serialization round-trips
for all domain models.  No external services or files are used.
"""

import math
from datetime import UTC, datetime

import pytest
from app.domain.models.answer import Confidence, GeneratedAnswer
from app.domain.models.chunk import Chunk
from app.domain.models.context import ContextBundle
from app.domain.models.document import Document, DocumentProcessingStage
from app.domain.models.embedded_chunk import EmbeddedChunk
from app.domain.models.eval_case import Difficulty, EvalCase, EvalCaseType
from app.domain.models.retrieval_result import RetrievalMethod, RetrievalResult
from app.domain.models.source import Source
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)


def _document(**kwargs: object) -> Document:
    defaults: dict = {
        "id": "doc-1",
        "source_type": "html",
        "text": "Stripe charges fees per transaction.",
        "content_hash": "abc123",
        "created_at": _NOW,
    }
    defaults.update(kwargs)
    return Document(**defaults)


def _chunk(**kwargs: object) -> Chunk:
    defaults: dict = {
        "id": "chunk-1",
        "document_id": "doc-1",
        "text": "Stripe charges fees.",
        "chunk_index": 0,
        "token_count": 5,
        "content_hash": "chunkhash1",
    }
    defaults.update(kwargs)
    return Chunk(**defaults)


def _source(**kwargs: object) -> Source:
    defaults: dict = {
        "title": "Stripe Fees Guide",
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
    }
    defaults.update(kwargs)
    return Source(**defaults)


def _retrieval_result(**kwargs: object) -> RetrievalResult:
    defaults: dict = {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "text": "Stripe charges fees.",
        "source": _source(),
        "final_score": 0.9,
    }
    defaults.update(kwargs)
    return RetrievalResult(**defaults)


def _embedded_chunk(**kwargs: object) -> EmbeddedChunk:
    defaults: dict = {
        "chunk": _chunk(),
        "vector": [0.1, 0.2, 0.3],
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dim": 3,
        "normalized": True,
    }
    defaults.update(kwargs)
    return EmbeddedChunk(**defaults)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class TestDocument:
    def test_valid_minimal(self) -> None:
        doc = _document()
        assert doc.id == "doc-1"
        assert doc.source_type == "html"
        assert doc.metadata == {}
        assert doc.processing_stage == DocumentProcessingStage.PARSED

    def test_valid_with_optional_fields(self) -> None:
        doc = _document(
            url="https://stripe.com/docs/fees",
            title="Stripe Fees",
            source_path="/data/fees.html",
            metadata={"lang": "en"},
        )
        assert doc.url == "https://stripe.com/docs/fees"
        assert doc.metadata["lang"] == "en"

    # --- new source identity fields ---

    def test_valid_with_source_identity_fields(self) -> None:
        doc = _document(
            source_id="fees-guide-v2",
            source_name="Stripe Fees Guide",
            source_mime_type="text/html",
        )
        assert doc.source_id == "fees-guide-v2"
        assert doc.source_name == "Stripe Fees Guide"
        assert doc.source_mime_type == "text/html"

    def test_empty_source_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            _document(source_id="")

    def test_whitespace_source_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            _document(source_id="   ")

    def test_empty_source_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            _document(source_name="")

    def test_empty_source_mime_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            _document(source_mime_type="")

    # --- processing_stage ---

    def test_default_processing_stage_is_parsed(self) -> None:
        doc = _document()
        assert doc.processing_stage == DocumentProcessingStage.PARSED

    def test_valid_processing_stage_raw(self) -> None:
        doc = _document(processing_stage=DocumentProcessingStage.RAW)
        assert doc.processing_stage == DocumentProcessingStage.RAW

    def test_valid_processing_stage_cleaned(self) -> None:
        doc = _document(processing_stage=DocumentProcessingStage.CLEANED)
        assert doc.processing_stage == DocumentProcessingStage.CLEANED

    def test_invalid_processing_stage_raises(self) -> None:
        with pytest.raises(ValidationError):
            _document(processing_stage="indexed")  # type: ignore[arg-type]

    # --- pre-existing validations ---

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            _document(id="")

    def test_whitespace_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            _document(id="   ")

    def test_empty_source_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            _document(source_type="")

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            _document(text="")

    def test_whitespace_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            _document(text="   ")

    def test_empty_content_hash_raises(self) -> None:
        with pytest.raises(ValidationError):
            _document(content_hash="")

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            Document(
                id="doc-1",
                source_type="html",
                text="text",
                content_hash="h",
                created_at=_NOW,
                unknown_field="oops",
            )

    def test_serialization_round_trip(self) -> None:
        doc = _document(
            source_id="sid",
            source_name="Guide",
            source_mime_type="application/pdf",
            processing_stage=DocumentProcessingStage.CLEANED,
        )
        dumped = doc.model_dump(mode="json")
        restored = Document.model_validate(dumped)
        assert restored == doc

    def test_datetime_serialized_as_string(self) -> None:
        doc = _document()
        dumped = doc.model_dump(mode="json")
        assert isinstance(dumped["created_at"], str)


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class TestChunk:
    def test_valid_minimal(self) -> None:
        c = _chunk()
        assert c.chunk_index == 0
        assert c.token_count == 5
        assert c.metadata == {}
        assert c.heading_path == []
        assert c.content_hash == "chunkhash1"

    def test_valid_with_char_offsets(self) -> None:
        c = _chunk(char_start=0, char_end=20)
        assert c.char_start == 0
        assert c.char_end == 20

    def test_valid_with_heading_path_and_section(self) -> None:
        c = _chunk(heading_path=["Introduction", "Fees"], section="Fees")
        assert c.heading_path == ["Introduction", "Fees"]
        assert c.section == "Fees"

    def test_valid_with_page_number(self) -> None:
        c = _chunk(page_number=3)
        assert c.page_number == 3

    def test_valid_with_line_range(self) -> None:
        c = _chunk(start_line=10, end_line=20)
        assert c.start_line == 10
        assert c.end_line == 20

    def test_valid_with_chunking_strategy(self) -> None:
        c = _chunk(chunking_strategy="heading_aware")
        assert c.chunking_strategy == "heading_aware"

    # --- content_hash ---

    def test_missing_content_hash_raises(self) -> None:
        with pytest.raises((ValidationError, TypeError)):
            Chunk(
                id="chunk-1",
                document_id="doc-1",
                text="text",
                chunk_index=0,
                token_count=5,
            )

    def test_empty_content_hash_raises(self) -> None:
        with pytest.raises(ValidationError):
            _chunk(content_hash="")

    # --- heading_path ---

    def test_empty_item_in_heading_path_raises(self) -> None:
        with pytest.raises(ValidationError):
            _chunk(heading_path=["Introduction", ""])

    def test_whitespace_heading_raises(self) -> None:
        with pytest.raises(ValidationError):
            _chunk(heading_path=["  "])

    # --- chunking_strategy / section ---

    def test_empty_chunking_strategy_raises(self) -> None:
        with pytest.raises(ValidationError):
            _chunk(chunking_strategy="")

    def test_empty_section_raises(self) -> None:
        with pytest.raises(ValidationError):
            _chunk(section="")

    # --- page_number ---

    def test_page_number_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            _chunk(page_number=0)

    def test_page_number_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            _chunk(page_number=-1)

    # --- line range ---

    def test_start_line_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            _chunk(start_line=0)

    def test_end_line_before_start_line_raises(self) -> None:
        with pytest.raises(ValidationError):
            _chunk(start_line=10, end_line=5)

    # --- pre-existing validations ---

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            _chunk(id="")

    def test_empty_document_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            _chunk(document_id="")

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            _chunk(text="")

    def test_negative_chunk_index_raises(self) -> None:
        with pytest.raises(ValidationError):
            _chunk(chunk_index=-1)

    def test_zero_token_count_raises(self) -> None:
        with pytest.raises(ValidationError):
            _chunk(token_count=0)

    def test_negative_token_count_raises(self) -> None:
        with pytest.raises(ValidationError):
            _chunk(token_count=-5)

    def test_char_start_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            _chunk(char_start=-1, char_end=10)

    def test_char_end_before_start_raises(self) -> None:
        with pytest.raises(ValidationError):
            _chunk(char_start=10, char_end=5)

    def test_only_char_start_provided_allowed(self) -> None:
        c = _chunk(char_start=0)
        assert c.char_start == 0
        assert c.char_end is None

    def test_serialization_round_trip(self) -> None:
        c = _chunk(
            char_start=0,
            char_end=20,
            heading_path=["Intro", "Fees"],
            section="Fees",
            page_number=2,
            start_line=5,
            end_line=10,
            chunking_strategy="heading_aware",
            metadata={"extra": "val"},
        )
        restored = Chunk.model_validate(c.model_dump(mode="json"))
        assert restored == c


# ---------------------------------------------------------------------------
# EmbeddedChunk
# ---------------------------------------------------------------------------


class TestEmbeddedChunk:
    def test_valid(self) -> None:
        ec = _embedded_chunk()
        assert ec.embedding_dim == 3
        assert ec.normalized is True

    def test_empty_vector_raises(self) -> None:
        with pytest.raises(ValidationError):
            _embedded_chunk(vector=[], embedding_dim=0)

    def test_dim_mismatch_raises(self) -> None:
        with pytest.raises(ValidationError):
            _embedded_chunk(vector=[0.1, 0.2], embedding_dim=3)

    def test_nan_in_vector_raises(self) -> None:
        with pytest.raises(ValidationError):
            _embedded_chunk(vector=[0.1, float("nan"), 0.3], embedding_dim=3)

    def test_positive_inf_in_vector_raises(self) -> None:
        with pytest.raises(ValidationError):
            _embedded_chunk(vector=[0.1, math.inf, 0.3], embedding_dim=3)

    def test_negative_inf_in_vector_raises(self) -> None:
        with pytest.raises(ValidationError):
            _embedded_chunk(vector=[0.1, -math.inf, 0.3], embedding_dim=3)

    def test_empty_embedding_model_raises(self) -> None:
        with pytest.raises(ValidationError):
            _embedded_chunk(embedding_model="")

    def test_zero_embedding_dim_raises(self) -> None:
        with pytest.raises(ValidationError):
            _embedded_chunk(vector=[0.1], embedding_dim=0)

    def test_serialization_round_trip(self) -> None:
        ec = _embedded_chunk()
        restored = EmbeddedChunk.model_validate(ec.model_dump(mode="json"))
        assert restored == ec


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


class TestSource:
    def test_valid_minimal(self) -> None:
        s = _source()
        assert s.title == "Stripe Fees Guide"
        assert s.support_score is None
        assert s.heading_path == []

    def test_valid_with_score(self) -> None:
        s = _source(support_score=0.85)
        assert s.support_score == pytest.approx(0.85)

    def test_valid_boundary_scores(self) -> None:
        assert _source(support_score=0.0).support_score == 0.0
        assert _source(support_score=1.0).support_score == 1.0

    def test_valid_local_file_source(self) -> None:
        s = _source(
            source_path="/data/stripe_fees.pdf",
            source_type="pdf",
            source_name="stripe_fees.pdf",
            page_number=5,
            heading_path=["Fees", "Card fees"],
        )
        assert s.source_path == "/data/stripe_fees.pdf"
        assert s.page_number == 5
        assert s.heading_path == ["Fees", "Card fees"]

    def test_valid_url_source_unchanged(self) -> None:
        s = _source(url="https://stripe.com/docs/fees", section="Card fees")
        assert s.url == "https://stripe.com/docs/fees"
        assert s.source_path is None

    def test_valid_line_range(self) -> None:
        s = _source(start_line=10, end_line=20)
        assert s.start_line == 10
        assert s.end_line == 20

    def test_empty_source_path_raises(self) -> None:
        with pytest.raises(ValidationError):
            _source(source_path="")

    def test_empty_source_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            _source(source_type="")

    def test_empty_source_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            _source(source_name="")

    def test_empty_heading_path_item_raises(self) -> None:
        with pytest.raises(ValidationError):
            _source(heading_path=["Fees", ""])

    def test_page_number_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            _source(page_number=0)

    def test_start_line_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            _source(start_line=0)

    def test_end_line_before_start_line_raises(self) -> None:
        with pytest.raises(ValidationError):
            _source(start_line=10, end_line=5)

    def test_empty_title_raises(self) -> None:
        with pytest.raises(ValidationError):
            _source(title="")

    def test_whitespace_title_raises(self) -> None:
        with pytest.raises(ValidationError):
            _source(title="   ")

    def test_empty_chunk_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            _source(chunk_id="")

    def test_empty_document_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            _source(document_id="")

    def test_score_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            _source(support_score=-0.01)

    def test_score_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            _source(support_score=1.01)

    def test_serialization_round_trip(self) -> None:
        s = _source(
            url="https://stripe.com",
            section="fees",
            support_score=0.7,
            source_path="/data/fees.pdf",
            source_type="pdf",
            source_name="fees.pdf",
            heading_path=["Fees"],
            page_number=2,
            start_line=5,
            end_line=10,
        )
        restored = Source.model_validate(s.model_dump(mode="json"))
        assert restored == s


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    def test_valid_minimal(self) -> None:
        r = _retrieval_result()
        assert r.final_score == pytest.approx(0.9)
        assert r.metadata == {}
        assert r.matched_terms == []
        assert r.rank is None

    def test_valid_with_all_scores(self) -> None:
        r = _retrieval_result(
            dense_score=0.8,
            lexical_score=0.5,
            reranker_score=0.9,
            retrieval_score=0.75,
        )
        assert r.dense_score == pytest.approx(0.8)

    def test_valid_retrieval_method(self) -> None:
        r = _retrieval_result(retrieval_method=RetrievalMethod.HYBRID)
        assert r.retrieval_method == RetrievalMethod.HYBRID

    def test_all_retrieval_methods_accepted(self) -> None:
        for method in RetrievalMethod:
            r = _retrieval_result(retrieval_method=method)
            assert r.retrieval_method == method

    def test_invalid_retrieval_method_raises(self) -> None:
        with pytest.raises(ValidationError):
            _retrieval_result(retrieval_method="fuzzy")  # type: ignore[arg-type]

    def test_valid_rank_and_original_rank(self) -> None:
        r = _retrieval_result(rank=1, original_rank=3)
        assert r.rank == 1
        assert r.original_rank == 3

    def test_rank_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            _retrieval_result(rank=0)

    def test_original_rank_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            _retrieval_result(original_rank=0)

    def test_matched_terms_valid(self) -> None:
        r = _retrieval_result(matched_terms=["PCI DSS", "3D Secure", "VAT"])
        assert r.matched_terms == ["PCI DSS", "3D Secure", "VAT"]

    def test_matched_terms_deduplicates(self) -> None:
        r = _retrieval_result(matched_terms=["fee", "fee", "tax"])
        assert r.matched_terms == ["fee", "tax"]

    def test_matched_terms_empty_item_raises(self) -> None:
        with pytest.raises(ValidationError):
            _retrieval_result(matched_terms=["fee", ""])

    def test_matched_terms_whitespace_item_raises(self) -> None:
        with pytest.raises(ValidationError):
            _retrieval_result(matched_terms=["  "])

    def test_empty_chunk_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            _retrieval_result(chunk_id="")

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            _retrieval_result(text="")

    def test_non_finite_final_score_raises(self) -> None:
        with pytest.raises(ValidationError):
            _retrieval_result(final_score=math.inf)

    def test_nan_final_score_raises(self) -> None:
        with pytest.raises(ValidationError):
            _retrieval_result(final_score=float("nan"))

    def test_non_finite_optional_score_raises(self) -> None:
        with pytest.raises(ValidationError):
            _retrieval_result(dense_score=float("inf"))

    def test_mismatched_source_chunk_id_raises(self) -> None:
        bad_source = _source(chunk_id="other-chunk")
        with pytest.raises(ValidationError):
            _retrieval_result(source=bad_source)

    def test_mismatched_source_document_id_raises(self) -> None:
        bad_source = _source(document_id="other-doc")
        with pytest.raises(ValidationError):
            _retrieval_result(source=bad_source)

    def test_serialization_round_trip(self) -> None:
        r = _retrieval_result(
            dense_score=0.8,
            lexical_score=0.6,
            retrieval_method=RetrievalMethod.HYBRID,
            rank=1,
            original_rank=3,
            matched_terms=["fee", "tax"],
        )
        restored = RetrievalResult.model_validate(r.model_dump(mode="json"))
        assert restored == r


# ---------------------------------------------------------------------------
# ContextBundle
# ---------------------------------------------------------------------------


class TestContextBundle:
    def _bundle(self, **kwargs: object) -> ContextBundle:
        result = _retrieval_result()
        source = result.source
        defaults: dict = {
            "query": "What are Stripe's fees?",
            "chunks": [result],
            "rendered_context": "[1] Stripe charges fees.",
            "token_count": 10,
            "sources": [source],
        }
        defaults.update(kwargs)
        return ContextBundle(**defaults)

    def test_valid(self) -> None:
        bundle = self._bundle()
        assert bundle.token_count == 10
        assert len(bundle.chunks) == 1
        assert bundle.truncated is False
        assert bundle.dropped_chunk_ids == []

    def test_empty_chunks_and_empty_rendered_context_allowed(self) -> None:
        bundle = ContextBundle(
            query="What?",
            chunks=[],
            rendered_context="",
            token_count=0,
            sources=[],
        )
        assert bundle.chunks == []
        assert bundle.rendered_context == ""

    def test_valid_token_budget_respected(self) -> None:
        bundle = self._bundle(token_count=10, token_budget=100)
        assert bundle.token_budget == 100

    def test_token_count_equals_budget_allowed(self) -> None:
        bundle = self._bundle(token_count=10, token_budget=10)
        assert bundle.token_count == bundle.token_budget

    def test_token_count_exceeds_budget_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._bundle(token_count=200, token_budget=100)

    def test_negative_token_budget_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._bundle(token_budget=-1)

    def test_valid_truncated_with_dropped_ids(self) -> None:
        bundle = self._bundle(
            truncated=True,
            dropped_chunk_ids=["chunk-dropped-1", "chunk-dropped-2"],
        )
        assert bundle.truncated is True
        assert "chunk-dropped-1" in bundle.dropped_chunk_ids

    def test_empty_dropped_chunk_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._bundle(dropped_chunk_ids=[""])

    def test_dropped_id_overlapping_included_chunk_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._bundle(dropped_chunk_ids=["chunk-1"])  # chunk-1 is in chunks

    def test_valid_context_format_version(self) -> None:
        bundle = self._bundle(context_format_version="v2")
        assert bundle.context_format_version == "v2"

    def test_empty_context_format_version_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._bundle(context_format_version="")

    def test_whitespace_context_format_version_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._bundle(context_format_version="   ")

    def test_chunks_present_but_empty_rendered_context_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._bundle(rendered_context="")

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._bundle(query="")

    def test_negative_token_count_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._bundle(token_count=-1)

    def test_source_not_in_chunks_raises(self) -> None:
        orphan_source = Source(
            title="Unknown",
            chunk_id="orphan-chunk",
            document_id="doc-1",
        )
        with pytest.raises(ValidationError):
            self._bundle(sources=[orphan_source])

    def test_serialization_round_trip(self) -> None:
        bundle = self._bundle(
            token_budget=100,
            truncated=True,
            dropped_chunk_ids=["chunk-99"],
            context_format_version="v1",
        )
        restored = ContextBundle.model_validate(bundle.model_dump(mode="json"))
        assert restored == bundle


# ---------------------------------------------------------------------------
# GeneratedAnswer
# ---------------------------------------------------------------------------


class TestGeneratedAnswer:
    def _answer(self, **kwargs: object) -> GeneratedAnswer:
        defaults: dict = {
            "answer": "Stripe charges 2.9% + 30¢ per transaction.",
            "confidence": Confidence.HIGH,
            "raw_output": '{"answer": "..."}',
            "parsed_successfully": True,
        }
        defaults.update(kwargs)
        return GeneratedAnswer(**defaults)

    def test_valid(self) -> None:
        a = self._answer()
        assert a.confidence == Confidence.HIGH
        assert a.sources == []
        assert a.metadata == {}

    def test_valid_no_answer_confidence_none(self) -> None:
        a = self._answer(
            answer="I cannot answer this question.",
            confidence=Confidence.NONE,
            parsed_successfully=False,
            raw_output="",
        )
        assert a.confidence == Confidence.NONE

    def test_empty_answer_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._answer(answer="")

    def test_whitespace_answer_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._answer(answer="   ")

    def test_invalid_confidence_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._answer(confidence="very_high")  # type: ignore[arg-type]

    def test_all_confidence_values_accepted(self) -> None:
        for conf in Confidence:
            a = self._answer(confidence=conf)
            assert a.confidence == conf

    def test_serialization_round_trip(self) -> None:
        a = self._answer(sources=[_source()])
        restored = GeneratedAnswer.model_validate(a.model_dump(mode="json"))
        assert restored == a


# ---------------------------------------------------------------------------
# EvalCase
# ---------------------------------------------------------------------------


class TestEvalCase:
    def _case(self, **kwargs: object) -> EvalCase:
        defaults: dict = {
            "id": "eval-1",
            "question": "What is Stripe?",
            "difficulty": Difficulty.EASY,
            "is_answerable": True,
        }
        defaults.update(kwargs)
        return EvalCase(**defaults)

    def test_valid_minimal(self) -> None:
        c = self._case()
        assert c.tags == []
        assert c.expected_sources == []
        assert c.expected_chunk_ids == []
        assert c.expected_source_titles == []
        assert c.notes is None
        assert c.case_type is None
        assert c.metadata == {}

    def test_valid_answerable_without_sources_allowed(self) -> None:
        c = self._case(is_answerable=True, expected_sources=[])
        assert c.is_answerable is True

    def test_valid_with_tags_and_sources(self) -> None:
        c = self._case(
            tags=["fees", "pricing"],
            expected_sources=["https://stripe.com/docs/fees"],
        )
        assert "fees" in c.tags

    def test_valid_expected_chunk_ids(self) -> None:
        c = self._case(expected_chunk_ids=["chunk-abc", "chunk-def"])
        assert c.expected_chunk_ids == ["chunk-abc", "chunk-def"]

    def test_empty_expected_chunk_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._case(expected_chunk_ids=["chunk-abc", ""])

    def test_valid_expected_source_titles(self) -> None:
        c = self._case(expected_source_titles=["Stripe Fees Guide"])
        assert c.expected_source_titles == ["Stripe Fees Guide"]

    def test_empty_expected_source_title_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._case(expected_source_titles=[""])

    def test_valid_notes(self) -> None:
        c = self._case(notes="Answer depends on country.")
        assert c.notes == "Answer depends on country."

    def test_empty_notes_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._case(notes="")

    def test_whitespace_notes_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._case(notes="   ")

    def test_valid_case_type(self) -> None:
        c = self._case(case_type=EvalCaseType.FACTUAL)
        assert c.case_type == EvalCaseType.FACTUAL

    def test_all_case_types_accepted(self) -> None:
        for ct in EvalCaseType:
            c = self._case(case_type=ct)
            assert c.case_type == ct

    def test_invalid_case_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._case(case_type="tricky")  # type: ignore[arg-type]

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._case(id="")

    def test_empty_question_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._case(question="")

    def test_whitespace_question_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._case(question="   ")

    def test_empty_string_in_tags_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._case(tags=["fees", ""])

    def test_whitespace_tag_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._case(tags=["  "])

    def test_empty_string_in_expected_sources_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._case(expected_sources=["https://stripe.com", ""])

    def test_all_difficulty_values_accepted(self) -> None:
        for diff in Difficulty:
            c = self._case(difficulty=diff)
            assert c.difficulty == diff

    def test_invalid_difficulty_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._case(difficulty="trivial")  # type: ignore[arg-type]

    def test_serialization_round_trip(self) -> None:
        c = self._case(
            tags=["billing"],
            expected_sources=["https://stripe.com"],
            expected_chunk_ids=["chunk-1"],
            expected_source_titles=["Fees Guide"],
            expected_answer="Stripe is a payments platform.",
            case_type=EvalCaseType.PROCEDURAL,
            notes="Check fee schedule.",
        )
        restored = EvalCase.model_validate(c.model_dump(mode="json"))
        assert restored == c


# ---------------------------------------------------------------------------
# Cross-model: package-level imports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_imports_from_models_package(self) -> None:
        from app.domain.models import (  # noqa: F401
            Chunk,
            Confidence,
            ContextBundle,
            Difficulty,
            Document,
            DocumentProcessingStage,
            EmbeddedChunk,
            EvalCase,
            EvalCaseType,
            GeneratedAnswer,
            RetrievalMethod,
            RetrievalResult,
            Source,
        )
