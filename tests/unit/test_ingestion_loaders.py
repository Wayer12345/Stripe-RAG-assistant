"""Unit tests for FileLoader, SourceRegistry, and TxtParser."""

import pytest
from app.domain.interfaces.document_loader import RawDocument
from app.domain.models.document import DocumentProcessingStage
from app.infrastructure.loaders.file_loader import FileLoader
from app.infrastructure.loaders.source_registry import SourceRegistry
from app.infrastructure.parsers.txt_parser import TxtParser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw(
    content: bytes = b"hello world",
    source_type: str = "txt",
    source_path: str | None = "/data/doc.txt",
    source_name: str | None = "doc.txt",
    mime_type: str | None = "text/plain",
) -> RawDocument:
    return RawDocument(
        source_type=source_type,
        content=content,
        source_path=source_path,
        source_name=source_name,
        mime_type=mime_type,
    )


# ---------------------------------------------------------------------------
# FileLoader
# ---------------------------------------------------------------------------


class TestFileLoader:
    def test_load_single_txt_file(self, tmp_path: pytest.TempPathFactory) -> None:
        (tmp_path / "doc.txt").write_text("hello", encoding="utf-8")
        loader = FileLoader(input_dir=tmp_path)
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].source_type == "txt"
        assert docs[0].source_name == "doc.txt"

    def test_load_reads_bytes_correctly(self, tmp_path: pytest.TempPathFactory) -> None:
        content = "stripe payment guide"
        (tmp_path / "guide.txt").write_text(content, encoding="utf-8")
        loader = FileLoader(input_dir=tmp_path)
        docs = loader.load()
        assert docs[0].content == content.encode("utf-8")

    def test_skips_unsupported_extension(self, tmp_path: pytest.TempPathFactory) -> None:
        (tmp_path / "doc.txt").write_text("ok", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        loader = FileLoader(input_dir=tmp_path, supported_extensions={".txt"})
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].source_name == "doc.txt"

    def test_skips_directories(self, tmp_path: pytest.TempPathFactory) -> None:
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "nested.txt").write_text("nested", encoding="utf-8")
        # Non-recursive: only top-level
        loader = FileLoader(input_dir=tmp_path, recursive=False)
        docs = loader.load()
        assert all(d.source_name != "nested.txt" for d in docs)

    def test_recursive_finds_nested_files(self, tmp_path: pytest.TempPathFactory) -> None:
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "deep.txt").write_text("deep", encoding="utf-8")
        loader = FileLoader(input_dir=tmp_path, recursive=True)
        docs = loader.load()
        names = [d.source_name for d in docs]
        assert "deep.txt" in names

    def test_sorted_deterministic_order(self, tmp_path: pytest.TempPathFactory) -> None:
        for name in ("c.txt", "a.txt", "b.txt"):
            (tmp_path / name).write_text("x", encoding="utf-8")
        loader = FileLoader(input_dir=tmp_path)
        docs = loader.load()
        names = [d.source_name for d in docs]
        assert names == sorted(names)

    def test_metadata_contains_file_size_and_extension(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        (tmp_path / "meta.txt").write_text("content", encoding="utf-8")
        loader = FileLoader(input_dir=tmp_path)
        doc = loader.load()[0]
        assert "file_size" in doc.metadata
        assert "extension" in doc.metadata
        assert doc.metadata["extension"] == ".txt"

    def test_mime_type_populated(self, tmp_path: pytest.TempPathFactory) -> None:
        (tmp_path / "file.txt").write_text("text", encoding="utf-8")
        loader = FileLoader(input_dir=tmp_path)
        doc = loader.load()[0]
        assert doc.mime_type == "text/plain"

    def test_source_path_is_absolute_str(self, tmp_path: pytest.TempPathFactory) -> None:
        (tmp_path / "file.txt").write_text("x", encoding="utf-8")
        loader = FileLoader(input_dir=tmp_path)
        doc = loader.load()[0]
        assert isinstance(doc.source_path, str)
        assert doc.source_path.startswith("/")

    def test_empty_directory_returns_empty_list(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        loader = FileLoader(input_dir=tmp_path)
        assert loader.load() == []

    def test_custom_extensions_accepted(self, tmp_path: pytest.TempPathFactory) -> None:
        (tmp_path / "note.md").write_text("# title", encoding="utf-8")
        (tmp_path / "doc.txt").write_text("text", encoding="utf-8")
        loader = FileLoader(input_dir=tmp_path, supported_extensions={".md"})
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].source_type == "md"

    def test_nonexistent_dir_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        missing = tmp_path / "missing"
        with pytest.raises(ValueError, match="does not exist"):
            FileLoader(input_dir=missing)

    def test_file_path_passed_as_input_dir_raises(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        f = tmp_path / "file.txt"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="not a directory"):
            FileLoader(input_dir=f)

    def test_source_type_lowercased_no_dot(self, tmp_path: pytest.TempPathFactory) -> None:
        (tmp_path / "DOC.TXT").write_text("text", encoding="utf-8")
        loader = FileLoader(input_dir=tmp_path, supported_extensions={".txt"})
        docs = loader.load()
        assert docs[0].source_type == "txt"


# ---------------------------------------------------------------------------
# SourceRegistry
# ---------------------------------------------------------------------------


class TestSourceRegistry:
    def test_resolves_registered_parser(self) -> None:
        parser = TxtParser()
        registry = SourceRegistry([parser])
        raw = _make_raw(source_type="txt")
        resolved = registry.resolve(raw)
        assert resolved is parser

    def test_resolves_by_mime_type(self) -> None:
        parser = TxtParser()
        registry = SourceRegistry([parser])
        raw = _make_raw(source_type="unknown", mime_type="text/plain")
        resolved = registry.resolve(raw)
        assert resolved is parser

    def test_raises_for_unsupported_type(self) -> None:
        registry = SourceRegistry([TxtParser()])
        raw = _make_raw(source_type="pdf", mime_type="application/pdf")
        with pytest.raises(ValueError, match="No registered parser"):
            registry.resolve(raw)

    def test_empty_registry_raises(self) -> None:
        registry = SourceRegistry([])
        raw = _make_raw()
        with pytest.raises(ValueError):
            registry.resolve(raw)

    def test_first_matching_parser_wins(self) -> None:
        parser_a = TxtParser()
        parser_b = TxtParser()
        registry = SourceRegistry([parser_a, parser_b])
        raw = _make_raw(source_type="txt")
        assert registry.resolve(raw) is parser_a

    def test_supported_source_types_includes_txt(self) -> None:
        registry = SourceRegistry([TxtParser()])
        assert "txt" in registry.supported_source_types()

    def test_supported_source_types_empty_when_no_parsers(self) -> None:
        registry = SourceRegistry([])
        assert registry.supported_source_types() == set()

    def test_supported_source_types_union_of_parsers(self) -> None:
        class FakeHtmlParser:
            def supports(self, source_type: str, mime_type: str | None = None) -> bool:
                return source_type == "html"

            def supported_source_types(self) -> set[str]:
                return {"html"}

            def parse(self, raw_document: RawDocument) -> list:
                return []

        registry = SourceRegistry([TxtParser(), FakeHtmlParser()])  # type: ignore[list-item]
        types = registry.supported_source_types()
        assert "txt" in types
        assert "html" in types

    def test_parser_without_supported_source_types_skipped(self) -> None:
        class MinimalParser:
            def supports(self, source_type: str, mime_type: str | None = None) -> bool:
                return source_type == "csv"

            def parse(self, raw_document: RawDocument) -> list:
                return []

        registry = SourceRegistry([MinimalParser()])  # type: ignore[list-item]
        # Should not raise; just returns empty set
        types = registry.supported_source_types()
        assert isinstance(types, set)


# ---------------------------------------------------------------------------
# TxtParser
# ---------------------------------------------------------------------------


class TestTxtParser:
    # ---- supports() --------------------------------------------------------

    def test_supports_txt_source_type(self) -> None:
        assert TxtParser().supports("txt") is True

    def test_supports_txt_uppercase(self) -> None:
        assert TxtParser().supports("TXT") is True

    def test_supports_text_plain_mime(self) -> None:
        assert TxtParser().supports("unknown", "text/plain") is True

    def test_does_not_support_html(self) -> None:
        assert TxtParser().supports("html", "text/html") is False

    def test_does_not_support_pdf(self) -> None:
        assert TxtParser().supports("pdf", "application/pdf") is False

    def test_supports_returns_false_without_mime(self) -> None:
        assert TxtParser().supports("pdf") is False

    # ---- supported_source_types() ------------------------------------------

    def test_supported_source_types_contains_txt(self) -> None:
        assert "txt" in TxtParser().supported_source_types()

    # ---- parse() - happy path ----------------------------------------------

    def test_parse_returns_single_document(self) -> None:
        raw = _make_raw(content=b"Hello, Stripe!")
        docs = TxtParser().parse(raw)
        assert len(docs) == 1

    def test_parse_text_matches_decoded_content(self) -> None:
        raw = _make_raw(content=b"Payment accepted.")
        doc = TxtParser().parse(raw)[0]
        assert doc.text == "Payment accepted."

    def test_parse_preserves_source_type(self) -> None:
        raw = _make_raw(content=b"text", source_type="txt")
        doc = TxtParser().parse(raw)[0]
        assert doc.source_type == "txt"

    def test_parse_preserves_source_path(self) -> None:
        raw = _make_raw(content=b"text", source_path="/data/stripe.txt")
        doc = TxtParser().parse(raw)[0]
        assert doc.source_path == "/data/stripe.txt"

    def test_parse_preserves_source_name(self) -> None:
        raw = _make_raw(content=b"text", source_name="stripe.txt")
        doc = TxtParser().parse(raw)[0]
        assert doc.source_name == "stripe.txt"

    def test_parse_preserves_mime_type(self) -> None:
        raw = _make_raw(content=b"text", mime_type="text/plain")
        doc = TxtParser().parse(raw)[0]
        assert doc.source_mime_type == "text/plain"

    def test_parse_preserves_metadata(self) -> None:
        raw = RawDocument(
            source_type="txt",
            content=b"text",
            metadata={"file_size": 4, "extension": ".txt"},
        )
        doc = TxtParser().parse(raw)[0]
        assert doc.metadata["file_size"] == 4

    def test_parse_processing_stage_is_parsed(self) -> None:
        raw = _make_raw(content=b"content")
        doc = TxtParser().parse(raw)[0]
        assert doc.processing_stage == DocumentProcessingStage.PARSED

    def test_parse_id_starts_with_doc(self) -> None:
        raw = _make_raw(content=b"content")
        doc = TxtParser().parse(raw)[0]
        assert doc.id.startswith("doc_")

    def test_parse_content_hash_is_sha256_hex(self) -> None:
        raw = _make_raw(content=b"content")
        doc = TxtParser().parse(raw)[0]
        assert len(doc.content_hash) == 64
        assert all(c in "0123456789abcdef" for c in doc.content_hash)

    def test_parse_id_deterministic(self) -> None:
        raw = _make_raw(content=b"same content")
        id1 = TxtParser().parse(raw)[0].id
        id2 = TxtParser().parse(raw)[0].id
        assert id1 == id2

    def test_parse_different_content_different_id(self) -> None:
        raw_a = _make_raw(content=b"content A")
        raw_b = _make_raw(content=b"content B")
        id_a = TxtParser().parse(raw_a)[0].id
        id_b = TxtParser().parse(raw_b)[0].id
        assert id_a != id_b

    def test_parse_created_at_is_utc(self) -> None:

        raw = _make_raw(content=b"text")
        doc = TxtParser().parse(raw)[0]
        assert doc.created_at.tzinfo is not None
        assert doc.created_at.utcoffset().total_seconds() == 0  # type: ignore[union-attr]

    # ---- title resolution --------------------------------------------------

    def test_title_defaults_to_source_name(self) -> None:
        raw = _make_raw(content=b"text", source_name="guide.txt", source_path="/a/guide.txt")
        doc = TxtParser().parse(raw)[0]
        assert doc.title == "guide.txt"

    def test_title_falls_back_to_path_filename(self) -> None:
        raw = RawDocument(
            source_type="txt",
            content=b"text",
            source_path="/data/stripe_guide.txt",
        )
        doc = TxtParser().parse(raw)[0]
        assert doc.title == "stripe_guide.txt"

    def test_title_falls_back_to_default_string(self) -> None:
        raw = RawDocument(source_type="txt", content=b"text")
        doc = TxtParser().parse(raw)[0]
        assert doc.title == "Untitled TXT document"

    # ---- line ending normalisation -----------------------------------------

    def test_crlf_normalised_to_lf(self) -> None:
        raw = _make_raw(content=b"line1\r\nline2\r\n")
        doc = TxtParser().parse(raw)[0]
        assert "\r" not in doc.text
        # Document model strips surrounding whitespace, so trailing \n is removed.
        assert doc.text == "line1\nline2"

    def test_cr_only_normalised_to_lf(self) -> None:
        raw = _make_raw(content=b"line1\rline2")
        doc = TxtParser().parse(raw)[0]
        assert doc.text == "line1\nline2"

    # ---- UTF-8 fallback ----------------------------------------------------

    def test_non_utf8_bytes_decoded_with_replacement(self) -> None:
        bad_bytes = b"Hello \xff World"
        raw = _make_raw(content=bad_bytes)
        doc = TxtParser().parse(raw)[0]
        assert "Hello" in doc.text
        assert "World" in doc.text

    # ---- empty content raises ----------------------------------------------

    def test_empty_content_raises(self) -> None:
        # RawDocument already rejects empty bytes, so we use whitespace-only
        raw = _make_raw(content=b"   \n\t  ")
        with pytest.raises(ValueError, match="empty or whitespace"):
            TxtParser().parse(raw)

    def test_newlines_only_raises(self) -> None:
        raw = _make_raw(content=b"\n\n\n")
        with pytest.raises(ValueError, match="empty or whitespace"):
            TxtParser().parse(raw)
