"""Unit tests for app/utils/hashing.py and app/utils/ids.py."""

import pytest
from app.utils.hashing import sha256_bytes, sha256_text
from app.utils.ids import make_document_id

# ---------------------------------------------------------------------------
# sha256_text
# ---------------------------------------------------------------------------


class TestSha256Text:
    def test_returns_64_hex_chars(self) -> None:
        digest = sha256_text("hello")
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_deterministic(self) -> None:
        assert sha256_text("stripe") == sha256_text("stripe")

    def test_different_inputs_differ(self) -> None:
        assert sha256_text("a") != sha256_text("b")

    def test_known_value(self) -> None:
        # SHA-256("hello") is a known constant
        expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        assert sha256_text("hello") == expected

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            sha256_text("")

    def test_unicode_input(self) -> None:
        digest = sha256_text("café")
        assert len(digest) == 64


# ---------------------------------------------------------------------------
# sha256_bytes
# ---------------------------------------------------------------------------


class TestSha256Bytes:
    def test_returns_64_hex_chars(self) -> None:
        digest = sha256_bytes(b"hello")
        assert len(digest) == 64

    def test_deterministic(self) -> None:
        assert sha256_bytes(b"data") == sha256_bytes(b"data")

    def test_different_inputs_differ(self) -> None:
        assert sha256_bytes(b"\x00") != sha256_bytes(b"\x01")

    def test_text_and_bytes_consistent(self) -> None:
        # sha256_text encodes as UTF-8 before hashing
        assert sha256_text("hello") == sha256_bytes(b"hello")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            sha256_bytes(b"")


# ---------------------------------------------------------------------------
# make_document_id
# ---------------------------------------------------------------------------


class TestMakeDocumentId:
    def test_has_doc_prefix(self) -> None:
        doc_id = make_document_id("/path/to/file.txt", "abc123")
        assert doc_id.startswith("doc_")

    def test_deterministic(self) -> None:
        id1 = make_document_id("/path/file.txt", "hash1")
        id2 = make_document_id("/path/file.txt", "hash1")
        assert id1 == id2

    def test_different_paths_differ(self) -> None:
        id1 = make_document_id("/path/a.txt", "hash1")
        id2 = make_document_id("/path/b.txt", "hash1")
        assert id1 != id2

    def test_different_hashes_differ(self) -> None:
        id1 = make_document_id("/path/file.txt", "hash1")
        id2 = make_document_id("/path/file.txt", "hash2")
        assert id1 != id2

    def test_length_is_stable(self) -> None:
        id1 = make_document_id("/a", "x")
        id2 = make_document_id("/very/long/path/to/some/document.txt", "y" * 64)
        # Both should have the same fixed length (prefix + 24 hex chars = 28)
        assert len(id1) == len(id2) == 28

    def test_empty_identity_raises(self) -> None:
        with pytest.raises(ValueError, match="source_identity"):
            make_document_id("", "hash1")

    def test_whitespace_only_identity_raises(self) -> None:
        with pytest.raises(ValueError, match="source_identity"):
            make_document_id("   ", "hash1")

    def test_empty_hash_raises(self) -> None:
        with pytest.raises(ValueError, match="content_hash"):
            make_document_id("/path/file.txt", "")

    def test_whitespace_only_hash_raises(self) -> None:
        with pytest.raises(ValueError, match="content_hash"):
            make_document_id("/path/file.txt", "  ")

    def test_only_hex_chars_in_suffix(self) -> None:
        doc_id = make_document_id("/some/path", "somehash")
        suffix = doc_id[len("doc_"):]
        assert all(c in "0123456789abcdef" for c in suffix)
