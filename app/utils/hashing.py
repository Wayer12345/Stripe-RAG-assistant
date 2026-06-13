"""Generic hashing utilities using the Python standard library only."""

import hashlib


def sha256_text(text: str) -> str:
    """Return lowercase hex SHA-256 digest of a UTF-8 encoded string.

    Args:
        text: Non-empty string to hash.

    Returns:
        64-character lowercase hexadecimal string.

    Raises:
        ValueError: If text is empty.
    """
    if not text:
        raise ValueError("text must not be empty.")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return lowercase hex SHA-256 digest of raw bytes.

    Args:
        data: Non-empty bytes to hash.

    Returns:
        64-character lowercase hexadecimal string.

    Raises:
        ValueError: If data is empty.
    """
    if not data:
        raise ValueError("data must not be empty.")
    return hashlib.sha256(data).hexdigest()
