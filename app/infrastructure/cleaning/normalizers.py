"""Pure text normalization functions for the RAG cleaning pipeline.

All functions are stateless and operate solely on strings using the Python
standard library.  They preserve document structure (headings, lists, FAQ
markers, page markers) while correcting encoding artefacts and whitespace
irregularities.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Structural-line detection
# ---------------------------------------------------------------------------

_STRUCTURAL_RE = re.compile(
    r"^(?:"
    r"\s*#{1,6}\s"  # Markdown headings:  ## Section
    r"|\s*[-*+]\s"  # Bullet lists:       - item  * item  + item
    r"|\s*\d+[.)]\s"  # Numbered lists:    1. step  2) step
    r"|\s*[QA]:\s*"  # FAQ short form:     Q:  A:
    r"|\s*(?:Question|Answer):\s*"  # FAQ long form
    r"|\s*\[Page\s+\d+\]"  # Page markers:  [Page 1]
    r")",
    re.IGNORECASE,
)


def _is_structural(line: str) -> bool:
    """Return True for lines that carry structural meaning and must not be
    deduplicated even when they repeat within a nearby window."""
    return bool(_STRUCTURAL_RE.match(line))


# ---------------------------------------------------------------------------
# Public normalizer functions
# ---------------------------------------------------------------------------


def normalize_unicode(text: str) -> str:
    """Return the NFC-normalized form of *text*.

    NFC normalization decomposes then recomposes characters, producing a
    canonical, byte-stable representation without altering visible content.

    Args:
        text: Input string (may be empty).

    Returns:
        NFC-normalized string.
    """
    return unicodedata.normalize("NFC", text)


def normalize_newlines(text: str) -> str:
    """Normalize all line endings to LF (``\\n``).

    Replaces Windows-style CRLF and legacy Mac-style CR endings with a single
    LF character so the rest of the pipeline can rely on ``\\n`` as the
    universal line separator.

    Args:
        text: Input string.

    Returns:
        String with only ``\\n`` line endings.
    """
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    return text


def strip_lines(text: str) -> str:
    """Remove trailing whitespace from every line.

    Leading whitespace is preserved so that indented lists and code-like
    blocks retain their visual indentation.

    Args:
        text: Input string with ``\\n`` line endings.

    Returns:
        String with trailing whitespace stripped per line.
    """
    return "\n".join(line.rstrip() for line in text.split("\n"))


def remove_duplicate_blank_lines(text: str, max_blank_lines: int = 1) -> str:
    """Collapse consecutive blank lines to at most *max_blank_lines*.

    A blank line is any line whose content after stripping is empty.

    Args:
        text: Input string with ``\\n`` line endings.
        max_blank_lines: Maximum number of consecutive blank lines to keep.
            Must be >= 0.

    Returns:
        String with consecutive blank-line runs shortened.
    """
    lines = text.split("\n")
    result: list[str] = []
    blank_run = 0

    for line in lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= max_blank_lines:
                result.append(line)
        else:
            blank_run = 0
            result.append(line)

    return "\n".join(result)


def normalize_intraline_whitespace(text: str) -> str:
    """Collapse multiple consecutive spaces or tabs within each line.

    Leading whitespace is preserved intact so indented lists and code blocks
    are not broken.  Paragraph boundaries (blank lines) are untouched.

    Args:
        text: Input string with ``\\n`` line endings.

    Returns:
        String with intraline whitespace normalized.
    """
    result: list[str] = []

    for line in text.split("\n"):
        if not line.strip():
            result.append(line)
            continue

        leading_len = len(line) - len(line.lstrip())
        leading_ws = line[:leading_len]
        content = line[leading_len:]

        content = re.sub(r" {2,}", " ", content)
        content = re.sub(r"\t{2,}", "\t", content)

        result.append(leading_ws + content)

    return "\n".join(result)


def remove_nearby_duplicate_lines(text: str, window_size: int = 5) -> tuple[str, int]:
    """Remove identical non-empty lines that appear within a rolling window.

    Only lines that are not structurally significant (headings, bullets,
    numbered lists, FAQ markers, page markers) are subject to deduplication.
    Structural lines always pass through unchanged.

    The window tracks the last *window_size* non-empty lines seen.  A
    candidate line is removed only when the same stripped content already
    exists in the current window.  Lines that appear far apart in the
    document (beyond *window_size* non-empty lines apart) are preserved.

    Args:
        text: Input string with ``\\n`` line endings.
        window_size: Number of recent non-empty lines to check for duplicates.
            Must be >= 1.

    Returns:
        Tuple of (cleaned_text, removed_count).
    """
    lines = text.split("\n")
    result: list[str] = []
    recent: list[str] = []
    removed = 0

    for line in lines:
        stripped = line.strip()

        if not stripped:
            result.append(line)
            continue

        if _is_structural(line):
            result.append(line)
            recent.append(stripped)
            if len(recent) > window_size:
                recent.pop(0)
            continue

        if stripped in recent:
            removed += 1
        else:
            result.append(line)
            recent.append(stripped)
            if len(recent) > window_size:
                recent.pop(0)

    return "\n".join(result), removed
