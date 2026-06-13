"""Residual HTML artifact cleanup for RAG document cleaning.

Removes leftover script/style/noscript/template blocks and stray HTML tags
from text that was not fully cleaned during parsing.  HTML entities are
unescaped to their unicode equivalents.

This module is intentionally conservative: it only removes clear technical
artifacts and never strips visible text content.
"""

from __future__ import annotations

import html
import re

# ---------------------------------------------------------------------------
# Compiled patterns for block-level HTML noise
# ---------------------------------------------------------------------------

_SCRIPT_RE = re.compile(
    r"<script[^>]*>.*?</script>",
    re.IGNORECASE | re.DOTALL,
)
_STYLE_RE = re.compile(
    r"<style[^>]*>.*?</style>",
    re.IGNORECASE | re.DOTALL,
)
_NOSCRIPT_RE = re.compile(
    r"<noscript[^>]*>.*?</noscript>",
    re.IGNORECASE | re.DOTALL,
)
_TEMPLATE_RE = re.compile(
    r"<template[^>]*>.*?</template>",
    re.IGNORECASE | re.DOTALL,
)

# Remaining simple / self-closing tags after block removal.
_TAG_RE = re.compile(r"<[^>]+>")

_BLOCK_PATTERNS = (_SCRIPT_RE, _STYLE_RE, _NOSCRIPT_RE, _TEMPLATE_RE)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def remove_html_artifacts(text: str) -> tuple[str, int]:
    """Remove residual HTML artifacts from *text* and unescape HTML entities.

    Steps applied in order:
    1. Strip ``<script>``, ``<style>``, ``<noscript>``, ``<template>`` blocks
       (including their content).
    2. Remove remaining simple / self-closing HTML tags.
    3. Unescape HTML entities (e.g. ``&amp;`` → ``&``, ``&lt;`` → ``<``).

    Args:
        text: Input string, possibly containing leftover HTML fragments.

    Returns:
        Tuple of (cleaned_text, approximate_artifact_count) where the count
        is the number of removed block-level elements plus individual tags.
    """
    removed_count = 0

    for pattern in _BLOCK_PATTERNS:
        matches = pattern.findall(text)
        removed_count += len(matches)
        text = pattern.sub("", text)

    remaining_tags = _TAG_RE.findall(text)
    removed_count += len(remaining_tags)
    text = _TAG_RE.sub("", text)

    text = html.unescape(text)

    return text, removed_count
