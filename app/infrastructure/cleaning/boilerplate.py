"""Conservative boilerplate-line removal for RAG document cleaning.

Only removes obvious standalone navigation/footer phrases that are exact
(case-insensitive) matches for a curated set of short patterns.  Lines
that merely *contain* these words as part of a meaningful sentence are
left untouched.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Boilerplate phrase set
# ---------------------------------------------------------------------------

# Every entry is a stripped, lower-cased phrase.  A document line is removed
# only when its stripped, lower-cased form is an exact member of this set.
_BOILERPLATE_PHRASES: frozenset[str] = frozenset(
    {
        "sign in",
        "log in",
        "contact sales",
        "contact us",
        "cookie settings",
        "privacy",
        "privacy policy",
        "terms",
        "terms of service",
        "documentation",
        "docs",
        "support",
        "pricing",
        "resources",
        "company",
        "developers",
    }
)

# Lines longer than this threshold are never removed even if they start with
# a boilerplate word, since they are almost certainly meaningful sentences.
_MAX_BOILERPLATE_LINE_LENGTH = 60


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def remove_common_boilerplate_lines(text: str) -> tuple[str, int]:
    """Remove standalone navigation/footer boilerplate lines from *text*.

    A line is removed only when ALL of the following hold:
    - it is non-empty after stripping;
    - its stripped length is <= ``_MAX_BOILERPLATE_LINE_LENGTH``;
    - its stripped, lower-cased content is an exact member of
      ``_BOILERPLATE_PHRASES``.

    Lines that merely contain a boilerplate word inside a longer sentence
    (e.g. "Contact our support team for pricing details.") are preserved.

    Args:
        text: Input string with ``\\n`` line endings.

    Returns:
        Tuple of (cleaned_text, removed_line_count).
    """
    lines = text.split("\n")
    result: list[str] = []
    removed = 0

    for line in lines:
        stripped = line.strip()
        normalized = stripped.lower()

        if (
            stripped
            and len(stripped) <= _MAX_BOILERPLATE_LINE_LENGTH
            and normalized in _BOILERPLATE_PHRASES
        ):
            removed += 1
        else:
            result.append(line)

    return "\n".join(result), removed
