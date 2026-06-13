"""Conservative TextCleaner implementing the Cleaner domain protocol.

Applies a deterministic, auditable sequence of cleaning steps to a parsed
Document and returns a new Document with cleaned text, an updated content
hash, and structured cleaning metadata stored under ``metadata["cleaning"]``.
"""

from __future__ import annotations

from typing import Any

from app.domain.models.document import Document, DocumentProcessingStage
from app.infrastructure.cleaning.boilerplate import remove_common_boilerplate_lines
from app.infrastructure.cleaning.html_cleaner import remove_html_artifacts
from app.infrastructure.cleaning.normalizers import (
    normalize_intraline_whitespace,
    normalize_newlines,
    normalize_unicode,
    remove_duplicate_blank_lines,
    remove_nearby_duplicate_lines,
    strip_lines,
)
from app.utils.hashing import sha256_text


def _non_whitespace_count(text: str) -> int:
    """Return the number of non-whitespace characters in *text*."""
    return sum(1 for c in text if not c.isspace())


class TextCleaner:
    """Conservative document cleaner that implements the ``Cleaner`` protocol.

    Cleans a parsed ``Document`` by applying up to nine deterministic steps
    in a fixed order:

    1. Residual HTML artifact removal (optional)
    2. Unicode normalization (optional)
    3. Newline normalization (always)
    4. Line-edge stripping (always)
    5. Conservative boilerplate line removal (optional)
    6. Nearby duplicate-line removal (optional)
    7. Consecutive blank-line reduction (always)
    8. Intraline whitespace normalization (optional)
    9. Final strip (always)

    Each optional step can be disabled at construction time.  The cleaner
    never mutates the input document — it always returns a new ``Document``
    instance created with ``model_copy``.

    Cleaning metadata is recorded under ``metadata["cleaning"]`` in the
    returned document.
    """

    CLEANER_NAME: str = "TextCleaner"

    def __init__(
        self,
        *,
        normalize_unicode_enabled: bool = True,
        remove_html_artifacts_enabled: bool = True,
        remove_boilerplate_enabled: bool = True,
        normalize_whitespace_enabled: bool = True,
        remove_duplicate_lines_enabled: bool = True,
        duplicate_line_window_size: int = 5,
        max_blank_lines: int = 1,
    ) -> None:
        """Initialise the cleaner with step toggles and tuning parameters.

        Args:
            normalize_unicode_enabled: Apply NFC unicode normalization.
            remove_html_artifacts_enabled: Strip residual HTML tags and blocks.
            remove_boilerplate_enabled: Remove standalone boilerplate lines.
            normalize_whitespace_enabled: Collapse multiple spaces within lines.
            remove_duplicate_lines_enabled: Remove nearby duplicate lines.
            duplicate_line_window_size: Rolling window size for duplicate
                detection.  Must be >= 1.
            max_blank_lines: Maximum consecutive blank lines allowed.
                Must be >= 0.

        Raises:
            ValueError: If ``duplicate_line_window_size < 1`` or
                ``max_blank_lines < 0``.
        """
        if duplicate_line_window_size < 1:
            raise ValueError(
                "duplicate_line_window_size must be >= 1, "
                f"got {duplicate_line_window_size}."
            )
        if max_blank_lines < 0:
            raise ValueError(
                f"max_blank_lines must be >= 0, got {max_blank_lines}."
            )

        self.normalize_unicode_enabled = normalize_unicode_enabled
        self.remove_html_artifacts_enabled = remove_html_artifacts_enabled
        self.remove_boilerplate_enabled = remove_boilerplate_enabled
        self.normalize_whitespace_enabled = normalize_whitespace_enabled
        self.remove_duplicate_lines_enabled = remove_duplicate_lines_enabled
        self.duplicate_line_window_size = duplicate_line_window_size
        self.max_blank_lines = max_blank_lines

    # ------------------------------------------------------------------
    # Cleaner protocol
    # ------------------------------------------------------------------

    def clean(self, document: Document) -> Document:
        """Clean *document* and return a new ``Document`` with cleaned text.

        Args:
            document: Parsed document.  Its ``text`` field must be non-empty
                after stripping.

        Returns:
            New ``Document`` with:
            - ``text`` replaced by the cleaned content;
            - ``content_hash`` updated to SHA-256 of the cleaned text;
            - ``processing_stage`` set to ``DocumentProcessingStage.CLEANED``;
            - ``metadata["cleaning"]`` populated with quality statistics.

        Raises:
            ValueError: If ``document.text`` is empty/whitespace-only, or if
                the cleaning steps produce empty output.
        """
        if not document.text.strip():
            raise ValueError(
                f"Document '{document.id}' has empty or whitespace-only text "
                "and cannot be cleaned."
            )

        original_text = document.text
        original_char_count = len(original_text)
        original_non_ws = _non_whitespace_count(original_text)
        original_hash = document.content_hash

        text = original_text
        html_artifacts_removed = 0
        boilerplate_lines_removed = 0
        duplicate_lines_removed = 0

        # Step 1: residual HTML artifact cleanup
        if self.remove_html_artifacts_enabled:
            text, html_artifacts_removed = remove_html_artifacts(text)

        # Step 2: unicode normalization
        if self.normalize_unicode_enabled:
            text = normalize_unicode(text)

        # Step 3: newline normalization (always)
        text = normalize_newlines(text)

        # Step 4: strip line edges (always)
        text = strip_lines(text)

        # Step 5: conservative boilerplate removal
        if self.remove_boilerplate_enabled:
            text, boilerplate_lines_removed = remove_common_boilerplate_lines(text)

        # Step 6: nearby duplicate-line removal
        if self.remove_duplicate_lines_enabled:
            text, duplicate_lines_removed = remove_nearby_duplicate_lines(
                text, window_size=self.duplicate_line_window_size
            )

        # Step 7: consecutive blank-line reduction (always)
        text = remove_duplicate_blank_lines(text, max_blank_lines=self.max_blank_lines)

        # Step 8: intraline whitespace normalization
        if self.normalize_whitespace_enabled:
            text = normalize_intraline_whitespace(text)

        # Step 9: final strip (always)
        text = text.strip()

        if not text:
            raise ValueError(
                f"Document '{document.id}' produced empty text after cleaning."
            )

        cleaned_char_count = len(text)
        cleaned_non_ws = _non_whitespace_count(text)
        cleaned_hash = sha256_text(text)

        length_ratio = (
            cleaned_char_count / original_char_count if original_char_count > 0 else 1.0
        )

        possible_overcleaning = (
            cleaned_non_ws < 0.10 * original_non_ws if original_non_ws > 0 else False
        )

        source_type_lower = document.source_type.lower()
        looks_html = "html" in source_type_lower or "htm" in source_type_lower
        possible_undercleaning = (
            looks_html
            and self.remove_html_artifacts_enabled
            and original_non_ws > 0
            and cleaned_non_ws > 0.95 * original_non_ws
        )

        warnings: list[str] = []
        if possible_overcleaning:
            warnings.append(
                "possible_overcleaning: cleaned non-whitespace < 10% of original"
            )
        if possible_undercleaning:
            warnings.append(
                "possible_undercleaning: HTML-like source still >95% of original "
                "non-whitespace length after artifact removal"
            )

        cleaning_meta: dict[str, Any] = {
            "cleaner_name": self.CLEANER_NAME,
            "enabled_steps": self._enabled_steps(),
            "original_char_count": original_char_count,
            "cleaned_char_count": cleaned_char_count,
            "original_non_whitespace_char_count": original_non_ws,
            "cleaned_non_whitespace_char_count": cleaned_non_ws,
            "length_ratio": length_ratio,
            "html_artifacts_removed": html_artifacts_removed,
            "boilerplate_lines_removed": boilerplate_lines_removed,
            "duplicate_lines_removed": duplicate_lines_removed,
            "possible_overcleaning": possible_overcleaning,
            "possible_undercleaning": possible_undercleaning,
            "warnings": warnings,
            "original_content_hash": original_hash,
            "cleaned_content_hash": cleaned_hash,
        }

        updated_metadata = {**document.metadata, "cleaning": cleaning_meta}

        return document.model_copy(
            update={
                "text": text,
                "content_hash": cleaned_hash,
                "processing_stage": DocumentProcessingStage.CLEANED,
                "metadata": updated_metadata,
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enabled_steps(self) -> list[str]:
        """Return the list of cleaning step names in execution order."""
        steps: list[str] = []
        if self.remove_html_artifacts_enabled:
            steps.append("remove_html_artifacts")
        if self.normalize_unicode_enabled:
            steps.append("normalize_unicode")
        steps.append("normalize_newlines")
        steps.append("strip_lines")
        if self.remove_boilerplate_enabled:
            steps.append("remove_boilerplate")
        if self.remove_duplicate_lines_enabled:
            steps.append("remove_nearby_duplicate_lines")
        steps.append("remove_duplicate_blank_lines")
        if self.normalize_whitespace_enabled:
            steps.append("normalize_intraline_whitespace")
        steps.append("final_strip")
        return steps
