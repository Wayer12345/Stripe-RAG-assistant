"""Factory for configured TextCleaner implementations."""

from __future__ import annotations

from app.infrastructure.cleaning.text_cleaner import TextCleaner
from app.utils.config import Settings


def create_cleaner(settings: Settings) -> TextCleaner:
    """Create and return a TextCleaner configured from project settings.

    Args:
        settings: Project settings loaded via load_settings.

    Returns:
        A TextCleaner instance wired to the configured cleaning steps and parameters.
    """
    return TextCleaner(
        normalize_unicode_enabled=settings.cleaning.steps.normalize_unicode,
        remove_html_artifacts_enabled=settings.cleaning.steps.remove_html_artifacts,
        remove_boilerplate_enabled=settings.cleaning.steps.remove_boilerplate,
        normalize_whitespace_enabled=settings.cleaning.steps.normalize_whitespace,
        remove_duplicate_lines_enabled=settings.cleaning.steps.remove_duplicate_lines,
        duplicate_line_window_size=settings.cleaning.duplicate_lines.window_size,
        max_blank_lines=settings.cleaning.blank_lines.max_blank_lines,
    )
