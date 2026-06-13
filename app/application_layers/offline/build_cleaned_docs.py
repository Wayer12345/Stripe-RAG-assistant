"""Build cleaned-document artifacts for the offline cleaning stage."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.domain.models.document import Document
from app.infrastructure.cleaning.cleaner_factory import create_cleaner
from app.infrastructure.cleaning.text_cleaner import TextCleaner
from app.infrastructure.storage.artifact_paths import (
    CleaningArtifactPaths,
    resolve_cleaning_artifact_paths,
)
from app.infrastructure.storage.jsonl_store import count_jsonl, read_jsonl, write_jsonl
from app.infrastructure.storage.manifest_store import build_base_manifest, write_manifest
from app.infrastructure.storage.pipeline_records import (
    build_document_error_record,
    compute_layer_status,
)
from app.utils.config import (
    Settings,
    load_settings,
    resolve_config_dir_and_path,
    to_optional_path,
    validate_positive_limit,
)
from app.utils.constants import (
    ARTIFACT_SCHEMA_VERSION,
    STAGE_CLEANING,
)
from app.utils.logging import get_logger, setup_logging
from app.utils.timing import finish_timed_run, start_timed_run

_STAGE_NAME = STAGE_CLEANING
logger = get_logger(__name__)


@dataclass(frozen=True)
class BuildCleanedDocsResult:
    """Summary of a cleaned-documents layer run."""

    run_id: str
    input_path: Path
    output_path: Path
    manifest_path: Path
    parsed_documents_total: int
    cleaned_documents_total: int
    failed_documents_total: int
    skipped_documents_total: int
    duration_ms: int


class BuildCleanedDocsLayer:
    """Offline layer that cleans parsed documents and writes cleaned artifacts."""

    @staticmethod
    def _build_cleaning_options(settings: Settings) -> dict[str, Any]:
        return {
            "normalize_unicode_enabled": settings.cleaning.steps.normalize_unicode,
            "remove_html_artifacts_enabled": settings.cleaning.steps.remove_html_artifacts,
            "remove_boilerplate_enabled": settings.cleaning.steps.remove_boilerplate,
            "normalize_whitespace_enabled": settings.cleaning.steps.normalize_whitespace,
            "remove_duplicate_lines_enabled": settings.cleaning.steps.remove_duplicate_lines,
            "duplicate_line_window_size": settings.cleaning.duplicate_lines.window_size,
            "max_blank_lines": settings.cleaning.blank_lines.max_blank_lines,
        }

    @staticmethod
    def _build_aggregate_stats(cleaned_documents: list[Document]) -> dict[str, Any]:
        original_char_total = 0
        cleaned_char_total = 0
        documents_with_warnings = 0
        warning_counts: dict[str, int] = {}

        for document in cleaned_documents:
            cleaning_meta = document.metadata.get("cleaning", {})
            if not isinstance(cleaning_meta, dict):
                continue

            original_char_total += int(cleaning_meta.get("original_char_count", 0))
            cleaned_char_total += int(cleaning_meta.get("cleaned_char_count", 0))

            warnings = cleaning_meta.get("warnings", [])
            if isinstance(warnings, list):
                if warnings:
                    documents_with_warnings += 1
                for warning in warnings:
                    if isinstance(warning, str):
                        warning_counts[warning] = warning_counts.get(warning, 0) + 1

        avg_length_ratio = (
            cleaned_char_total / original_char_total if original_char_total > 0 else 1.0
        )

        return {
            "total_original_char_count": original_char_total,
            "total_cleaned_char_count": cleaned_char_total,
            "avg_length_ratio": avg_length_ratio,
            "documents_with_warnings": documents_with_warnings,
            "warnings": warning_counts,
        }

    def __init__(
        self,
        *,
        config_path: Path | str = Path("configs/config.yaml"),
        input_path: Path | str | None = None,
        output_path: Path | str | None = None,
        manifest_path: Path | str | None = None,
        fail_fast: bool | None = None,
        limit: int | None = None,
    ) -> None:
        validate_positive_limit(limit)

        self._config_path = Path(config_path)
        self._input_path_override = to_optional_path(input_path)
        self._output_path_override = to_optional_path(output_path)
        self._manifest_path_override = to_optional_path(manifest_path)
        self._fail_fast_override = fail_fast
        self._limit = limit

    def run(self) -> BuildCleanedDocsResult:
        """Clean parsed documents, write cleaned JSONL + manifest, and return a result."""

        config_dir, resolved_config_path = resolve_config_dir_and_path(self._config_path)

        settings = load_settings(config_dir)
        setup_logging(settings)

        artifact_paths: CleaningArtifactPaths = resolve_cleaning_artifact_paths(
            settings,
            input_path_override=self._input_path_override,
            output_path_override=self._output_path_override,
            manifest_path_override=self._manifest_path_override,
        )

        fail_fast = (
            self._fail_fast_override
            if self._fail_fast_override is not None
            else not settings.ingestion.failure_policy.continue_on_clean_error
        )

        timed_run = start_timed_run(_STAGE_NAME)

        logger.info(
            "Starting cleaning layer: run_id=%s stage=%s config_path=%s fail_fast=%s limit=%s",
            timed_run.run_id,
            _STAGE_NAME,
            resolved_config_path,
            fail_fast,
            self._limit,
        )
        logger.info(
            "Resolved cleaning paths: input_path=%s output_path=%s manifest_path=%s",
            artifact_paths.input_path,
            artifact_paths.output_path,
            artifact_paths.manifest_path,
        )
        cleaning_options = self._build_cleaning_options(settings)
        logger.info(
            "Cleaning options: cleaner_name=%s normalize_unicode=%s remove_html_artifacts=%s remove_boilerplate=%s normalize_whitespace=%s remove_duplicate_lines=%s duplicate_line_window_size=%s max_blank_lines=%s",
            TextCleaner.CLEANER_NAME,
            cleaning_options["normalize_unicode_enabled"],
            cleaning_options["remove_html_artifacts_enabled"],
            cleaning_options["remove_boilerplate_enabled"],
            cleaning_options["normalize_whitespace_enabled"],
            cleaning_options["remove_duplicate_lines_enabled"],
            cleaning_options["duplicate_line_window_size"],
            cleaning_options["max_blank_lines"],
        )

        total_available = (
            count_jsonl(artifact_paths.input_path) if self._limit is not None else None
        )
        parsed_documents = read_jsonl(artifact_paths.input_path, Document, limit=self._limit)

        logger.info("Loaded parsed documents: count=%s limit=%s", len(parsed_documents), self._limit)
        if not parsed_documents:
            logger.warning("Zero parsed documents loaded: input_path=%s", artifact_paths.input_path)
        logger.info(
            "Parsed documents by source_type: counts=%s",
            dict(Counter(doc.source_type for doc in parsed_documents)),
        )
        logger.info("Documents to process: count=%s", len(parsed_documents))

        cleaner = create_cleaner(settings)

        cleaned_documents: list[Document] = []
        errors: list[dict[str, Any]] = []
        processed_documents_total = 0

        for parsed_document in parsed_documents:
            try:
                cleaned = cleaner.clean(parsed_document)

                cleaning_meta = (
                    cleaned.metadata.get("cleaning", {})
                    if isinstance(cleaned.metadata, dict)
                    else {}
                )
                warning_labels = (
                    cleaning_meta.get("warnings", []) if isinstance(cleaning_meta, dict) else []
                )

                if isinstance(warning_labels, list) and warning_labels:
                    logger.warning(
                        "Document cleaning warnings: document_id=%s source_path=%s warnings=%s",
                        cleaned.id,
                        cleaned.source_path,
                        warning_labels,
                    )
                    if "possible_overcleaning" in warning_labels:
                        logger.warning(
                            "Possible overcleaning detected: document_id=%s source_path=%s",
                            cleaned.id,
                            cleaned.source_path,
                        )
                    if "possible_undercleaning" in warning_labels:
                        logger.warning(
                            "Possible undercleaning detected: document_id=%s source_path=%s",
                            cleaned.id,
                            cleaned.source_path,
                        )
                cleaned_documents.append(cleaned)

            except Exception as exc:
                logger.error(
                    "Cleaning failure: document_id=%s source_path=%s source_type=%s title=%s error_type=%s error_message=%s",
                    parsed_document.id,
                    parsed_document.source_path,
                    parsed_document.source_type,
                    parsed_document.title,
                    type(exc).__name__,
                    str(exc),
                )

                errors.append(build_document_error_record(document=parsed_document, exc=exc))
                processed_documents_total += 1

                if fail_fast:
                    break
                continue

            processed_documents_total += 1

        actual_total = total_available if total_available is not None else len(parsed_documents)
        skipped_documents_total = actual_total - processed_documents_total

        write_jsonl(artifact_paths.output_path, cleaned_documents)
        logger.info("Wrote cleaned documents artifact: path=%s", artifact_paths.output_path)

        finished_at, duration_ms = finish_timed_run(timed_run)
        stats = self._build_aggregate_stats(cleaned_documents)
        avg_original_chars = (
            stats["total_original_char_count"] / len(cleaned_documents)
            if cleaned_documents
            else 0.0
        )

        avg_cleaned_chars = (
            stats["total_cleaned_char_count"] / len(cleaned_documents) if cleaned_documents else 0.0
        )

        failure_rate = len(errors) / len(parsed_documents) if parsed_documents else 0.0

        logger.info(
            "Cleaning stats: original_chars=%s cleaned_chars=%s avg_original_chars=%.2f avg_cleaned_chars=%.2f avg_length_ratio=%.4f documents_with_warnings=%s failures=%s failure_rate=%.3f",
            stats["total_original_char_count"],
            stats["total_cleaned_char_count"],
            avg_original_chars,
            avg_cleaned_chars,
            stats["avg_length_ratio"],
            stats["documents_with_warnings"],
            len(errors),
            failure_rate,
        )
        if stats["warnings"]:
            logger.info("Cleaning warning labels aggregate: counts=%s", stats["warnings"])
        if len(cleaned_documents) == 0:
            logger.warning("Zero cleaned documents produced for cleaning stage.")

        manifest: dict[str, Any] = {
            **build_base_manifest(
                run_id=timed_run.run_id,
                stage=_STAGE_NAME,
                started_at=timed_run.started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
                config_path=resolved_config_path,
                errors=errors,
                schema_version=ARTIFACT_SCHEMA_VERSION,
            ),
            "input_path": str(artifact_paths.input_path),
            "output_path": str(artifact_paths.output_path),
            "parsed_documents_total": actual_total,
            "cleaned_documents_total": len(cleaned_documents),
            "failed_documents_total": len(errors),
            "skipped_documents_total": skipped_documents_total,
            "cleaner_name": TextCleaner.CLEANER_NAME,
            "cleaning_options": cleaning_options,
            **stats,
        }
        write_manifest(artifact_paths.manifest_path, manifest)
        logger.info("Wrote cleaning manifest: path=%s", artifact_paths.manifest_path)
        status = compute_layer_status(
            success_count=len(cleaned_documents),
            error_count=len(errors),
        )
        logger.info(
            "Finished cleaning layer: status=%s duration_ms=%s parsed_documents_total=%s cleaned_documents_total=%s failed_documents_total=%s skipped_documents_total=%s",
            status,
            duration_ms,
            actual_total,
            len(cleaned_documents),
            len(errors),
            skipped_documents_total,
        )

        return BuildCleanedDocsResult(
            run_id=timed_run.run_id,
            input_path=artifact_paths.input_path,
            output_path=artifact_paths.output_path,
            manifest_path=artifact_paths.manifest_path,
            parsed_documents_total=actual_total,
            cleaned_documents_total=len(cleaned_documents),
            failed_documents_total=len(errors),
            skipped_documents_total=skipped_documents_total,
            duration_ms=duration_ms,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build cleaned documents from parsed JSONL artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to config.yaml or a config directory.",
    )
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--fail-fast", action="store_true", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    setup_logging()
    args = _build_arg_parser().parse_args()
    layer = BuildCleanedDocsLayer(
        config_path=args.config,
        input_path=args.input_path,
        output_path=args.output_path,
        manifest_path=args.manifest_path,
        fail_fast=args.fail_fast,
        limit=args.limit,
    )
    try:
        result = layer.run()
    except Exception:
        logger.exception("BuildCleanedDocsLayer failed")
        sys.exit(1)

    if args.fail_fast and result.failed_documents_total > 0:
        sys.exit(1)
    if result.cleaned_documents_total == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
