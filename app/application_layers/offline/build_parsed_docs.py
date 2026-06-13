"""Build parsed-document artifacts for the offline ingestion stage."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.domain.models.document import Document
from app.infrastructure.loaders import FileLoader, SourceRegistry
from app.infrastructure.parsers import (
    CsvParser,
    DocxParser,
    HtmlParser,
    JsonParser,
    MarkdownParser,
    PdfParser,
    TxtParser,
)
from app.infrastructure.storage.artifact_paths import (
    IngestionArtifactPaths,
    resolve_ingestion_artifact_paths,
)
from app.infrastructure.storage.jsonl_store import write_jsonl
from app.infrastructure.storage.manifest_store import build_base_manifest, write_manifest
from app.utils.config import (
    apply_limit,
    load_settings,
    resolve_config_dir_and_path,
    to_optional_path,
    validate_positive_limit,
)
from app.infrastructure.storage.pipeline_records import (
    build_raw_document_error_record,
    compute_layer_status,
)
from app.utils.constants import (
    ARTIFACT_SCHEMA_VERSION,
    STAGE_INGESTION,
)
from app.utils.logging import get_logger, setup_logging
from app.utils.timing import finish_timed_run, start_timed_run

_STAGE_NAME = STAGE_INGESTION
logger = get_logger(__name__)


@dataclass(frozen=True)
class BuildParsedDocsResult:
    """Summary of a parsed-documents ingestion-layer run."""

    run_id: str
    input_dir: Path
    output_path: Path
    manifest_path: Path
    raw_documents_total: int
    parsed_documents_total: int
    failed_documents_total: int
    skipped_documents_total: int
    duration_ms: int


class BuildParsedDocsLayer:
    """Offline layer that parses raw files into parsed document artifacts."""

    @staticmethod
    def _build_parser_registry() -> SourceRegistry:
        return SourceRegistry(
            parsers=[
                TxtParser(),
                MarkdownParser(),
                HtmlParser(),
                PdfParser(),
                DocxParser(),
                JsonParser(),
                CsvParser(),
            ]
        )

    def __init__(
        self,
        *,
        config_path: Path | str = Path("configs/config.yaml"),
        input_dir: Path | str | None = None,
        output_path: Path | str | None = None,
        manifest_path: Path | str | None = None,
        fail_fast: bool | None = None,
        limit: int | None = None,
    ) -> None:
        validate_positive_limit(limit)

        self._config_path = Path(config_path)
        self._input_dir_override = to_optional_path(input_dir)
        self._output_path_override = to_optional_path(output_path)
        self._manifest_path_override = to_optional_path(manifest_path)
        self._fail_fast = fail_fast if fail_fast is not None else False
        self._limit = limit

    def run(self) -> BuildParsedDocsResult:
        """Parse raw documents, write parsed JSONL + ingestion manifest, and return a result."""

        config_dir, resolved_config_path = resolve_config_dir_and_path(self._config_path)

        settings = load_settings(config_dir)

        setup_logging(settings)

        artifact_paths: IngestionArtifactPaths = resolve_ingestion_artifact_paths(
            settings,
            input_dir_override=self._input_dir_override,
            output_path_override=self._output_path_override,
            manifest_path_override=self._manifest_path_override,
        )

        timed_run = start_timed_run(_STAGE_NAME)

        logger.info(
            "Starting ingestion layer: run_id=%s stage=%s config_path=%s fail_fast=%s limit=%s",
            timed_run.run_id,
            _STAGE_NAME,
            resolved_config_path,
            self._fail_fast,
            self._limit,
        )
        logger.info(
            "Resolved ingestion paths: input_dir=%s output_path=%s manifest_path=%s",
            artifact_paths.input_dir,
            artifact_paths.output_path,
            artifact_paths.manifest_path,
        )
        logger.info(
            "Ingestion options: recursive=%s supported_extensions=%s",
            settings.ingestion.recursive,
            sorted(set(settings.ingestion.supported_extensions)),
        )

        loader = FileLoader(
            input_dir=artifact_paths.input_dir,
            supported_extensions=set(settings.ingestion.supported_extensions),
            recursive=settings.ingestion.recursive,
        )

        registry = self._build_parser_registry()

        raw_documents = loader.load()

        logger.info("Raw documents loaded: total=%s", len(raw_documents))
        if not raw_documents:
            logger.warning("No raw documents found: input_dir=%s", artifact_paths.input_dir)
        source_type_counts = dict(Counter(doc.source_type for doc in raw_documents))
        logger.info("Raw documents by source_type: counts=%s", source_type_counts)
        raw_documents_to_process, skipped_documents_total = apply_limit(raw_documents, self._limit)
        if skipped_documents_total > 0:
            logger.warning(
                "Skipped documents due to limit: skipped=%s limit=%s",
                skipped_documents_total,
                self._limit,
            )
        logger.info(
            "Raw documents to process: total=%s",
            len(raw_documents_to_process),
        )

        parser_counts: dict[str, int] = {}
        parsed_documents: list[Document] = []
        errors: list[dict[str, Any]] = []
        failed_documents_total = 0
        processed_raw_documents_total = 0

        for raw_document in raw_documents_to_process:
            parser_name: str | None = None
            try:
                parser = registry.resolve(raw_document)

                parser_name = type(parser).__name__
                parser_counts[parser_name] = parser_counts.get(parser_name, 0) + 1

                parsed_from_source = parser.parse(raw_document)

                if not parsed_from_source:
                    logger.warning(
                        "Parser produced zero documents: source_path=%s source_name=%s source_type=%s parser_name=%s",
                        raw_document.source_path,
                        raw_document.source_name,
                        raw_document.source_type,
                        parser_name,
                    )

                parsed_documents.extend(parsed_from_source)

            except Exception as exc:
                failed_documents_total += 1

                if parser_name is None:
                    logger.error(
                        "Parser selection failure: source_path=%s source_name=%s source_type=%s error_type=%s error_message=%s",
                        raw_document.source_path,
                        raw_document.source_name,
                        raw_document.source_type,
                        type(exc).__name__,
                        str(exc),
                    )
                else:
                    logger.error(
                        "Parser failure: source_path=%s source_name=%s source_type=%s parser_name=%s error_type=%s error_message=%s",
                        raw_document.source_path,
                        raw_document.source_name,
                        raw_document.source_type,
                        parser_name,
                        type(exc).__name__,
                        str(exc),
                    )
                errors.append(
                    build_raw_document_error_record(
                        raw_document=raw_document,
                        parser_name=parser_name,
                        exc=exc,
                    )
                )

                if self._fail_fast:
                    logger.warning(
                        "Fail-fast triggered on parse failure: processed_documents=%s",
                        processed_raw_documents_total + 1,
                    )
                    break
            finally:
                processed_raw_documents_total += 1

        parsed_per_raw = (
            len(parsed_documents) / len(raw_documents_to_process)
            if raw_documents_to_process
            else 0.0
        )
        failure_rate = (
            failed_documents_total / len(raw_documents_to_process)
            if raw_documents_to_process
            else 0.0
        )

        logger.info(
            "Parsing complete: parsed_documents=%s failed_documents=%s skipped_documents=%s parsed_per_raw=%.3f failure_rate=%.3f",
            len(parsed_documents),
            failed_documents_total,
            skipped_documents_total,
            parsed_per_raw,
            failure_rate,
        )
        logger.info("Parser counts: parser_counts=%s", parser_counts)
        if len(parsed_documents) == 0:
            logger.warning("Zero parsed documents produced for ingestion stage.")

        write_jsonl(artifact_paths.output_path, parsed_documents)

        logger.info("Wrote parsed documents artifact: path=%s", artifact_paths.output_path)

        finished_at, duration_ms = finish_timed_run(timed_run)

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
            "input_dir": str(artifact_paths.input_dir),
            "output_path": str(artifact_paths.output_path),
            "raw_documents_total": len(raw_documents),
            "parsed_documents_total": len(parsed_documents),
            "failed_documents_total": failed_documents_total,
            "skipped_documents_total": skipped_documents_total,
            "parser_counts": parser_counts,
        }

        write_manifest(artifact_paths.manifest_path, manifest)

        logger.info("Wrote ingestion manifest: path=%s", artifact_paths.manifest_path)

        status = compute_layer_status(
            success_count=len(parsed_documents),
            error_count=failed_documents_total,
        )

        logger.info(
            "Finished ingestion layer: status=%s duration_ms=%s raw_documents_total=%s parsed_documents_total=%s failed_documents_total=%s skipped_documents_total=%s",
            status,
            duration_ms,
            len(raw_documents),
            len(parsed_documents),
            failed_documents_total,
            skipped_documents_total,
        )

        return BuildParsedDocsResult(
            run_id=timed_run.run_id,
            input_dir=artifact_paths.input_dir,
            output_path=artifact_paths.output_path,
            manifest_path=artifact_paths.manifest_path,
            raw_documents_total=len(raw_documents),
            parsed_documents_total=len(parsed_documents),
            failed_documents_total=failed_documents_total,
            skipped_documents_total=skipped_documents_total,
            duration_ms=duration_ms,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build parsed documents from raw local sources.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to config.yaml or a config directory.",
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--fail-fast", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    setup_logging()
    args = _build_arg_parser().parse_args()
    layer = BuildParsedDocsLayer(
        config_path=args.config,
        input_dir=args.input_dir,
        output_path=args.output_path,
        manifest_path=args.manifest_path,
        fail_fast=args.fail_fast,
        limit=args.limit,
    )
    try:
        result = layer.run()
    except Exception:
        logger.exception("BuildParsedDocsLayer failed")
        sys.exit(1)

    if args.fail_fast and result.failed_documents_total > 0:
        sys.exit(1)
    if result.parsed_documents_total == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
