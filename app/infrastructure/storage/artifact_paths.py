"""Helpers for resolving offline artifact paths."""

from dataclasses import dataclass
from pathlib import Path

from app.utils.config import Settings


@dataclass(frozen=True)
class IngestionArtifactPaths:
    """Resolved filesystem paths required for parsed-document build stage."""

    input_dir: Path
    output_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class CleaningArtifactPaths:
    """Resolved filesystem paths required for cleaned-document build stage."""

    input_path: Path
    output_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class ChunkingArtifactPaths:
    """Resolved filesystem paths required for chunking build stage."""

    input_path: Path
    output_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class EmbeddingArtifactPaths:
    """Resolved filesystem paths required for embedding build stage."""

    input_path: Path
    output_path: Path
    manifest_path: Path
    cache_path: Path


@dataclass(frozen=True)
class VectorIndexArtifactPaths:
    """Resolved filesystem paths required for vector indexing stage."""

    input_path: Path
    manifest_path: Path


def resolve_ingestion_artifact_paths(
    settings: Settings,
    *,
    input_dir_override: Path | None = None,
    output_path_override: Path | None = None,
    manifest_path_override: Path | None = None,
) -> IngestionArtifactPaths:
    """Resolve ingestion stage paths from config with optional CLI overrides."""
    input_dir = input_dir_override or settings.ingestion.input_dir
    output_path = output_path_override or settings.ingestion.outputs.parsed_documents_path
    manifest_path = manifest_path_override or settings.ingestion.outputs.manifest_path

    return IngestionArtifactPaths(
        input_dir=Path(input_dir),
        output_path=Path(output_path),
        manifest_path=Path(manifest_path),
    )


def resolve_cleaning_artifact_paths(
    settings: Settings,
    *,
    input_path_override: Path | None = None,
    output_path_override: Path | None = None,
    manifest_path_override: Path | None = None,
) -> CleaningArtifactPaths:
    """Resolve cleaning stage paths from config with optional CLI overrides."""
    input_path = input_path_override or settings.ingestion.outputs.parsed_documents_path
    output_path = output_path_override or settings.ingestion.outputs.cleaned_documents_path
    manifest_path = manifest_path_override or settings.cleaning.outputs.manifest_path

    return CleaningArtifactPaths(
        input_path=Path(input_path),
        output_path=Path(output_path),
        manifest_path=Path(manifest_path),
    )


def resolve_chunking_artifact_paths(
    settings: Settings,
    *,
    input_path_override: Path | None = None,
    output_path_override: Path | None = None,
    manifest_path_override: Path | None = None,
) -> ChunkingArtifactPaths:
    """Resolve chunking stage paths from config with optional CLI overrides."""
    input_path = input_path_override or settings.chunking.input_path
    output_path = output_path_override or settings.chunking.outputs.chunks_path
    manifest_path = manifest_path_override or settings.chunking.outputs.manifest_path
    return ChunkingArtifactPaths(
        input_path=Path(input_path),
        output_path=Path(output_path),
        manifest_path=Path(manifest_path),
    )


def resolve_embedding_artifact_paths(
    settings: Settings,
    *,
    input_path_override: Path | None = None,
    output_path_override: Path | None = None,
    manifest_path_override: Path | None = None,
    cache_path_override: Path | None = None,
) -> EmbeddingArtifactPaths:
    """Resolve embedding stage paths from config with optional CLI overrides."""
    input_path = input_path_override or settings.embeddings.input_path
    output_path = output_path_override or settings.embeddings.output_path
    manifest_path = manifest_path_override or settings.embeddings.manifest_path
    cache_path = cache_path_override or settings.embeddings.cache_path
    return EmbeddingArtifactPaths(
        input_path=Path(input_path),
        output_path=Path(output_path),
        manifest_path=Path(manifest_path),
        cache_path=Path(cache_path),
    )


def resolve_vector_index_artifact_paths(
    settings: Settings,
    *,
    input_path_override: Path | None = None,
    manifest_path_override: Path | None = None,
) -> VectorIndexArtifactPaths:
    """Resolve vector indexing paths from config with optional CLI overrides."""
    input_path = input_path_override or settings.indexing.input_path
    manifest_path = manifest_path_override or settings.indexing.manifest_path
    return VectorIndexArtifactPaths(
        input_path=Path(input_path),
        manifest_path=Path(manifest_path),
    )
