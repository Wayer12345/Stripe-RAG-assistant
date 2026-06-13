"""Typed settings loader for the stripe-rag-assistant project.

Reads a single ``config.yaml`` from the configs directory and validates the
result with Pydantic.

Usage::

    from app.utils.config import load_settings

    settings = load_settings()           # uses configs/ relative to cwd
    settings = load_settings(Path("configs"))  # explicit path

Raises:
    FileNotFoundError: If ``config.yaml`` is missing.
    pydantic.ValidationError: If any value fails validation.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.utils.constants import DEFAULT_CONFIG_FILE_NAME


# Enforce fail-fast behavior for unknown config keys across all settings models.
class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# App and path settings
# ---------------------------------------------------------------------------


class AppSettings(StrictBaseModel):
    """Top-level application identity settings."""

    name: str
    environment: str
    log_level: str

    @field_validator("name", "environment", "log_level")
    @classmethod
    def _non_empty(cls, v: str, info: Any) -> str:
        if not v.strip():
            raise ValueError(f"app.{info.field_name} must not be empty")
        return v


class PathSettings(StrictBaseModel):
    """Filesystem path roots used by all pipelines."""

    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    indexes_dir: Path
    manifests_dir: Path
    eval_dir: Path


# ---------------------------------------------------------------------------
# Ingestion settings
# ---------------------------------------------------------------------------


class IngestionSettings(StrictBaseModel):
    """Full ingestion pipeline configuration."""

    class OutputSettings(StrictBaseModel):
        """Output artifact paths produced by the ingestion pipeline."""

        parsed_documents_path: Path
        cleaned_documents_path: Path
        manifest_path: Path

    class FailurePolicySettings(StrictBaseModel):
        """Per-stage failure handling policy."""

        continue_on_resolve_error: bool = True
        continue_on_parse_error: bool = True
        continue_on_clean_error: bool = True
        fail_on_empty_input: bool = False

    class TxtParserOptions(StrictBaseModel):
        encoding: str = "utf-8"
        fallback_errors: str = "replace"

    class MarkdownParserOptions(StrictBaseModel):
        encoding: str = "utf-8"
        fallback_errors: str = "replace"

    class HtmlParserOptions(StrictBaseModel):
        remove_script_style: bool = True
        preserve_visible_text: bool = True

    class PdfParserOptions(StrictBaseModel):
        preserve_page_markers: bool = True
        page_marker_template: str = "[Page {page_number}]"

    class DocxParserOptions(StrictBaseModel):
        extract_tables: bool = True

    class JsonParserOptions(StrictBaseModel):
        text_fields: list[str] = [
            "text",
            "content",
            "body",
            "markdown",
            "html",
            "answer",
            "description",
        ]
        title_fields: list[str] = ["title", "name", "heading", "question"]

    class CsvParserOptions(StrictBaseModel):
        delimiter: str | None = None
        include_headers: bool = True

    class ParserOptionsSettings(StrictBaseModel):
        """Typed parser options for each supported file format."""

        model_config = ConfigDict(populate_by_name=True, extra="forbid")

        txt: IngestionSettings.TxtParserOptions = Field(
            default_factory=lambda: IngestionSettings.TxtParserOptions()
        )
        markdown: IngestionSettings.MarkdownParserOptions = Field(
            default_factory=lambda: IngestionSettings.MarkdownParserOptions()
        )
        html: IngestionSettings.HtmlParserOptions = Field(
            default_factory=lambda: IngestionSettings.HtmlParserOptions()
        )
        pdf: IngestionSettings.PdfParserOptions = Field(
            default_factory=lambda: IngestionSettings.PdfParserOptions()
        )
        docx: IngestionSettings.DocxParserOptions = Field(
            default_factory=lambda: IngestionSettings.DocxParserOptions()
        )
        json_files: IngestionSettings.JsonParserOptions = Field(
            default_factory=lambda: IngestionSettings.JsonParserOptions(),
            alias="json",
        )
        csv: IngestionSettings.CsvParserOptions = Field(
            default_factory=lambda: IngestionSettings.CsvParserOptions()
        )

    input_dir: Path
    recursive: bool = True
    supported_extensions: list[str]
    outputs: OutputSettings
    failure_policy: FailurePolicySettings = Field(default_factory=FailurePolicySettings)
    parser_options: ParserOptionsSettings = Field(default_factory=ParserOptionsSettings)

    @field_validator("supported_extensions")
    @classmethod
    def _validate_extensions(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("ingestion.supported_extensions must not be empty")
        normalized: list[str] = []
        for ext in v:
            lowered = ext.lower()
            if not lowered.startswith("."):
                raise ValueError(f"Each extension must start with '.', got: {ext!r}")
            normalized.append(lowered)
        return normalized


# ---------------------------------------------------------------------------
# Cleaning settings
# ---------------------------------------------------------------------------


class CleaningSettings(StrictBaseModel):
    """Full cleaning pipeline configuration."""

    class OutputSettings(StrictBaseModel):
        """Output artifact paths produced by the cleaning stage."""

        manifest_path: Path = Path("data/manifests/cleaning_manifest.json")

    class StepSettings(StrictBaseModel):
        """Toggle individual cleaning steps."""

        normalize_unicode: bool = True
        remove_html_artifacts: bool = True
        remove_boilerplate: bool = True
        normalize_whitespace: bool = True
        remove_duplicate_lines: bool = True

    class DuplicateLineSettings(StrictBaseModel):
        """Nearby duplicate-line removal tuning."""

        window_size: int = 5

        @field_validator("window_size")
        @classmethod
        def _validate_window_size(cls, v: int) -> int:
            if v < 1:
                raise ValueError(f"cleaning.duplicate_lines.window_size must be >= 1, got {v}")
            return v

    class BlankLineSettings(StrictBaseModel):
        """Consecutive blank-line reduction tuning."""

        max_blank_lines: int = 1

        @field_validator("max_blank_lines")
        @classmethod
        def _validate_max_blank_lines(cls, v: int) -> int:
            if v < 0:
                raise ValueError(f"cleaning.blank_lines.max_blank_lines must be >= 0, got {v}")
            return v

    class BoilerplateSettings(StrictBaseModel):
        """Boilerplate-line removal configuration."""

        remove_standalone_lines: bool = True
        max_line_length: int = 80
        phrases: list[str]

        @field_validator("max_line_length")
        @classmethod
        def _validate_max_line_length(cls, v: int) -> int:
            if v <= 0:
                raise ValueError(f"cleaning.boilerplate.max_line_length must be > 0, got {v}")
            return v

        @field_validator("phrases")
        @classmethod
        def _validate_phrases(cls, v: list[str]) -> list[str]:
            if not v:
                raise ValueError("cleaning.boilerplate.phrases must not be empty")
            return v

    class QualitySettings(StrictBaseModel):
        """Quality-check thresholds and warn/fail flags."""

        overcleaning_threshold: float = 0.10
        undercleaning_threshold_for_html: float = 0.95
        fail_on_empty_output: bool = True
        warn_on_possible_overcleaning: bool = True
        warn_on_possible_undercleaning: bool = True

        @field_validator("overcleaning_threshold")
        @classmethod
        def _validate_overcleaning(cls, v: float) -> float:
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"cleaning.quality.overcleaning_threshold must be between 0 and 1, got {v}"
                )
            return v

        @field_validator("undercleaning_threshold_for_html")
        @classmethod
        def _validate_undercleaning(cls, v: float) -> float:
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    "cleaning.quality.undercleaning_threshold_for_html must be between "
                    f"0 and 1, got {v}"
                )
            return v

    class PreserveSettings(StrictBaseModel):
        """Preservation flags for content that must survive cleaning."""

        headings: bool = True
        lists: bool = True
        faq_markers: bool = True
        table_like_lines: bool = True
        page_markers: bool = True
        urls_in_meaningful_lines: bool = True

    mode: str
    outputs: OutputSettings = Field(default_factory=OutputSettings)
    steps: StepSettings = Field(default_factory=StepSettings)
    duplicate_lines: DuplicateLineSettings = Field(default_factory=DuplicateLineSettings)
    blank_lines: BlankLineSettings = Field(default_factory=BlankLineSettings)
    boilerplate: BoilerplateSettings
    quality: QualitySettings = Field(default_factory=QualitySettings)
    preserve: PreserveSettings = Field(default_factory=PreserveSettings)

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        if v != "conservative":
            raise ValueError(f"cleaning.mode must be 'conservative', got {v!r}")
        return v


# ---------------------------------------------------------------------------
# Chunking settings
# ---------------------------------------------------------------------------


class ChunkingSettings(StrictBaseModel):
    """Chunking strategy and chunk size settings."""

    class OutputSettings(StrictBaseModel):
        """Output artifact paths produced by the chunking stage."""

        chunks_path: Path = Path("data/processed/chunks.jsonl")
        manifest_path: Path = Path("data/manifests/chunking_manifest.json")

    strategy: str = "semantic"
    input_path: Path = Path("data/interim/cleaned_documents.jsonl")
    outputs: OutputSettings = Field(default_factory=OutputSettings)

    chunk_size: int = 1800
    chunk_size_min: int = 300
    chunk_size_max: int = 1800
    chunk_overlap: int = 250
    min_chunk_chars: int = 1
    max_chunk_chars: int = 1800
    overlap_chars: int = 250
    max_overlap_units: int = 3
    use_semantic_boundaries: bool = False
    similarity_threshold: float = 0.55
    boundary_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    unit_embed_batch_size: int = 32

    @field_validator(
        "chunk_size",
        "chunk_size_min",
        "chunk_size_max",
        "chunk_overlap",
        "min_chunk_chars",
        "max_chunk_chars",
        "overlap_chars",
        "max_overlap_units",
        "unit_embed_batch_size",
    )
    @classmethod
    def _must_be_positive(cls, v: int, info: Any) -> int:
        if v <= 0:
            raise ValueError(f"chunking.{info.field_name} must be > 0, got {v}")
        return v

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("chunking.strategy must not be empty")
        return v.strip()

    @field_validator("boundary_embedding_model_name")
    @classmethod
    def _validate_boundary_model_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("chunking.boundary_embedding_model_name must not be empty")
        return v.strip()

    @field_validator("similarity_threshold")
    @classmethod
    def _validate_similarity_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"chunking.similarity_threshold must be between 0 and 1, got {v}")
        return v

    @field_validator("chunk_size_max")
    @classmethod
    def _validate_chunk_size_max(cls, v: int, info: Any) -> int:
        min_value = info.data.get("chunk_size_min")
        if isinstance(min_value, int) and v < min_value:
            raise ValueError("chunking.chunk_size_max must be >= chunking.chunk_size_min")
        return v

    @field_validator("max_chunk_chars")
    @classmethod
    def _validate_max_chunk_chars(cls, v: int, info: Any) -> int:
        min_value = info.data.get("min_chunk_chars")
        if isinstance(min_value, int) and v < min_value:
            raise ValueError("chunking.max_chunk_chars must be >= chunking.min_chunk_chars")
        return v


# ---------------------------------------------------------------------------
# Embeddings settings
# ---------------------------------------------------------------------------


class EmbeddingsSettings(StrictBaseModel):
    """Embedding provider/model and embedding artifact path settings."""

    provider: str = "sentence_transformers"
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 32
    normalize_embeddings: bool = True
    prefix_mode: str = "bge"
    cache_enabled: bool = True
    cache_path: Path = Path("data/indexes/embedding_cache")
    input_path: Path = Path("data/processed/chunks.jsonl")
    output_path: Path = Path("data/processed/embedded_chunks.jsonl")
    manifest_path: Path = Path("data/manifests/embedding_manifest.json")

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        value = v.strip()
        if not value:
            raise ValueError("embeddings.provider must not be empty")
        return value

    @field_validator("model_name", "prefix_mode")
    @classmethod
    def _validate_non_empty_strings(cls, v: str, info: Any) -> str:
        value = v.strip()
        if not value:
            raise ValueError(f"embeddings.{info.field_name} must not be empty")
        return value

    @field_validator("batch_size")
    @classmethod
    def _validate_batch_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("embeddings.batch_size must be > 0")
        return v

    @field_validator("prefix_mode")
    @classmethod
    def _validate_prefix_mode(cls, v: str) -> str:
        allowed = {"none", "bge", "e5"}
        normalized = v.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"embeddings.prefix_mode must be one of {sorted(allowed)}, got {v!r}")
        return normalized


# ---------------------------------------------------------------------------
# Vector store and indexing settings
# ---------------------------------------------------------------------------


class VectorStoreSettings(StrictBaseModel):
    """Vector store provider + Qdrant connection and write behavior."""

    provider: str = "qdrant"
    mode: str = "embedded"
    local_path: Path = Path("data/indexes/qdrant")
    host: str = "localhost"
    port: int = 6333
    url: str | None = None
    api_key: str | None = None
    prefer_grpc: bool = False
    collection_name: str = "stripe_guides_v1"
    distance: str = "cosine"
    timeout: int = 30
    recreate_collection: bool = False
    create_payload_indexes: bool = True
    upsert_batch_size: int = 64
    wait: bool = True
    payload_indexes: dict[str, str] = Field(
        default_factory=lambda: {
            "document_id": "keyword",
            "url": "keyword",
            "category": "keyword",
            "source_type": "keyword",
            "content_hash": "keyword",
            "token_count": "integer",
            "source_path": "keyword",
            "chunk_id": "keyword",
        }
    )

    @field_validator("provider", "mode", "host", "collection_name", "distance")
    @classmethod
    def _validate_non_empty_strings(cls, v: str, info: Any) -> str:
        value = v.strip()
        if not value:
            raise ValueError(f"vector_store.{info.field_name} must not be empty")
        return value

    @field_validator("port", "timeout", "upsert_batch_size")
    @classmethod
    def _validate_positive_ints(cls, v: int, info: Any) -> int:
        if v <= 0:
            raise ValueError(f"vector_store.{info.field_name} must be > 0")
        return v

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        normalized = v.strip().lower()
        if normalized != "embedded":
            raise ValueError(
                "vector_store.mode currently supports only 'embedded' in this project."
            )
        return normalized

    @field_validator("distance")
    @classmethod
    def _validate_distance(cls, v: str) -> str:
        normalized = v.strip().lower()
        allowed = {"cosine", "dot", "euclid", "manhattan"}
        if normalized not in allowed:
            raise ValueError(f"vector_store.distance must be one of {sorted(allowed)}, got {v!r}")
        return normalized

    @field_validator("payload_indexes")
    @classmethod
    def _validate_payload_indexes(cls, v: dict[str, str]) -> dict[str, str]:
        allowed_types = {"keyword", "integer", "float", "bool", "datetime", "text"}
        normalized: dict[str, str] = {}
        for field_name, schema_type in v.items():
            normalized_field = field_name.strip()
            normalized_schema = schema_type.strip().lower()
            if not normalized_field:
                raise ValueError("vector_store.payload_indexes keys must not be empty")
            if normalized_schema not in allowed_types:
                raise ValueError(
                    "vector_store.payload_indexes values must be one of "
                    f"{sorted(allowed_types)}, got {schema_type!r} for {field_name!r}"
                )
            normalized[normalized_field] = normalized_schema
        return normalized


class IndexingSettings(StrictBaseModel):
    """Vector indexing stage artifact paths and validation toggles."""

    input_path: Path = Path("data/processed/embedded_chunks.jsonl")
    manifest_path: Path = Path("data/manifests/index_manifest.json")
    recreate_collection: bool = False
    create_payload_indexes: bool = True
    upsert_batch_size: int = 64
    validate_after_upsert: bool = True
    validate_only: bool = False

    @field_validator("upsert_batch_size")
    @classmethod
    def _validate_upsert_batch_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("indexing.upsert_batch_size must be > 0")
        return v


# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------


class RetrievalSettings(StrictBaseModel):
    """Online retrieval strategy and tracing settings."""

    strategy: str = "dense"
    dense_top_k: int = 30
    write_trace: bool = True
    trace_dir: Path = Path("data/traces/queries")

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        normalized = v.strip().lower()
        allowed = {"dense", "lexical", "hybrid"}
        if normalized not in allowed:
            raise ValueError(f"retrieval.strategy must be one of {sorted(allowed)}, got {v!r}")
        return normalized

    @field_validator("dense_top_k")
    @classmethod
    def _validate_dense_top_k(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("retrieval.dense_top_k must be > 0")
        return v

    @field_validator("trace_dir")
    @classmethod
    def _validate_trace_dir(cls, v: Path) -> Path:
        if not str(v).strip():
            raise ValueError("retrieval.trace_dir must not be empty")
        return v


# ---------------------------------------------------------------------------
# Reranking settings
# ---------------------------------------------------------------------------


class RerankingSettings(StrictBaseModel):
    """Online reranking strategy and trace settings."""

    enabled: bool = True
    provider: str = "cross_encoder"
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_before: int = 12
    top_k_after: int = 8
    batch_size: int = 12
    max_query_chars: int = 512
    max_pair_chars: int = 1800
    warmup_enabled: bool = True
    cache_enabled: bool = True
    cache_path: Path = Path("data/indexes/reranker_cache")
    latency_budget_ms: int = 50
    on_latency_budget_exceeded: str = "warn"
    write_trace: bool = True
    trace_dir: Path = Path("data/traces/queries")
    text_preview_chars: int = 300

    @field_validator("provider", "model_name", "on_latency_budget_exceeded")
    @classmethod
    def _validate_non_empty_strings(cls, v: str, info: Any) -> str:
        value = v.strip()
        if not value:
            raise ValueError(f"reranking.{info.field_name} must not be empty")
        return value

    @field_validator(
        "top_k_before",
        "top_k_after",
        "batch_size",
        "max_query_chars",
        "max_pair_chars",
        "latency_budget_ms",
        "text_preview_chars",
    )
    @classmethod
    def _validate_positive_ints(cls, v: int, info: Any) -> int:
        if v <= 0:
            raise ValueError(f"reranking.{info.field_name} must be > 0")
        return v

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        normalized = v.strip().lower()
        if normalized != "cross_encoder":
            raise ValueError("reranking.provider currently supports only 'cross_encoder'.")
        return normalized

    @field_validator("on_latency_budget_exceeded")
    @classmethod
    def _validate_latency_policy(cls, v: str) -> str:
        normalized = v.strip().lower()
        allowed = {"warn", "ignore", "raise"}
        if normalized not in allowed:
            raise ValueError(
                f"reranking.on_latency_budget_exceeded must be one of {sorted(allowed)}, got {v!r}"
            )
        return normalized

    @field_validator("trace_dir")
    @classmethod
    def _validate_trace_dir(cls, v: Path) -> Path:
        if not str(v).strip():
            raise ValueError("reranking.trace_dir must not be empty")
        return v

    @field_validator("top_k_after")
    @classmethod
    def _validate_top_k_relationship(cls, v: int, info: Any) -> int:
        top_k_before = info.data.get("top_k_before")
        if isinstance(top_k_before, int) and v > top_k_before:
            raise ValueError("reranking.top_k_after must be <= reranking.top_k_before")
        return v


# ---------------------------------------------------------------------------
# Context-building settings
# ---------------------------------------------------------------------------


class ContextSettings(StrictBaseModel):
    """Online context-building and trace settings."""

    token_budget: int = 3500
    max_chunks: int = 8
    max_sources: int = 5
    min_chunk_tokens: int = 20
    max_chunk_tokens: int = 700
    truncate_long_chunks: bool = True
    deduplicate_by: list[str] = Field(default_factory=lambda: ["chunk_id", "text_hash", "url"])
    include_scores: bool = True
    include_metadata: bool = True
    context_format_version: str = "context_v1"
    write_trace: bool = True
    trace_dir: Path = Path("data/traces/queries")
    text_preview_chars: int = 500
    include_full_context_in_trace: bool = False

    @field_validator(
        "token_budget",
        "max_chunks",
        "max_sources",
        "min_chunk_tokens",
        "max_chunk_tokens",
        "text_preview_chars",
    )
    @classmethod
    def _validate_positive_ints(cls, v: int, info: Any) -> int:
        if v <= 0:
            raise ValueError(f"context.{info.field_name} must be > 0")
        return v

    @field_validator("context_format_version")
    @classmethod
    def _validate_context_format_version(cls, v: str) -> str:
        value = v.strip()
        if not value:
            raise ValueError("context.context_format_version must not be empty")
        return value

    @field_validator("deduplicate_by")
    @classmethod
    def _validate_deduplicate_by(cls, v: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for key in v:
            candidate = key.strip().lower()
            if not candidate:
                continue
            if candidate not in seen:
                normalized.append(candidate)
                seen.add(candidate)
        if not normalized:
            raise ValueError("context.deduplicate_by must contain at least one key")
        return normalized

    @field_validator("trace_dir")
    @classmethod
    def _validate_trace_dir(cls, v: Path) -> Path:
        if not str(v).strip():
            raise ValueError("context.trace_dir must not be empty")
        return v


# ---------------------------------------------------------------------------
# Generation settings
# ---------------------------------------------------------------------------


class GenerationSettings(StrictBaseModel):
    """Online generation provider/model, prompt, and trace settings."""

    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model_name: str = "llama3.1:8b"
    temperature: float = 0.1
    max_tokens: int = 700
    keep_alive: str | None = "30m"
    top_p: float | None = None
    timeout_seconds: int = 120
    context_token_budget: int = 3500
    min_context_tokens: int = 1
    prompts_dir: Path = Path("prompts")
    answer_template_name: str = "answer_prompt_v1.jinja"
    no_answer_template_name: str = "no_answer_prompt_v1.jinja"
    write_trace: bool = True
    trace_dir: Path = Path("data/traces/queries")
    text_preview_chars: int = 500
    include_full_prompt_in_trace: bool = False
    include_raw_output_in_trace: bool = False
    include_full_answer_in_trace: bool = False
    no_answer_quality_threshold_pct: float = 50.0

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        normalized = v.strip().lower()
        if normalized != "ollama":
            raise ValueError("generation.provider currently supports only 'ollama'.")
        return normalized

    @field_validator("base_url", "model_name", "answer_template_name", "no_answer_template_name")
    @classmethod
    def _validate_non_empty_strings(cls, v: str, info: Any) -> str:
        value = v.strip()
        if not value:
            raise ValueError(f"generation.{info.field_name} must not be empty")
        return value

    @field_validator("keep_alive")
    @classmethod
    def _validate_keep_alive(cls, v: str | None) -> str | None:
        if v is None:
            return None
        value = v.strip()
        if not value:
            raise ValueError("generation.keep_alive must not be empty when provided")
        return value

    @field_validator(
        "max_tokens",
        "timeout_seconds",
        "context_token_budget",
        "min_context_tokens",
        "text_preview_chars",
    )
    @classmethod
    def _validate_positive_ints(cls, v: int, info: Any) -> int:
        if v <= 0:
            raise ValueError(f"generation.{info.field_name} must be > 0")
        return v

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("generation.temperature must be >= 0")
        return v

    @field_validator("top_p")
    @classmethod
    def _validate_top_p(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 < v <= 1.0):
            raise ValueError("generation.top_p must be in (0, 1] when provided")
        return v

    @field_validator("no_answer_quality_threshold_pct")
    @classmethod
    def _validate_no_answer_quality_threshold_pct(cls, v: float) -> float:
        if not (0.0 <= v <= 100.0):
            raise ValueError("generation.no_answer_quality_threshold_pct must be between 0 and 100")
        return v

    @field_validator("trace_dir")
    @classmethod
    def _validate_trace_dir(cls, v: Path) -> Path:
        if not str(v).strip():
            raise ValueError("generation.trace_dir must not be empty")
        return v


# ---------------------------------------------------------------------------
# Online query CLI settings
# ---------------------------------------------------------------------------


class OnlineQuerySettings(StrictBaseModel):
    """Top-level online query orchestration and output settings."""

    write_trace: bool = True
    trace_dir: Path = Path("data/traces/queries")
    text_preview_chars: int = 500
    include_full_answer_in_trace: bool = False
    print_answer_by_default: bool = True
    print_sources_by_default: bool = True
    fail_fast: bool = True

    @field_validator("text_preview_chars")
    @classmethod
    def _validate_text_preview_chars(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("online_query.text_preview_chars must be > 0")
        return v

    @field_validator("trace_dir")
    @classmethod
    def _validate_trace_dir(cls, v: Path) -> Path:
        if not str(v).strip():
            raise ValueError("online_query.trace_dir must not be empty")
        return v


class ApiSettings(StrictBaseModel):
    """FastAPI runtime and lifecycle configuration."""

    class WarmupSettings(StrictBaseModel):
        retrieval_enabled: bool = True
        retrieval_embed_query_enabled: bool = True
        retrieval_embed_query_text: str = "warmup query"
        retrieval_tiny_search_enabled: bool = False
        retrieval_tiny_search_top_k: int = 1
        reranker_enabled: bool = True
        context_enabled: bool = True
        generation_enabled: bool = True
        ollama_healthcheck_enabled: bool = True
        ollama_generate_enabled: bool = False
        ollama_generate_prompt: str = "Respond with: OK"
        ollama_generate_max_tokens: int = 8
        qdrant_healthcheck_enabled: bool = True

        @field_validator("ollama_generate_prompt")
        @classmethod
        def _validate_ollama_generate_prompt(cls, v: str) -> str:
            value = v.strip()
            if not value:
                raise ValueError("api.warmup.ollama_generate_prompt must not be empty")
            return value

        @field_validator("retrieval_embed_query_text")
        @classmethod
        def _validate_retrieval_embed_query_text(cls, v: str) -> str:
            value = v.strip()
            if not value:
                raise ValueError("api.warmup.retrieval_embed_query_text must not be empty")
            return value

        @field_validator("retrieval_tiny_search_top_k")
        @classmethod
        def _validate_retrieval_tiny_search_top_k(cls, v: int) -> int:
            if v <= 0:
                raise ValueError("api.warmup.retrieval_tiny_search_top_k must be > 0")
            return v

        @field_validator("ollama_generate_max_tokens")
        @classmethod
        def _validate_ollama_generate_max_tokens(cls, v: int) -> int:
            if v <= 0:
                raise ValueError("api.warmup.ollama_generate_max_tokens must be > 0")
            return v

    enabled: bool = True
    title: str = "Stripe RAG Assistant"
    version: str = "0.1.0"
    debug: bool = False
    warmup_on_startup: bool = True
    fail_startup_on_warmup_error: bool = False
    shutdown_on_exit: bool = True
    warmup: WarmupSettings = WarmupSettings()

    @field_validator("title", "version")
    @classmethod
    def _validate_non_empty_strings(cls, v: str, info: Any) -> str:
        value = v.strip()
        if not value:
            raise ValueError(f"api.{info.field_name} must not be empty")
        return value


class EvalSettings(StrictBaseModel):
    """Reserved top-level settings section for evaluation configuration."""

    class PreflightSettings(StrictBaseModel):
        """Strict readiness checks required before eval starts."""

        enabled: bool = True
        require_qdrant_healthcheck: bool = True
        require_embed_query_warmup: bool = True
        retrieval_warmup_query: str = "warmup query"
        require_tiny_search_warmup: bool = True
        tiny_search_top_k: int = 1
        require_ollama_healthcheck: bool = True
        require_ollama_generate_warmup: bool = True
        ollama_generate_prompt: str = "Respond with: OK"
        ollama_generate_max_tokens: int = 8

        @field_validator("tiny_search_top_k", "ollama_generate_max_tokens")
        @classmethod
        def _validate_positive_ints(cls, value: int, info: Any) -> int:
            if value <= 0:
                raise ValueError(f"eval.preflight.{info.field_name} must be > 0")
            return value

        @field_validator("ollama_generate_prompt")
        @classmethod
        def _validate_ollama_generate_prompt(cls, value: str) -> str:
            normalized = value.strip()
            if not normalized:
                raise ValueError("eval.preflight.ollama_generate_prompt must not be empty")
            return normalized

        @field_validator("retrieval_warmup_query")
        @classmethod
        def _validate_retrieval_warmup_query(cls, value: str) -> str:
            normalized = value.strip()
            if not normalized:
                raise ValueError("eval.preflight.retrieval_warmup_query must not be empty")
            return normalized

    datasets_dir: Path = Path("data/eval/datasets")
    runs_dir: Path = Path("data/eval/runs")
    baselines_dir: Path = Path("data/eval/baselines")
    default_suite: str = "full"
    default_limit: int | None = None
    default_seed: int = 42
    default_synthetic_target_size: int = 200
    default_negative_target_size: int = 25
    default_robustness_target_size: int = 50
    default_audit_target_size: int = 50
    min_chunk_chars: int = 300
    write_trace: bool = False
    fail_fast: bool = False
    judge_enabled: bool = False
    judge_backend: str = "heuristic"
    text_preview_chars: int = 500
    preflight: PreflightSettings = Field(default_factory=PreflightSettings)

    @field_validator(
        "default_seed",
        "default_synthetic_target_size",
        "default_negative_target_size",
        "default_robustness_target_size",
        "default_audit_target_size",
        "min_chunk_chars",
        "text_preview_chars",
    )
    @classmethod
    def _positive_ints(cls, value: int, info: Any) -> int:
        if value <= 0:
            raise ValueError(f"eval.{info.field_name} must be > 0")
        return value

    @field_validator("default_limit")
    @classmethod
    def _optional_positive_limit(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("eval.default_limit must be > 0 when provided")
        return value

    @field_validator("default_suite")
    @classmethod
    def _validate_default_suite(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {"retrieval", "rerank", "context", "generation", "citation", "robustness", "full"}
        if normalized not in allowed:
            raise ValueError(f"eval.default_suite must be one of {sorted(allowed)}, got {value!r}")
        return normalized

    @field_validator("judge_backend")
    @classmethod
    def _validate_judge_backend(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {"heuristic", "local_llm", "none"}
        if normalized not in allowed:
            raise ValueError(f"eval.judge_backend must be one of {sorted(allowed)}, got {value!r}")
        return normalized


class LocalSettings(StrictBaseModel):
    """Reserved top-level settings section for local-only runtime settings."""


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


class Settings(StrictBaseModel):
    """Root validated settings object assembled from all config files."""

    app: AppSettings
    paths: PathSettings
    ingestion: IngestionSettings
    cleaning: CleaningSettings
    chunking: ChunkingSettings = ChunkingSettings()
    embeddings: EmbeddingsSettings = EmbeddingsSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    indexing: IndexingSettings = IndexingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    reranking: RerankingSettings = RerankingSettings()
    context: ContextSettings = ContextSettings()
    generation: GenerationSettings = GenerationSettings()
    online_query: OnlineQuerySettings = OnlineQuerySettings()
    api: ApiSettings = ApiSettings()
    eval: EvalSettings = EvalSettings()
    local: LocalSettings = LocalSettings()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_DIR = _PROJECT_ROOT / "configs"


def resolve_config_dir_and_path(config_path: Path | str) -> tuple[Path, Path]:
    """Resolve a config directory and concrete config.yaml path."""
    resolved_path = Path(config_path)
    if resolved_path.is_dir():
        return resolved_path, resolved_path / DEFAULT_CONFIG_FILE_NAME
    if resolved_path.name != DEFAULT_CONFIG_FILE_NAME:
        raise ValueError("--config must point to config.yaml or a directory.")
    return resolved_path.parent, resolved_path


def to_optional_path(path_value: Path | str | None) -> Path | None:
    """Return ``None`` or a ``Path`` converted from *path_value*."""
    if path_value is None:
        return None
    return Path(path_value)


def validate_positive_limit(limit: int | None, *, name: str = "--limit") -> None:
    """Validate that optional integer limit is positive when provided."""
    if limit is not None and limit <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def apply_limit[T](items: Sequence[T], limit: int | None) -> tuple[list[T], int]:
    """Apply optional limit to sequence and return selected items + skipped count."""
    if limit is None:
        return list(items), 0
    selected_items = list(items[:limit])
    skipped_total = len(items) - len(selected_items)
    return selected_items, skipped_total


def load_settings(config_dir: Path = _DEFAULT_CONFIG_DIR) -> Settings:
    """Load and validate ``config.yaml`` from *config_dir*.

    Args:
        config_dir: Directory containing the YAML config files.
            Defaults to project-root ``configs`` and is resolved robustly even
            when scripts are launched outside the repository root.

    Returns:
        Fully validated :class:`Settings` instance.

    Raises:
        FileNotFoundError: If ``config.yaml`` is absent.
        pydantic.ValidationError: If any value fails type or constraint validation.
    """
    candidate_dir = Path(config_dir)
    if not candidate_dir.is_absolute() and not candidate_dir.exists():
        candidate_dir = _PROJECT_ROOT / candidate_dir

    path = candidate_dir / DEFAULT_CONFIG_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"Required config file not found: {path}. Ensure config.yaml exists before running."
        )

    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    return Settings.model_validate(data)
