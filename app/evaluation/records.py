"""Core data records for evaluation datasets and eval runs."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvalExpectedBehavior(StrEnum):
    """Expected assistant behavior for an eval sample."""

    ANSWER = "answer"
    ABSTAIN = "abstain"
    EITHER = "either"


class EvalSubset(StrEnum):
    """Dataset subset labels used by the eval layer."""

    SYNTHETIC_SOURCE_GROUNDED = "synthetic_source_grounded"
    NEGATIVE = "negative"
    ROBUSTNESS = "robustness"
    AUDIT = "audit"
    MANUAL = "manual"


class EvalDifficulty(StrEnum):
    """Difficulty label for one eval sample."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    UNKNOWN = "unknown"


class EvalQueryType(StrEnum):
    """Eval question type label."""

    FACTOID = "factoid"
    DEFINITION = "definition"
    HOW_TO = "how_to"
    COMPARISON = "comparison"
    MULTI_HOP = "multi_hop"
    OPEN_ENDED = "open_ended"
    UNANSWERABLE = "unanswerable"
    OOD = "ood"
    TYPO = "typo"
    AMBIGUOUS = "ambiguous"
    ADVERSARIAL = "adversarial"
    UNKNOWN = "unknown"


def _dedupe_non_empty(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = raw.strip()
        if not value:
            continue
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


class EvalSample(BaseModel):
    """One evaluation sample with source-grounded expectations."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str
    question: str
    subset: EvalSubset
    type: EvalQueryType
    difficulty: EvalDifficulty = EvalDifficulty.UNKNOWN
    expected_behavior: EvalExpectedBehavior
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_urls: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    source_chunk_id: str | None = None
    source_document_id: str | None = None
    source_url: str | None = None
    source_title: str | None = None
    source_section: str | None = None
    created_at: str | None = None

    @field_validator("id", "question")
    @classmethod
    def required_non_empty_fields(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Field must not be empty.")
        return normalized

    @field_validator("reference_answer")
    @classmethod
    def optional_non_empty_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        if not normalized:
            return None
        return normalized

    @field_validator("expected_chunk_ids", "expected_document_ids", "expected_urls")
    @classmethod
    def dedupe_expected_identifiers(cls, value: list[str]) -> list[str]:
        return _dedupe_non_empty(value)

    @model_validator(mode="after")
    def validate_expected_behavior_requirements(self) -> EvalSample:
        if self.expected_behavior == EvalExpectedBehavior.ABSTAIN:
            return self

        if self.expected_behavior == EvalExpectedBehavior.ANSWER:
            has_expected_sources = bool(
                self.expected_chunk_ids
                or self.expected_document_ids
                or self.expected_urls
                or self.reference_answer
            )
            if not has_expected_sources and self.subset != EvalSubset.MANUAL:
                raise ValueError(
                    "answer samples should include expected source ids/urls or reference_answer "
                    "unless subset is manual."
                )
        return self


class EvalDatasetManifest(BaseModel):
    """Manifest metadata for one eval dataset snapshot."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dataset_id: str
    dataset_version: str
    created_at: str
    samples_total: int
    subsets: dict[str, int]
    types: dict[str, int]
    difficulties: dict[str, int]
    source_artifacts: dict[str, str]
    build_config: dict[str, Any]
    schema_version: str = "1.0"
    content_hash: str | None = None
    notes: str | None = None


class EvalDataset(BaseModel):
    """In-memory eval dataset with samples and optional manifest."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    samples: list[EvalSample]
    manifest: EvalDatasetManifest | None = None

    def __len__(self) -> int:
        return len(self.samples)

    def subset_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sample in self.samples:
            key = sample.subset.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sample in self.samples:
            key = sample.type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def difficulty_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sample in self.samples:
            key = sample.difficulty.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def answerable_count(self) -> int:
        return sum(1 for sample in self.samples if sample.expected_behavior == EvalExpectedBehavior.ANSWER)

    def abstain_count(self) -> int:
        return sum(1 for sample in self.samples if sample.expected_behavior == EvalExpectedBehavior.ABSTAIN)


class EvalDatasetBuildStats(BaseModel):
    """Counters for dataset builder diagnostics."""

    model_config = ConfigDict(extra="forbid")

    input_chunks_total: int = 0
    eligible_chunks_total: int = 0
    samples_created_total: int = 0
    synthetic_samples_total: int = 0
    negative_samples_total: int = 0
    robustness_samples_total: int = 0
    audit_samples_total: int = 0
    dropped_chunks_total: int = 0
    dropped_reasons: dict[str, int] = Field(default_factory=dict)


class EvalRunnerOptions(BaseModel):
    """Execution options for core eval runner."""

    model_config = ConfigDict(extra="forbid")

    config_path: str = "configs/config.yaml"
    retrieve_top_k: int | None = None
    rerank_top_k_before: int | None = None
    rerank_top_k_after: int | None = None
    context_token_budget: int | None = None
    context_max_chunks: int | None = None
    write_trace: bool = False
    fail_fast: bool = False
    judge_enabled: bool = False
    judge_backend: str = "heuristic"

    @field_validator("config_path")
    @classmethod
    def config_path_non_empty(cls, value: str) -> str:
        normalized = str(Path(value)).strip()
        if not normalized:
            raise ValueError("config_path must not be empty.")
        return normalized

    @field_validator(
        "retrieve_top_k",
        "rerank_top_k_before",
        "rerank_top_k_after",
        "context_token_budget",
        "context_max_chunks",
    )
    @classmethod
    def optional_positive_int(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("Optional override values must be > 0 when provided.")
        return value

    @field_validator("judge_backend")
    @classmethod
    def validate_judge_backend(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"heuristic", "local_llm", "none"}:
            raise ValueError("judge_backend must be one of: heuristic, local_llm, none.")
        return normalized


class RetrievalEvalRecord(BaseModel):
    """Retrieval-stage eval record."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    query: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_document_ids: list[str] = Field(default_factory=list)
    retrieved_urls: list[str] = Field(default_factory=list)
    retrieved_scores: list[float] = Field(default_factory=list)
    results_total: int = 0
    top_k: int | None = None
    strategy: str | None = None
    duration_ms: float = 0.0
    trace_path: str | None = None


class RerankEvalRecord(BaseModel):
    """Rerank-stage eval record."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    reranked_chunk_ids: list[str] = Field(default_factory=list)
    reranked_document_ids: list[str] = Field(default_factory=list)
    reranked_urls: list[str] = Field(default_factory=list)
    input_results_total: int = 0
    reranked_results_total: int = 0
    model_name: str | None = None
    top_k_before: int | None = None
    top_k_after: int | None = None
    latency_budget_exceeded: bool = False
    cache_hits: int = 0
    cache_misses: int = 0
    duration_ms: float = 0.0
    trace_path: str | None = None


class ContextEvalRecord(BaseModel):
    """Context-stage eval record."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    context_chunk_ids: list[str] = Field(default_factory=list)
    context_document_ids: list[str] = Field(default_factory=list)
    context_urls: list[str] = Field(default_factory=list)
    token_count: int = 0
    token_budget: int | None = None
    sources_total: int = 0
    truncated: bool = False
    included_chunks_total: int = 0
    dropped_chunks_total: int = 0
    duration_ms: float = 0.0
    trace_path: str | None = None


class GenerationEvalRecord(BaseModel):
    """Generation-stage eval record."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    answer: str = ""
    confidence: str = "none"
    sources_total: int = 0
    cited_chunk_ids: list[str] = Field(default_factory=list)
    cited_document_ids: list[str] = Field(default_factory=list)
    cited_urls: list[str] = Field(default_factory=list)
    parsed_successfully: bool = False
    provider: str | None = None
    model_name: str | None = None
    duration_ms: float = 0.0
    trace_path: str | None = None


class CitationEvalRecord(BaseModel):
    """Citation-stage eval record."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    context_chunk_ids: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    valid_cited_chunk_ids: list[str] = Field(default_factory=list)
    invented_cited_chunk_ids: list[str] = Field(default_factory=list)
    cited_document_ids: list[str] = Field(default_factory=list)
    cited_urls: list[str] = Field(default_factory=list)


class JudgeRecord(BaseModel):
    """Judge outcome for one generated answer."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    groundedness_score: float = 0.0
    relevance_score: float = 0.0
    source_support_score: float = 0.0
    completeness_score: float = 0.0
    hallucination_risk: float = 0.0
    verdict: str = "warn"
    reason: str | None = None
    judge_backend: str = "heuristic"
    raw_output: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalCaseResult(BaseModel):
    """Case-level result for a finished eval run."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    question: str = ""
    subset: str = ""
    type: str = ""
    difficulty: str = ""
    expected_behavior: str = ""
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_urls: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    retrieval: RetrievalEvalRecord | None = None
    rerank: RerankEvalRecord | None = None
    context: ContextEvalRecord | None = None
    generation: GenerationEvalRecord | None = None
    citation: CitationEvalRecord | None = None
    judge: JudgeRecord | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    trace_paths: dict[str, str | None] = Field(default_factory=dict)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    passed: bool = False
    notes: str | None = None
    error: str | None = None


class EvalRunPaths(BaseModel):
    """Filesystem paths of one eval run output."""

    model_config = ConfigDict(extra="forbid")

    run_dir: str
    manifest_path: str
    cases_path: str | None = None
    metrics_path: str | None = None
    summary_path: str | None = None
    failures_path: str | None = None
    worst_cases_path: str | None = None
    report_path: str | None = None


class EvalRunManifest(BaseModel):
    """Top-level metadata for one eval run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    schema_version: str = "1.0"
    dataset_id: str | None = None
    dataset_path: str | None = None
    mode: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_ms: float | None = None
    config_path: str | None = None
    artifact_paths: EvalRunPaths | None = None
    build_config: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class EvalRunSummary(BaseModel):
    """Aggregate summary for one eval run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: str = "success"
    cases_total: int = 0
    cases_successful: int = 0
    cases_failed: int = 0
    failure_rate: float = 0.0
    started_at: str | None = None
    finished_at: str | None = None
    duration_ms: float | None = None
    metric_groups: list[str] = Field(default_factory=list)
    overall_score: float = 0.0
    latency: dict[str, float] = Field(default_factory=dict)
    dataset_id: str | None = None
    mode: str | None = None
    notes: str | None = None
