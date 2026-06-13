"""Dataset loading, validation, filtering, and export helpers."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.evaluation.records import EvalDataset, EvalDatasetManifest, EvalSample, EvalSubset
from app.infrastructure.storage.jsonl_store import write_jsonl
from app.infrastructure.storage.manifest_store import read_manifest, write_manifest
from app.utils.constants import ARTIFACT_SCHEMA_VERSION
from app.utils.hashing import sha256_text


def _as_path(path: Path | str) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _infer_dataset_id_from_samples_path(path: Path) -> str:
    if path.name == "dataset.jsonl":
        return path.parent.name
    return path.stem


def load_eval_samples(path: Path | str) -> list[EvalSample]:
    """Load and validate eval samples from a JSONL file."""
    samples = _load_eval_samples_without_unique_validation(path)
    return validate_eval_samples(samples)


def _load_eval_samples_without_unique_validation(path: Path | str) -> list[EvalSample]:
    """Load and validate eval samples from JSONL without unique-id checks."""
    samples_path = _as_path(path)
    if not samples_path.exists():
        raise FileNotFoundError(f"Eval samples file not found: {samples_path}")
    if not samples_path.is_file():
        raise ValueError(f"Eval samples path is not a file: {samples_path}")

    samples: list[EvalSample] = []
    with samples_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"Invalid JSON in {samples_path} at line {line_number}: {err.msg}"
                ) from err

            sample_id = payload.get("id", "<missing-id>") if isinstance(payload, dict) else "<non-object>"
            try:
                samples.append(EvalSample.model_validate(payload))
            except ValidationError as err:
                raise ValueError(
                    f"Invalid eval sample in {samples_path} at line {line_number} "
                    f"(sample_id={sample_id}): {err}"
                ) from err

    return samples


def write_eval_samples(path: Path | str, samples: list[EvalSample]) -> None:
    """Write eval samples into JSONL format."""
    output_path = _as_path(path)
    write_jsonl(output_path, samples)


def write_eval_dataset_manifest(path: Path | str, manifest: EvalDatasetManifest) -> None:
    """Write eval dataset manifest as formatted JSON."""
    output_path = _as_path(path)
    write_manifest(output_path, manifest)


def validate_eval_samples(samples: list[EvalSample]) -> list[EvalSample]:
    """Validate uniqueness and return a copied list."""
    seen_ids: set[str] = set()
    duplicates: set[str] = set()
    for sample in samples:
        if sample.id in seen_ids:
            duplicates.add(sample.id)
            continue
        seen_ids.add(sample.id)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate eval sample ids found: {duplicate_list}")
    return list(samples)


def dedupe_eval_sample_ids(samples: list[EvalSample]) -> tuple[list[EvalSample], int]:
    """Return samples with unique ids and total renamed duplicates."""
    if not samples:
        return [], 0

    deduped: list[EvalSample] = []
    seen_count: dict[str, int] = {}
    occupied_ids: set[str] = set()
    renamed_total = 0

    for sample in samples:
        base_id = sample.id
        base_seen = seen_count.get(base_id, 0)
        seen_count[base_id] = base_seen + 1

        if base_seen == 0 and base_id not in occupied_ids:
            final_id = base_id
        else:
            suffix = base_seen
            final_id = f"{base_id}__dup{suffix}"
            while final_id in occupied_ids:
                suffix += 1
                final_id = f"{base_id}__dup{suffix}"
            renamed_total += 1

        occupied_ids.add(final_id)
        if final_id == sample.id:
            deduped.append(sample)
        else:
            deduped.append(sample.model_copy(update={"id": final_id}))

    return deduped, renamed_total


def ensure_unique_eval_samples_file(path: Path | str) -> tuple[Path, int]:
    """Rewrite dataset file with unique sample ids when duplicates exist."""
    dataset_path = _as_path(path)
    raw_samples = _load_eval_samples_without_unique_validation(dataset_path)
    deduped_samples, renamed_total = dedupe_eval_sample_ids(raw_samples)
    if renamed_total == 0:
        return dataset_path, 0

    write_eval_samples(dataset_path, deduped_samples)
    manifest_path = dataset_path.parent / "manifest.json"
    if manifest_path.exists():
        manifest_payload = read_manifest(manifest_path)
        payload = manifest_payload if isinstance(manifest_payload, dict) else {}
        source_artifacts = payload.get("source_artifacts")
        build_config = payload.get("build_config")
        manifest = build_dataset_manifest(
            dataset_id=str(payload.get("dataset_id") or _infer_dataset_id_from_samples_path(dataset_path)),
            samples=deduped_samples,
            source_artifacts=source_artifacts if isinstance(source_artifacts, dict) else {},
            build_config=build_config if isinstance(build_config, dict) else {},
            dataset_version=str(payload.get("dataset_version") or "v1"),
            notes=payload.get("notes") if isinstance(payload.get("notes"), str) else None,
        )
        write_eval_dataset_manifest(manifest_path, manifest)

    return dataset_path, renamed_total


def filter_eval_samples(
    samples: list[EvalSample],
    *,
    subsets: set[str] | None = None,
    types: set[str] | None = None,
    difficulties: set[str] | None = None,
    expected_behaviors: set[str] | None = None,
    limit: int | None = None,
    seed: int | None = None,
) -> list[EvalSample]:
    """Filter samples by dimensions and optionally shuffle deterministically."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be > 0 when provided.")

    filtered: list[EvalSample] = []
    for sample in samples:
        if subsets is not None and sample.subset.value not in subsets:
            continue
        if types is not None and sample.type.value not in types:
            continue
        if difficulties is not None and sample.difficulty.value not in difficulties:
            continue
        if expected_behaviors is not None and sample.expected_behavior.value not in expected_behaviors:
            continue
        filtered.append(sample)

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(filtered)

    if limit is not None:
        return filtered[:limit]
    return filtered


def split_eval_samples_by_subset(samples: list[EvalSample]) -> dict[str, list[EvalSample]]:
    """Group samples by subset."""
    grouped: dict[str, list[EvalSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.subset.value].append(sample)
    return dict(grouped)


def split_eval_samples_by_type(samples: list[EvalSample]) -> dict[str, list[EvalSample]]:
    """Group samples by query type."""
    grouped: dict[str, list[EvalSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.type.value].append(sample)
    return dict(grouped)


def _build_content_hash(samples: list[EvalSample]) -> str | None:
    if not samples:
        return None
    payload = "\n".join(
        sample.model_dump_json(by_alias=False, exclude_none=False) for sample in samples
    )
    return sha256_text(payload)


def build_dataset_manifest(
    *,
    dataset_id: str,
    samples: list[EvalSample],
    source_artifacts: dict[str, str] | None = None,
    build_config: dict[str, Any] | None = None,
    dataset_version: str = "v1",
    notes: str | None = None,
) -> EvalDatasetManifest:
    """Build a dataset manifest from in-memory samples."""
    validated_samples = validate_eval_samples(samples)
    dataset = EvalDataset(dataset_id=dataset_id, samples=validated_samples)
    return EvalDatasetManifest(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        created_at=datetime.now(UTC).isoformat(),
        samples_total=len(validated_samples),
        subsets=dataset.subset_counts(),
        types=dataset.type_counts(),
        difficulties=dataset.difficulty_counts(),
        source_artifacts=source_artifacts or {},
        build_config=build_config or {},
        schema_version=ARTIFACT_SCHEMA_VERSION,
        content_hash=_build_content_hash(validated_samples),
        notes=notes,
    )


def load_eval_dataset(path: Path | str) -> EvalDataset:
    """Load an eval dataset from canonical JSONL file."""
    dataset_path = _as_path(path)
    samples = load_eval_samples(dataset_path)
    return EvalDataset(
        dataset_id=_infer_dataset_id_from_samples_path(dataset_path),
        samples=samples,
        manifest=None,
    )


def load_eval_dataset_dir(dataset_dir: Path | str) -> EvalDataset:
    """Load eval dataset from a directory with dataset/manifest files."""
    base_dir = _as_path(dataset_dir)
    dataset_path = base_dir / "dataset.jsonl"
    manifest_path = base_dir / "manifest.json"

    samples = load_eval_samples(dataset_path)
    manifest: EvalDatasetManifest | None = None
    if manifest_path.exists():
        try:
            manifest = EvalDatasetManifest.model_validate(read_manifest(manifest_path))
        except ValidationError as err:
            raise ValueError(f"Invalid dataset manifest at {manifest_path}: {err}") from err

    dataset_id = manifest.dataset_id if manifest is not None else base_dir.name
    return EvalDataset(dataset_id=dataset_id, samples=samples, manifest=manifest)


def export_eval_dataset_dir(
    dataset_dir: Path | str,
    *,
    dataset_id: str,
    samples: list[EvalSample],
    source_artifacts: dict[str, str] | None = None,
    build_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Export canonical dataset files into a dataset directory."""
    validated_samples = validate_eval_samples(samples)
    output_dir = _as_path(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "dataset.jsonl"
    write_eval_samples(dataset_path, validated_samples)

    manifest = build_dataset_manifest(
        dataset_id=dataset_id,
        samples=validated_samples,
        source_artifacts=source_artifacts,
        build_config=build_config,
    )
    manifest_path = output_dir / "manifest.json"
    write_eval_dataset_manifest(manifest_path, manifest)

    outputs: dict[str, Path] = {
        "dataset": dataset_path,
        "manifest": manifest_path,
    }

    audit_samples = [sample for sample in validated_samples if sample.subset == EvalSubset.AUDIT]
    if audit_samples:
        audit_path = output_dir / "audit.jsonl"
        write_eval_samples(audit_path, audit_samples)
        outputs["audit"] = audit_path

    return outputs
