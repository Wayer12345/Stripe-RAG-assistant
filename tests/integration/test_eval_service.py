"""Integration tests for EvalService orchestration run()."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import app.application.eval_service as eval_service_module
import pytest
from app.application.eval_service import EvalService
from app.evaluation.records import EvalDifficulty, EvalExpectedBehavior, EvalQueryType, EvalSubset


@pytest.mark.integration
class TestEvalService:
    @staticmethod
    def _sample_payload(sample_id: str, question: str) -> dict[str, object]:
        return {
            "id": sample_id,
            "question": question,
            "subset": EvalSubset.SYNTHETIC_SOURCE_GROUNDED.value,
            "type": EvalQueryType.FACTOID.value,
            "difficulty": EvalDifficulty.EASY.value,
            "expected_behavior": EvalExpectedBehavior.ANSWER.value,
            "expected_chunk_ids": ["chunk_1"],
            "expected_document_ids": ["doc_1"],
            "expected_urls": [],
            "metadata": {},
        }

    @pytest.fixture(autouse=True)
    def _skip_preflight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Disable preflight checks so tests work without Ollama/full local stack."""
        monkeypatch.setattr(
            eval_service_module.EvalService,
            "_run_preflight_checks",
            lambda self, settings: None,
        )

    def test_run_raises_without_dataset_or_chunks(self) -> None:
        service = EvalService(config_path=Path("configs/config.yaml"))
        with pytest.raises(ValueError, match="Either dataset_path or chunks_path"):
            service.run()

    def test_run_uses_dataset_path_and_calls_layers_in_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        calls: list[tuple[str, dict[str, Any]]] = []
        dataset = tmp_path / "dataset.jsonl"
        dataset.write_text(
            json.dumps(self._sample_payload("eval_ok", "Valid question")) + "\n",
            encoding="utf-8",
        )

        def make_layer(name: str, run_value: Any = None) -> type:
            class FakeLayer:
                def __init__(self, **kwargs: Any) -> None:
                    calls.append((f"{name}.__init__", kwargs))

                def run(self) -> Any:
                    calls.append((f"{name}.run", {}))
                    return run_value

            return FakeLayer

        monkeypatch.setattr(eval_service_module, "BuildEvalDatasetLayer", make_layer("build_dataset"))
        monkeypatch.setattr(eval_service_module, "RunRetrievalEvalLayer", make_layer("retrieval"))
        monkeypatch.setattr(eval_service_module, "RunRerankEvalLayer", make_layer("rerank"))
        monkeypatch.setattr(eval_service_module, "RunContextEvalLayer", make_layer("context"))
        monkeypatch.setattr(eval_service_module, "RunGenerationEvalLayer", make_layer("generation"))
        monkeypatch.setattr(eval_service_module, "RunCitationEvalLayer", make_layer("citation"))
        monkeypatch.setattr(eval_service_module, "RunRobustnessEvalLayer", make_layer("robustness"))

        service = EvalService(config_path=Path("configs/config.yaml"))
        service.run(dataset_path=dataset, run_id_prefix="smoke")

        init_order = [name for name, _ in calls if name.endswith(".__init__")]
        assert init_order == [
            "retrieval.__init__",
            "rerank.__init__",
            "context.__init__",
            "generation.__init__",
            "citation.__init__",
            "robustness.__init__",
        ]
        retrieval_kwargs = next(kwargs for name, kwargs in calls if name == "retrieval.__init__")
        assert retrieval_kwargs["dataset_path"] == dataset
        assert retrieval_kwargs["run_id"] == "smoke_retrieval"

    def test_run_builds_dataset_when_chunks_provided(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        calls: list[tuple[str, dict[str, Any]]] = []
        chunks = tmp_path / "chunks.jsonl"
        chunks.write_text("{}", encoding="utf-8")
        built_dataset = tmp_path / "datasets" / "built" / "dataset.jsonl"
        built_dataset.parent.mkdir(parents=True, exist_ok=True)
        built_dataset.write_text(
            json.dumps(self._sample_payload("eval_ok", "Valid question")) + "\n",
            encoding="utf-8",
        )

        class FakeBuildLayer:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(("build.__init__", kwargs))

            def run(self) -> Any:
                calls.append(("build.run", {}))
                return type("BuildResult", (), {"dataset_path": built_dataset})()

        class FakeRunLayer:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(("retrieval.__init__", kwargs))

            def run(self) -> Any:
                calls.append(("retrieval.run", {}))
                return object()

        class FakeNoopLayer:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def run(self) -> Any:
                return object()

        monkeypatch.setattr(eval_service_module, "BuildEvalDatasetLayer", FakeBuildLayer)
        monkeypatch.setattr(eval_service_module, "RunRetrievalEvalLayer", FakeRunLayer)
        monkeypatch.setattr(eval_service_module, "RunRerankEvalLayer", FakeNoopLayer)
        monkeypatch.setattr(eval_service_module, "RunContextEvalLayer", FakeNoopLayer)
        monkeypatch.setattr(eval_service_module, "RunGenerationEvalLayer", FakeNoopLayer)
        monkeypatch.setattr(eval_service_module, "RunCitationEvalLayer", FakeNoopLayer)
        monkeypatch.setattr(eval_service_module, "RunRobustnessEvalLayer", FakeNoopLayer)

        service = EvalService(config_path=Path("configs/config.yaml"))
        service.run(chunks_path=chunks)

        build_kwargs = next(kwargs for name, kwargs in calls if name == "build.__init__")
        assert build_kwargs["chunks_path"] == chunks
        retrieval_kwargs = next(kwargs for name, kwargs in calls if name == "retrieval.__init__")
        assert retrieval_kwargs["dataset_path"] == built_dataset

    def test_run_auto_fixes_duplicate_dataset_ids(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        calls: list[tuple[str, dict[str, Any]]] = []
        dataset = tmp_path / "dataset.jsonl"
        rows = [
            self._sample_payload("eval_dup", "Question A"),
            self._sample_payload("eval_dup", "Question B"),
        ]
        dataset.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")

        def make_layer(name: str) -> type:
            class FakeLayer:
                def __init__(self, **kwargs: Any) -> None:
                    calls.append((f"{name}.__init__", kwargs))

                def run(self) -> Any:
                    calls.append((f"{name}.run", {}))
                    return object()

            return FakeLayer

        monkeypatch.setattr(eval_service_module, "RunRetrievalEvalLayer", make_layer("retrieval"))
        monkeypatch.setattr(eval_service_module, "RunRerankEvalLayer", make_layer("rerank"))
        monkeypatch.setattr(eval_service_module, "RunContextEvalLayer", make_layer("context"))
        monkeypatch.setattr(eval_service_module, "RunGenerationEvalLayer", make_layer("generation"))
        monkeypatch.setattr(eval_service_module, "RunCitationEvalLayer", make_layer("citation"))
        monkeypatch.setattr(eval_service_module, "RunRobustnessEvalLayer", make_layer("robustness"))

        service = EvalService(config_path=Path("configs/config.yaml"))
        service.run(dataset_path=dataset, run_id_prefix="smoke")

        from app.evaluation.datasets import load_eval_samples

        ids = [sample.id for sample in load_eval_samples(dataset)]
        assert len(ids) == len(set(ids))
        retrieval_kwargs = next(kwargs for name, kwargs in calls if name == "retrieval.__init__")
        assert retrieval_kwargs["dataset_path"] == dataset
