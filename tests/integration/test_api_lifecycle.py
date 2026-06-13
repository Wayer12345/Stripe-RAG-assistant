"""Integration tests for API lifecycle and health route."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import main as app_main
import pytest
from app.utils.config import Settings, load_settings
from fastapi.testclient import TestClient


@dataclass
class _FakeWarmup:
    status: str
    components: dict[str, dict[str, Any]]


class _FakeApiQueryService:
    warmup_called = False
    shutdown_called = False

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.last_warmup_result = None

    def warmup(self) -> _FakeWarmup:
        _FakeApiQueryService.warmup_called = True
        result = _FakeWarmup(
            status="success",
            components={
                "retrieval": {"qdrant_healthcheck_ok": True},
                "generation": {"ollama_healthcheck_ok": True},
            },
        )
        self.last_warmup_result = result
        return result

    def shutdown(self) -> None:
        _FakeApiQueryService.shutdown_called = True


@pytest.mark.integration
def test_startup_creates_service_and_calls_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    settings: Settings = load_settings().model_copy(
        deep=True,
        update={
            "api": load_settings().api.model_copy(
                update={"warmup_on_startup": True, "shutdown_on_exit": True}
            )
        },
    )
    monkeypatch.setattr(app_main, "load_settings", lambda: settings)
    monkeypatch.setattr(app_main, "ApiQueryService", _FakeApiQueryService)

    with TestClient(app_main.app) as client:
        assert _FakeApiQueryService.warmup_called is True
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["warmup"] is not None
        assert payload["dependencies"]["service_initialized"] is True

    assert _FakeApiQueryService.shutdown_called is True
