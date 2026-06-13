"""Ollama HTTP client for raw text generation."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import httpx

from app.utils.logging import get_logger

logger = get_logger(__name__)


class OllamaClient:
    """Thin, testable wrapper over local Ollama generation API."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        timeout_seconds: int | float,
        temperature: float,
        max_tokens: int,
        top_p: float | None = None,
        stop: list[str] | None = None,
        keep_alive: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty.")
        if not model_name.strip():
            raise ValueError("model_name must not be empty.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0.")
        if top_p is not None and not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1] when provided.")

        self._base_url = base_url.rstrip("/")
        self._model_name = model_name.strip()
        self._timeout_seconds = float(timeout_seconds)
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._top_p = top_p
        self._stop = list(stop) if stop else None
        self._keep_alive = (
            keep_alive.strip() if isinstance(keep_alive, str) and keep_alive.strip() else None
        )
        self._http_client = http_client
        self._last_stats: dict[str, Any] = {}

    def model_name(self) -> str:
        """Return configured Ollama model name."""
        return self._model_name

    def last_stats(self) -> dict[str, Any]:
        """Return last request stats."""
        return dict(self._last_stats)

    def healthcheck(self) -> bool:
        """Return ``True`` when Ollama server is reachable."""
        url = f"{self._base_url}/api/tags"
        client = self._http_client or httpx.Client(timeout=self._timeout_seconds)
        should_close = self._http_client is None
        try:
            response = client.get(url)
            return response.status_code == 200
        except httpx.HTTPError:
            return False
        finally:
            if should_close:
                client.close()

    def generate(self, prompt: str) -> str:
        """Generate raw text from prompt with local Ollama."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty.")

        payload: dict[str, Any] = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
            "raw": True,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        if self._keep_alive is not None:
            payload["keep_alive"] = self._keep_alive
        if self._top_p is not None:
            payload["options"]["top_p"] = self._top_p
        if self._stop:
            payload["options"]["stop"] = self._stop

        url = f"{self._base_url}/api/generate"
        started = perf_counter()
        client = self._http_client or httpx.Client(timeout=self._timeout_seconds)
        should_close = self._http_client is None

        logger.info(
            "Starting Ollama generation: provider=ollama model_name=%s prompt_chars=%s",
            self._model_name,
            len(prompt),
        )
        try:
            response = client.post(url, json=payload)
        except httpx.TimeoutException as err:
            duration_ms = int((perf_counter() - started) * 1000)
            logger.error(
                "Ollama request timed out: model_name=%s duration_ms=%s",
                self._model_name,
                duration_ms,
            )
            raise RuntimeError(
                "Ollama request timed out. Ensure local Ollama is running and responsive."
            ) from err
        except httpx.HTTPError as err:
            duration_ms = int((perf_counter() - started) * 1000)
            logger.error(
                "Ollama request failed: model_name=%s duration_ms=%s",
                self._model_name,
                duration_ms,
            )
            raise RuntimeError(
                "Ollama is unavailable. Ensure local Ollama is running at the configured base_url."
            ) from err
        finally:
            if should_close:
                client.close()

        duration_ms = int((perf_counter() - started) * 1000)
        if response.status_code != 200:
            logger.error(
                "Ollama returned non-200: status_code=%s model_name=%s duration_ms=%s",
                response.status_code,
                self._model_name,
                duration_ms,
            )
            raise RuntimeError(
                f"Ollama returned non-200 response: {response.status_code} for model {self._model_name!r}."
            )

        try:
            data = response.json()
        except ValueError as err:
            logger.error(
                "Ollama returned invalid JSON response: model_name=%s duration_ms=%s",
                self._model_name,
                duration_ms,
            )
            raise RuntimeError("Ollama returned invalid JSON response shape.") from err

        raw_output = data.get("response")
        if not isinstance(raw_output, str):
            raise RuntimeError("Ollama response missing expected 'response' text field.")
        if not raw_output.strip():
            raise RuntimeError("Ollama returned empty generated text.")

        self._last_stats = {
            "provider": "ollama",
            "model_name": self._model_name,
            "prompt_chars": len(prompt),
            "raw_output_chars": len(raw_output),
            "duration_ms": duration_ms,
            "base_url": self._base_url,
        }
        logger.info(
            "Finished Ollama generation: model_name=%s raw_output_chars=%s duration_ms=%s",
            self._model_name,
            len(raw_output),
            duration_ms,
        )
        return raw_output

    def warmup_generate(
        self,
        *,
        prompt: str = "Respond with: OK",
        max_tokens: int = 8,
    ) -> bool:
        """Run a tiny generate call to warm model weights/runtime."""
        if not prompt.strip():
            raise ValueError("warmup prompt must not be empty.")
        if max_tokens <= 0:
            raise ValueError("warmup max_tokens must be > 0.")

        payload: dict[str, Any] = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
            "raw": True,
            "options": {
                "temperature": 0.0,
                "num_predict": max_tokens,
            },
        }
        if self._keep_alive is not None:
            payload["keep_alive"] = self._keep_alive

        url = f"{self._base_url}/api/generate"
        client = self._http_client or httpx.Client(timeout=self._timeout_seconds)
        should_close = self._http_client is None
        started = perf_counter()
        try:
            response = client.post(url, json=payload)
        except httpx.HTTPError:
            return False
        finally:
            if should_close:
                client.close()
        duration_ms = int((perf_counter() - started) * 1000)
        if response.status_code != 200:
            logger.warning(
                "Ollama warmup generate failed: status_code=%s model_name=%s duration_ms=%s",
                response.status_code,
                self._model_name,
                duration_ms,
            )
            return False
        logger.info(
            "Ollama warmup generate completed: model_name=%s duration_ms=%s",
            self._model_name,
            duration_ms,
        )
        return True
