"""Model backends for running delay-discounting experiments."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import random
import time
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)


class TransientModelError(Exception):
    """Raised for transient model/query failures."""


@dataclass
class ModelResponse:
    """Unified model response."""

    raw_text: str
    model_id: str
    latency: float
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "raw_text": self.raw_text,
            "model_id": self.model_id,
            "latency": self.latency,
            "error": self.error,
        }


class BaseModelClient:
    """Base class implementing rate limiting and retries."""

    def __init__(
        self,
        model_id: str,
        temperature: float,
        requests_per_minute: int,
        seed: int,
        max_retries: int = 5,
    ) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.requests_per_minute = max(1, int(requests_per_minute))
        self.max_retries = max_retries
        self._min_interval = 60.0 / self.requests_per_minute
        self._last_call_ts = 0.0
        self._rng = random.Random(seed)

    def _respect_rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call_ts
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_ts = time.monotonic()

    def _should_retry(self, error_text: str) -> bool:
        message = error_text.lower()
        tokens = [
            "429",
            "rate limit",
            "temporar",
            "timeout",
            "timed out",
            "connection",
            "unavailable",
            "internal",
            "bad gateway",
            "service unavailable",
        ]
        return any(token in message for token in tokens)

    def query(self, prompt: str) -> dict[str, Any]:
        """Query model with retries and backoff."""
        backoff = 1.0
        for attempt in range(1, self.max_retries + 1):
            start = time.perf_counter()
            self._respect_rate_limit()
            try:
                raw_text = self._query_once(prompt)
                latency = time.perf_counter() - start
                return ModelResponse(
                    raw_text=raw_text,
                    model_id=self.model_id,
                    latency=latency,
                    error=None,
                ).to_dict()
            except Exception as exc:  # noqa: BLE001
                latency = time.perf_counter() - start
                error_text = str(exc)
                if attempt < self.max_retries and self._should_retry(error_text):
                    sleep_s = backoff + self._rng.uniform(0.0, 0.25)
                    LOGGER.warning(
                        "Transient error for model %s on attempt %s/%s: %s. Retrying in %.2fs",
                        self.model_id,
                        attempt,
                        self.max_retries,
                        error_text,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    backoff *= 2.0
                    continue
                return ModelResponse(
                    raw_text="",
                    model_id=self.model_id,
                    latency=latency,
                    error=error_text,
                ).to_dict()
        return ModelResponse(raw_text="", model_id=self.model_id, latency=0.0, error="Unknown error").to_dict()

    def _query_once(self, prompt: str) -> str:
        raise NotImplementedError


class GeminiClient(BaseModelClient):
    """Gemini backend via google-genai."""

    def __init__(self, model_id: str, temperature: float, requests_per_minute: int, seed: int) -> None:
        super().__init__(model_id, temperature, requests_per_minute, seed)
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")

        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise ImportError("google-genai is required for gemini backend") from exc

        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=api_key)

    def _query_once(self, prompt: str) -> str:
        try:
            config = self._types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=4,
                thinking_config=self._types.ThinkingConfig(thinking_budget=0),
            )
        except Exception:
            config = self._types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=4,
            )

        response = self._client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=config,
        )
        text = getattr(response, "text", None)
        if text is None:
            raise TransientModelError("Gemini returned empty text")
        return text.strip()


class OllamaClient(BaseModelClient):
    """Ollama backend via local HTTP API."""

    def __init__(self, model_id: str, temperature: float, requests_per_minute: int, seed: int) -> None:
        super().__init__(model_id, temperature, requests_per_minute, seed)
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

    def _query_once(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_id,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                },
            },
            timeout=90,
        )
        if response.status_code >= 500:
            raise TransientModelError(f"Ollama server error: {response.status_code}")
        response.raise_for_status()
        payload = response.json()
        text = payload.get("response", "")
        if not isinstance(text, str):
            raise ValueError("Ollama response missing 'response' text")
        return text.strip()


class MockClient(BaseModelClient):
    """Mock backend for pipeline tests; no network."""

    def _query_once(self, prompt: str) -> str:  # noqa: ARG002
        return self._rng.choice(["A", "B"])


def build_model_client(model_cfg: dict[str, Any], temperature: float, requests_per_minute: int, seed: int) -> BaseModelClient:
    """Factory to build model backend clients."""
    backend = str(model_cfg["backend"]).lower()
    model_id = str(model_cfg["model_id"])
    if backend == "gemini":
        return GeminiClient(model_id=model_id, temperature=temperature, requests_per_minute=requests_per_minute, seed=seed)
    if backend == "ollama":
        return OllamaClient(model_id=model_id, temperature=temperature, requests_per_minute=requests_per_minute, seed=seed)
    if backend == "mock":
        return MockClient(model_id=model_id, temperature=temperature, requests_per_minute=requests_per_minute, seed=seed)
    raise ValueError(f"Unsupported backend: {backend}")
