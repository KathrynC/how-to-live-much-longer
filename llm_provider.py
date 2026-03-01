"""Typed LLM provider abstraction with Ollama implementation.

This module keeps transport concerns (HTTP retries, timeout, backoff,
error typing) separate from prompt/parsing logic in llm_common.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any, Protocol
from urllib import error as url_error
from urllib import request as url_request


class LLMError(Exception):
    """Base class for provider/transport errors."""


class LLMTimeoutError(LLMError):
    """Raised when the provider times out."""


class LLMTransportError(LLMError):
    """Raised for network/transport-level failures."""


class LLMResponseError(LLMError):
    """Raised for malformed provider responses."""


@dataclass
class LLMRequest:
    """Structured generation request."""
    model: str
    prompt: str
    temperature: float = 0.8
    max_tokens: int = 800
    timeout: int = 180
    retries: int = 2
    backoff_sec: float = 0.4


@dataclass
class LLMResult:
    """Structured generation response."""
    ok: bool
    response_text: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    status_code: int | None = None
    attempts_used: int = 0
    latency_sec: float = 0.0
    raw_payload: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    """Protocol for provider implementations."""
    def generate(self, request: LLMRequest) -> LLMResult:
        """Generate text for the given request."""


class OllamaProvider:
    """Ollama HTTP client with retries and typed error reporting."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def _build_payload(self, req: LLMRequest) -> bytes:
        payload = {
            "model": req.model,
            "prompt": req.prompt,
            "stream": False,
            "options": {
                "temperature": req.temperature,
                "num_predict": req.max_tokens,
            },
        }
        return json.dumps(payload).encode("utf-8")

    def generate(self, request: LLMRequest) -> LLMResult:
        attempts = max(1, request.retries + 1)
        start = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                req_obj = url_request.Request(
                    self.base_url,
                    data=self._build_payload(request),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with url_request.urlopen(req_obj, timeout=request.timeout) as resp:
                    status_code = getattr(resp, "status", None)
                    raw = resp.read().decode("utf-8")

                data = json.loads(raw)
                if "error" in data:
                    return LLMResult(
                        ok=False,
                        error_type="provider_error",
                        error_message=str(data.get("error")),
                        status_code=status_code,
                        attempts_used=attempt,
                        latency_sec=time.perf_counter() - start,
                        raw_payload=data,
                    )

                response_text = data.get("response")
                if not isinstance(response_text, str):
                    raise LLMResponseError("missing or non-string 'response' in provider payload")

                return LLMResult(
                    ok=True,
                    response_text=response_text,
                    status_code=status_code,
                    attempts_used=attempt,
                    latency_sec=time.perf_counter() - start,
                    raw_payload=data,
                )

            except TimeoutError as exc:
                last_error = LLMTimeoutError(str(exc))
            except url_error.HTTPError as exc:
                last_error = LLMTransportError(f"HTTP {exc.code}: {exc.reason}")
            except url_error.URLError as exc:
                last_error = LLMTransportError(str(exc.reason))
            except json.JSONDecodeError as exc:
                last_error = LLMResponseError(f"invalid JSON from provider: {exc}")
            except Exception as exc:
                last_error = exc

            if attempt < attempts:
                time.sleep(request.backoff_sec * attempt)

        return LLMResult(
            ok=False,
            error_type=type(last_error).__name__ if last_error is not None else "LLMError",
            error_message=str(last_error) if last_error is not None else "unknown provider failure",
            attempts_used=attempts,
            latency_sec=time.perf_counter() - start,
        )
