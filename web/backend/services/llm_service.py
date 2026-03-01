"""LLM orchestration service for Ollama-backed endpoints."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from urllib import error as url_error
from urllib import request as url_request
from urllib.parse import urlparse

from constants import CONFIRMATION_MODEL, OFFER_MODEL
from llm_common import (
    MODELS,
    query_ollama_detailed,
    query_ollama_raw_detailed,
)
from prompt_templates import PROMPT_STYLES, get_prompt
from web.backend.schemas import LlmExplainRequest, LlmSuggestRequest


def _safe_excerpt(text: str | None, length: int = 320) -> str | None:
    if not text:
        return None
    clean = text.strip()
    if len(clean) <= length:
        return clean
    return clean[:length].rstrip() + "..."


@dataclass
class LlmSuggestResult:
    vector: dict[str, Any] | None
    warnings: list[str]
    parse_status: str
    raw_excerpt: str | None
    raw_response: str | None
    provider: dict[str, Any]
    prompt: str
    model: str


@dataclass
class LlmExplainResult:
    explanation: str | None
    raw_response: str | None
    provider: dict[str, Any]
    prompt: str
    model: str


class LlmService:
    """High-level LLM request handler for the web API."""

    def suggest(self, request: LlmSuggestRequest) -> LlmSuggestResult:
        model = request.model or OFFER_MODEL
        prompt_template = get_prompt(request.style, "offer")
        prompt = prompt_template.format(scenario=request.scenario)

        detailed = query_ollama_detailed(
            model=model,
            prompt=prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            min_intervention_keys=request.min_intervention_keys,
        )
        provider = detailed.get("provider", {})
        warnings = list(detailed.get("warnings", []))
        raw_response = detailed.get("raw_response")
        parse_status = str(detailed.get("parse_status", "provider_error"))

        if not detailed.get("ok"):
            provider_type = provider.get("error_type") or "provider_error"
            provider_msg = provider.get("error_message") or "LLM call failed"
            warnings.append(f"{provider_type}: {provider_msg}")

        return LlmSuggestResult(
            vector=detailed.get("vector"),
            warnings=warnings,
            parse_status=parse_status,
            raw_excerpt=_safe_excerpt(raw_response),
            raw_response=raw_response,
            provider=provider,
            prompt=prompt,
            model=model,
        )

    def explain(self, request: LlmExplainRequest) -> LlmExplainResult:
        model = request.model or CONFIRMATION_MODEL
        prompt = self._build_explain_prompt(request)
        provider = query_ollama_raw_detailed(
            model=model,
            prompt=prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        if not provider.get("ok"):
            return LlmExplainResult(
                explanation=None,
                raw_response=None,
                provider=provider,
                prompt=prompt,
                model=model,
            )
        response_text = provider.get("response_text")
        return LlmExplainResult(
            explanation=response_text.strip() if isinstance(response_text, str) else None,
            raw_response=response_text if isinstance(response_text, str) else None,
            provider=provider,
            prompt=prompt,
            model=model,
        )

    def model_catalog(self) -> list[dict[str, Any]]:
        return [dict(m) for m in MODELS]

    def prompt_styles(self) -> list[str]:
        return list(PROMPT_STYLES.keys())

    def health(self, ollama_generate_url: str) -> dict[str, Any]:
        parsed = urlparse(ollama_generate_url)
        tags_url = f"{parsed.scheme}://{parsed.netloc}/api/tags"
        try:
            req = url_request.Request(tags_url, method="GET")
            with url_request.urlopen(req, timeout=3) as resp:
                body = resp.read().decode("utf-8")
                data = json.loads(body)
            models = data.get("models", [])
            names = [m.get("name") for m in models if isinstance(m, dict) and m.get("name")]
            return {
                "ok": True,
                "url": tags_url,
                "models": names,
            }
        except (url_error.URLError, TimeoutError, json.JSONDecodeError):
            return {
                "ok": False,
                "url": tags_url,
                "models": [],
            }

    @staticmethod
    def _build_explain_prompt(request: LlmExplainRequest) -> str:
        summary_blob = json.dumps(request.summary, indent=2, sort_keys=True)
        analytics_blob = json.dumps(request.analytics, indent=2, sort_keys=True)
        scenario = request.scenario or "N/A"

        return (
            "You are reviewing a mitochondrial simulation run for a local "
            "single-user workbench.\n"
            "Write a concise interpretation in plain language:\n"
            "1) What improved or worsened.\n"
            "2) Most likely mechanism behind the trajectory.\n"
            "3) One practical next adjustment.\n\n"
            f"Scenario:\n{scenario}\n\n"
            f"Summary JSON:\n{summary_blob}\n\n"
            f"Analytics JSON:\n{analytics_blob}\n\n"
            "Keep the response under 180 words."
        )
