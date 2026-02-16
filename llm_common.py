"""Shared LLM utilities for seed experiments.

Analogous to structured_random_common.py in the parent Evolutionary-Robotics
project. Provides Ollama query, response parsing, and grid snapping for
the 12D intervention+patient parameter space.

Usage:
    from llm_common import query_ollama, parse_intervention_vector
    vector, raw = query_ollama("qwen3-coder:30b", prompt)
"""
from __future__ import annotations

import json
import subprocess

from constants import (
    OLLAMA_URL, REASONING_MODELS,
    INTERVENTION_NAMES, PATIENT_NAMES, ALL_PARAM_NAMES,
    DEFAULT_INTERVENTION, DEFAULT_PATIENT,
    snap_all,
)
from schemas import FullProtocol


# ── Models ──────────────────────────────────────────────────────────────────

MODELS = [
    {"name": "qwen3-coder:30b", "type": "ollama"},
    {"name": "deepseek-r1:8b", "type": "ollama"},
    {"name": "llama3.1:latest", "type": "ollama"},
    {"name": "gpt-oss:20b", "type": "ollama"},
]


# ── Text cleaning ──────────────────────────────────────────────────────────

def strip_think_tags(text):
    """Remove <think>...</think> reasoning tokens from model output."""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


def strip_markdown_fences(text):
    """Extract content from markdown code fences."""
    if "```" not in text:
        return text
    parts = text.split("```")
    for part in parts:
        part = part.strip()
        if part.startswith("json"):
            part = part[4:].strip()
        if part.startswith("{"):
            return part
    return text


# ── Response parsing ───────────────────────────────────────────────────────

def parse_json_response(response):
    """Parse a JSON object from an LLM response, handling common artifacts.

    Strips markdown code fences, <think>...</think> tags, and finds the
    outermost { ... } pair. Returns the raw parsed dict without any
    domain-specific validation or snapping.

    Args:
        response: Raw string from LLM.

    Returns:
        Parsed dict, or None on failure.
    """
    if not response:
        return None

    text = strip_think_tags(response.strip())
    text = strip_markdown_fences(text)

    # Find outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def detect_flattening(params):
    """Detect and fix common LLM flattening errors.

    Zimmerman (2025) §3.5.3: tokenization-induced flattening collapses
    qualitative distinctions.
    A common failure mode is treating all parameters as 0-1 normalized,
    producing baseline_age=0.5 instead of 50.

    Args:
        params: Dict of raw parameter values from LLM.

    Returns:
        Dict with corrected values and list of fixes applied.
    """
    fixes = []
    corrected = dict(params)

    # Age: if value is 0-1, it was likely normalized (should be 20-90)
    if "baseline_age" in corrected:
        age = corrected["baseline_age"]
        if 0 <= age <= 1.0:
            corrected["baseline_age"] = 20 + age * 70  # rescale to 20-90
            fixes.append(f"baseline_age {age} → {corrected['baseline_age']:.0f} "
                         "(rescaled from 0-1 to 20-90)")
        elif 1.0 < age < 20:
            corrected["baseline_age"] = max(20, age * 10)
            fixes.append(f"baseline_age {age} → {corrected['baseline_age']:.0f} "
                         "(likely scale error)")

    # Genetic vulnerability: if value is 0-1, it was likely normalized
    if "genetic_vulnerability" in corrected:
        gv = corrected["genetic_vulnerability"]
        if 0 <= gv <= 0.49:
            corrected["genetic_vulnerability"] = 0.5 + gv
            fixes.append(f"genetic_vulnerability {gv} → "
                         f"{corrected['genetic_vulnerability']:.2f} "
                         "(rescaled from 0-0.5 to 0.5-1.0)")

    # Metabolic demand: similar issue
    if "metabolic_demand" in corrected:
        md = corrected["metabolic_demand"]
        if 0 <= md <= 0.49:
            corrected["metabolic_demand"] = 0.5 + md
            fixes.append(f"metabolic_demand {md} → "
                         f"{corrected['metabolic_demand']:.2f} "
                         "(rescaled from 0-0.5 to 0.5-1.0)")

    return corrected, fixes


def validate_llm_output(raw_dict: dict) -> tuple[dict, list[str]]:
    """Validate a raw parameter dict using pydantic schema.

    Applies FullProtocol validation, then detect_flattening(), then snap_all().
    Warns on out-of-range values but does not reject if at least 6 intervention
    params are present.

    Args:
        raw_dict: Raw parameter dict from JSON parsing.

    Returns:
        (validated_and_snapped_dict, list_of_warnings)
    """
    warnings: list[str] = []
    recognized = {k: v for k, v in raw_dict.items() if k in ALL_PARAM_NAMES}

    # Try pydantic validation — collect errors but don't reject
    try:
        validated = FullProtocol(**recognized)
        # Use only non-None validated values
        recognized = {k: v for k, v in validated.model_dump().items()
                      if v is not None}
    except Exception as e:
        warnings.append(f"pydantic validation: {e}")
        # Fall through with recognized dict as-is (backwards compatible)

    # Detect and fix flattening errors (Zimmerman §3.5.3)
    recognized, fixes = detect_flattening(recognized)
    warnings.extend(fixes)

    return snap_all(recognized), warnings


def parse_intervention_vector(response: str) -> dict | None:
    """Parse a 12D intervention+patient vector from LLM response.

    Handles think tags, markdown fences, extracts the outermost JSON
    object, validates via pydantic schema, detects flattening errors,
    and snaps to grid.

    Args:
        response: Raw string from LLM.

    Returns:
        Dict with snapped parameter values, or None.
    """
    obj = parse_json_response(response)
    if obj is None:
        return None

    # Check we got at least the 6 intervention params
    recognized = {k: v for k, v in obj.items() if k in ALL_PARAM_NAMES}
    if len(recognized) < 6:
        return None

    snapped, _ = validate_llm_output(recognized)
    return snapped


def split_vector(snapped):
    """Split a snapped 12D vector into intervention and patient dicts.

    Args:
        snapped: Dict from parse_intervention_vector().

    Returns:
        (intervention_dict, patient_dict) tuple, with defaults for missing keys.
    """
    intervention = {k: snapped.get(k, DEFAULT_INTERVENTION[k])
                    for k in INTERVENTION_NAMES}
    patient = {k: snapped.get(k, DEFAULT_PATIENT[k])
               for k in PATIENT_NAMES}
    return intervention, patient


# ── Ollama query ───────────────────────────────────────────────────────────

def query_ollama(model, prompt, temperature=0.8, max_tokens=800, timeout=180):
    """Query Ollama and return (parsed_vector, raw_response).

    Args:
        model: Ollama model name (e.g. "qwen3-coder:30b").
        prompt: The prompt string.
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate.
        timeout: Request timeout in seconds.

    Returns:
        (snapped_vector_dict_or_None, raw_response_string)
    """
    effective_max = 3000 if model in REASONING_MODELS else max_tokens
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": effective_max},
    })
    try:
        r = subprocess.run(
            ["curl", "-s", OLLAMA_URL, "-d", payload],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode != 0:
            return None, f"curl error: {r.stderr}"
        data = json.loads(r.stdout)
        if "error" in data:
            return None, f"ollama error: {data['error']}"
        resp = data["response"]
        vector = parse_intervention_vector(resp)
        return vector, resp
    except Exception as e:
        return None, str(e)


def query_ollama_raw(model, prompt, temperature=0.8, max_tokens=800, timeout=180):
    """Query Ollama and return raw response only (no parsing).

    Args:
        model: Ollama model name.
        prompt: The prompt string.

    Returns:
        Raw response string, or None.
    """
    effective_max = 3000 if model in REASONING_MODELS else max_tokens
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": effective_max},
    })
    try:
        r = subprocess.run(
            ["curl", "-s", OLLAMA_URL, "-d", payload],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout)
        if "error" in data:
            return None
        return data["response"]
    except Exception:
        return None
