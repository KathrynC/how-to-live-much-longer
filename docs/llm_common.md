# llm_common

Shared LLM utilities for query, parsing, validation, and flattening detection.

---

## Overview

Provides the complete LLM→simulation pipeline: query Ollama, strip artifacts (`<think>` tags, markdown fences), extract JSON, validate via pydantic schema, detect flattening errors, and snap to grid.

---

## Key Functions

### Query

**`query_ollama(model, prompt, temperature, max_tokens, timeout) → (dict|None, str)`**

Query Ollama and return `(parsed_vector, raw_response)`. The parsed vector is fully validated and grid-snapped. Returns `(None, error_string)` on failure.

**`query_ollama_raw(model, prompt, ...) → str|None`**

Query Ollama and return raw response string only (no parsing).

### Parsing

**`parse_json_response(response) → dict|None`**

Extract a JSON object from an LLM response. Strips `<think>` tags, markdown fences, finds outermost `{...}`.

**`parse_intervention_vector(response) → dict|None`**

Full pipeline: parse JSON → validate via pydantic → detect flattening → snap to grid. Returns 12D dict or None.

### Validation

**`validate_llm_output(raw_dict) → (dict, list[str])`**

Apply pydantic schema validation, flattening detection, and grid snapping. Returns `(snapped_dict, list_of_warnings)`.

**`detect_flattening(params) → (dict, list[str])`**

Detect and fix common LLM flattening errors (Zimmerman §3.5.3). Key corrections:
- `baseline_age` in [0, 1] → rescale to [20, 90]
- `genetic_vulnerability` in [0, 0.49] → shift to [0.5, 1.0]
- `metabolic_demand` in [0, 0.49] → shift to [0.5, 1.0]

### Splitting

**`split_vector(snapped) → (intervention_dict, patient_dict)`**

Split a 12D vector into 6D intervention + 6D patient dicts, filling defaults for missing keys.

### Text Cleaning

- `strip_think_tags(text)` — Remove `<think>...</think>` reasoning tokens
- `strip_markdown_fences(text)` — Extract content from markdown code fences

---

## Models

```python
MODELS = [
    {"name": "qwen3-coder:30b", "type": "ollama"},
    {"name": "deepseek-r1:8b", "type": "ollama"},
    {"name": "llama3.1:latest", "type": "ollama"},
    {"name": "gpt-oss:20b", "type": "ollama"},
]
```

---

## Reference

Zimmerman, J.W. (2025). PhD dissertation, University of Vermont. §3.5.3 (flattening).
