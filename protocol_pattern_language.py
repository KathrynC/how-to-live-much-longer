"""protocol_pattern_language.py — DAG pipeline definition and orchestrator.

Ported from rosetta-motion's pattern_language.py and pattern_orchestrator.py.
Defines a validated DAG of pipeline stages for protocol curation, with
registered handler functions executed in topological order.
"""
from __future__ import annotations

import copy
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
DEFAULT_PATTERN_LANGUAGE = ROOT / "patterns" / "protocol_pattern_language.v1.json"

VALID_STAGES = {"global", "ingest", "analytics", "robustness", "classify", "review", "report"}
VALID_TIERS = {"core", "adaptable", "experimental"}

PatternHandler = Callable[[dict[str, Any], dict[str, Any]], bool]


def load_default_pattern_language() -> dict:
    """Load the default pattern language JSON."""
    return json.loads(DEFAULT_PATTERN_LANGUAGE.read_text())


def validate_pattern_language(data: dict) -> list[str]:
    """Validate pattern language payload. Returns list of errors."""
    errors: list[str] = []
    if not isinstance(data, dict) or "patterns" not in data:
        return ["root must have 'patterns' key"]

    patterns = data["patterns"]
    if not isinstance(patterns, list) or not patterns:
        return ["'patterns' must be a non-empty array"]

    by_id: dict[str, dict] = {}
    for i, p in enumerate(patterns):
        if not isinstance(p, dict):
            errors.append(f"patterns[{i}] is not an object")
            continue
        pid = p.get("id")
        if not isinstance(pid, str) or not pid:
            errors.append(f"patterns[{i}] has invalid id")
            continue
        if pid in by_id:
            errors.append(f"duplicate pattern id: {pid}")
        by_id[pid] = p
        if p.get("stage") not in VALID_STAGES:
            errors.append(f"pattern {pid} has invalid stage: {p.get('stage')}")
        if p.get("confidence_tier") not in VALID_TIERS:
            errors.append(f"pattern {pid} has invalid tier: {p.get('confidence_tier')}")

    if errors:
        return errors

    # Link integrity
    for pid, p in by_id.items():
        for parent in p.get("larger_patterns", []):
            if parent not in by_id:
                errors.append(f"pattern {pid} references unknown larger pattern: {parent}")
        for child in p.get("smaller_patterns", []):
            if child not in by_id:
                errors.append(f"pattern {pid} references unknown smaller pattern: {child}")

    if errors:
        return errors

    # Cycle detection via topological sort
    in_deg = {pid: 0 for pid in by_id}
    out = defaultdict(list)
    for pid, p in by_id.items():
        for child in p.get("smaller_patterns", []):
            out[pid].append(child)
            in_deg[child] += 1

    q = deque(pid for pid, deg in in_deg.items() if deg == 0)
    seen = 0
    while q:
        cur = q.popleft()
        seen += 1
        for nxt in out[cur]:
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                q.append(nxt)
    if seen != len(by_id):
        errors.append("pattern graph contains at least one cycle")

    return errors


def build_sequence(data: dict) -> list[dict]:
    """Build a topologically sorted sequence of pattern rows."""
    errors = validate_pattern_language(data)
    if errors:
        raise ValueError("invalid pattern language: " + "; ".join(errors))

    by_id = {p["id"]: p for p in data["patterns"]}
    out = defaultdict(list)
    in_deg = {pid: 0 for pid in by_id}
    for p in data["patterns"]:
        for child in p["smaller_patterns"]:
            out[p["id"]].append(child)
            in_deg[child] += 1

    ready = [pid for pid, deg in in_deg.items() if deg == 0]
    ready.sort(key=lambda pid: (by_id[pid].get("order_hint", 0), pid))

    ordered: list[str] = []
    while ready:
        cur = ready.pop(0)
        ordered.append(cur)
        for nxt in out[cur]:
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=lambda pid: (by_id[pid].get("order_hint", 0), pid))

    return [
        {"index": i, "id": pid, "name": by_id[pid]["name"],
         "stage": by_id[pid]["stage"], "confidence_tier": by_id[pid]["confidence_tier"]}
        for i, pid in enumerate(ordered)
    ]


def orchestrate_record(
    record: dict[str, Any],
    sequence: list[dict[str, Any]],
    registry: dict[str, PatternHandler] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply pattern handlers in sequence to a single record."""
    out = copy.deepcopy(record)
    events: list[dict[str, Any]] = []
    reg = registry or {}

    refs = list(out.get("pattern_refs") or [])
    out["pattern_refs"] = refs

    for row in sequence:
        pid = row.get("id")
        handler = reg.get(pid)
        if handler is None:
            status = "missing_handler"
        else:
            applied = bool(handler(out, row))
            status = "applied" if applied else "noop"

        if pid not in refs:
            refs.append(pid)

        events.append({
            "pattern_id": pid,
            "stage": row.get("stage"),
            "status": status,
        })

    return out, events


def orchestrate_records(
    records: list[dict[str, Any]],
    sequence: list[dict[str, Any]],
    registry: dict[str, PatternHandler] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply orchestration to all records."""
    updated: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        out, events = orchestrate_record(rec, sequence, registry=registry)
        updated.append(out)
        for ev in events:
            ev["record_index"] = i
            all_events.append(ev)
    return updated, all_events
