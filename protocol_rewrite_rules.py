"""protocol_rewrite_rules.py — Declarative rule engine for protocol normalization.

Ported from rosetta-motion's rewrite_rules.py. Three layers:
  - normalization: grid snapping, field defaults, dose clamping
  - clinical: safety flags, dose warnings, contradiction detection
  - semantic: low-confidence routing, review flags

Rules are priority-ordered and traced for audit (JSONL output).
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

VALID_LAYERS = {"normalization", "clinical", "semantic"}


def validate_rules(data: dict) -> list[str]:
    """Validate rewrite rule payload schema. Returns list of errors."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["root payload must be an object"]
    version = data.get("version")
    if not isinstance(version, str) or not re.match(r"^v\d+$", str(version)):
        errors.append("'version' must match ^v\\d+$")
    rules = data.get("rules")
    if not isinstance(rules, list):
        errors.append("'rules' must be a list")
        return errors

    seen_ids: set[str] = set()
    for i, rule in enumerate(rules):
        if not isinstance(rule, dict):
            errors.append(f"rules[{i}] is not an object")
            continue
        rid = rule.get("id", "")
        if rid in seen_ids:
            errors.append(f"duplicate rule id: {rid}")
        seen_ids.add(rid)
        if rule.get("layer") not in VALID_LAYERS:
            errors.append(f"rule {rid} has invalid layer: {rule.get('layer')}")
        if not isinstance(rule.get("enabled"), bool):
            errors.append(f"rule {rid} enabled must be boolean")
        if not isinstance(rule.get("priority"), int):
            errors.append(f"rule {rid} priority must be integer")
        if not isinstance(rule.get("match"), dict):
            errors.append(f"rule {rid} match must be an object")
        if not isinstance(rule.get("rewrite"), dict):
            errors.append(f"rule {rid} rewrite must be an object")
    return errors


def matches_condition(record: dict, field: str, cond: dict) -> bool:
    """Evaluate a single match condition against a record field."""
    value = record.get(field)
    if "exists" in cond:
        exists = value is not None and value != "" and value != []
        if bool(cond["exists"]) != exists:
            return False
    if "equals" in cond:
        if value != cond["equals"]:
            return False
    if "in" in cond:
        if value not in cond["in"]:
            return False
    if "gt" in cond:
        if not isinstance(value, (int, float)) or not (value > cond["gt"]):
            return False
    if "gte" in cond:
        if not isinstance(value, (int, float)) or not (value >= cond["gte"]):
            return False
    if "lt" in cond:
        if not isinstance(value, (int, float)) or not (value < cond["lt"]):
            return False
    if "lte" in cond:
        if not isinstance(value, (int, float)) or not (value <= cond["lte"]):
            return False
    return True


def _rule_matches(record: dict, rule: dict) -> bool:
    """Check if all match conditions of a rule are satisfied."""
    match = rule.get("match", {})
    return all(matches_condition(record, f, cond) for f, cond in match.items())


def _apply_ops(record: dict, rewrite: dict) -> None:
    """Apply rewrite operations (set, append_unique, remove_values)."""
    for field, value in (rewrite.get("set") or {}).items():
        record[field] = value
    for field, values in (rewrite.get("append_unique") or {}).items():
        current = record.get(field)
        if current is None:
            current = []
        if not isinstance(current, list):
            current = [current]
        incoming = values if isinstance(values, list) else [values]
        for item in incoming:
            if item not in current:
                current.append(item)
        record[field] = current
    for field, values in (rewrite.get("remove_values") or {}).items():
        current = record.get(field)
        if not isinstance(current, list):
            continue
        to_remove = set(values if isinstance(values, list) else [values])
        record[field] = [item for item in current if item not in to_remove]


def apply_rules(
    record: dict,
    rules_data: dict,
) -> tuple[dict, list[dict]]:
    """Apply ordered rewrite rules to a record. Returns (updated_record, trace)."""
    errors = validate_rules(rules_data)
    if errors:
        raise ValueError("invalid rewrite rules: " + "; ".join(errors))

    out = copy.deepcopy(record)
    trace: list[dict] = []
    enabled = [r for r in rules_data.get("rules", []) if r.get("enabled")]
    enabled.sort(key=lambda r: (int(r.get("priority", 0)), str(r.get("id"))))

    for rule in enabled:
        if not _rule_matches(out, rule):
            continue
        _apply_ops(out, rule.get("rewrite", {}))
        trace.append({
            "rule_id": rule.get("id"),
            "layer": rule.get("layer"),
            "priority": rule.get("priority"),
        })

    return out, trace


def apply_rules_to_records(
    records: list[dict],
    rules_data: dict,
) -> tuple[list[dict], list[dict]]:
    """Apply rewrite rules to a list of records. Returns (updated_records, all_trace_events)."""
    updated: list[dict] = []
    events: list[dict] = []
    for idx, rec in enumerate(records):
        out, trace = apply_rules(rec, rules_data)
        updated.append(out)
        for ev in trace:
            ev_copy = dict(ev)
            ev_copy["record_index"] = idx
            events.append(ev_copy)
    return updated, events
