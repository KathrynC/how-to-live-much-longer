"""protocol_review.py — Review governance for uncertain protocol classifications.

Ported from rosetta-motion's review_queue_append() and _flag_review_needed().
Low-confidence and paradoxical protocols are routed to a JSONL review queue.
Reviewers can accept, override, or flag with provenance.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from protocol_record import ProtocolRecord

# Confidence threshold below which protocols are routed to review
REVIEW_CONFIDENCE_THRESHOLD = 0.6

# Outcome classes that always require review regardless of confidence
ALWAYS_REVIEW_CLASSES = {"paradoxical"}


def needs_review(
    confidence: float | None = None,
    outcome_class: str | None = None,
) -> bool:
    """Determine if a protocol needs human review."""
    if outcome_class in ALWAYS_REVIEW_CLASSES:
        return True
    if confidence is not None and confidence < REVIEW_CONFIDENCE_THRESHOLD:
        return True
    return False


def append_to_review_queue(
    path: Path,
    record: ProtocolRecord,
    reason: str = "",
) -> None:
    """Append a protocol to the review queue (JSONL)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "record": record.to_dict(),
        "reason": reason,
        "status": "pending",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": None,
        "reviewer": None,
        "notes": None,
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_review_queue(path: Path) -> list[dict[str, Any]]:
    """Load review queue entries from JSONL file."""
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().strip().split("\n"):
        if line.strip():
            entries.append(json.loads(line))
    return entries


def resolve_review(
    path: Path,
    index: int,
    decision: str,
    reviewer: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Resolve a review queue entry by index.

    Args:
        decision: "accept", "override", or "flag_expert"
        reviewer: who made the decision
        notes: free-text rationale
    """
    entries = load_review_queue(path)
    if index < 0 or index >= len(entries):
        raise IndexError(f"review index {index} out of range (queue has {len(entries)} entries)")

    entries[index]["status"] = "resolved"
    entries[index]["decision"] = decision
    entries[index]["reviewer"] = reviewer
    entries[index]["notes"] = notes
    entries[index]["resolved_at"] = datetime.now(timezone.utc).isoformat()

    # Rewrite the entire queue
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return entries[index]
