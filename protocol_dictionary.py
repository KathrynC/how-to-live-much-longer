"""protocol_dictionary.py — Persistent catalog of intervention protocols.

Ported from rosetta-motion's MotionDiscovery class. Stores ProtocolRecord
entries with JSON persistence, querying by outcome/source/confidence,
deduplication, and summary statistics.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from protocol_record import ProtocolRecord, protocol_fingerprint


class ProtocolDictionary:
    """Persistent dictionary of discovered intervention protocols.

    Analogous to rosetta-motion's MotionDiscovery class.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.records: list[ProtocolRecord] = []
        self.meta: dict[str, Any] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.records = [ProtocolRecord.from_dict(r)
                           for r in data.get("records", [])]
            self.meta = data.get("meta", {})

    def __len__(self) -> int:
        return len(self.records)

    def add(self, record: ProtocolRecord) -> None:
        """Add a protocol record to the dictionary."""
        self.records.append(record)

    def save(self) -> None:
        """Persist to JSON."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "records": [r.to_dict() for r in self.records],
            "meta": self.meta,
        }
        self.path.write_text(json.dumps(payload, indent=2))

    def query(
        self,
        outcome_class: str | None = None,
        source: str | None = None,
        min_confidence: float | None = None,
    ) -> list[ProtocolRecord]:
        """Query records by filter criteria."""
        results = self.records
        if outcome_class is not None:
            results = [r for r in results if r.outcome_class == outcome_class]
        if source is not None:
            results = [r for r in results if r.source == source]
        if min_confidence is not None:
            results = [r for r in results
                       if r.confidence is not None and r.confidence >= min_confidence]
        return results

    def summary(self) -> dict[str, Any]:
        """Return summary statistics about the dictionary contents."""
        by_outcome = Counter(r.outcome_class for r in self.records
                            if r.outcome_class is not None)
        by_source = Counter(r.source for r in self.records)
        confidences = [r.confidence for r in self.records
                      if r.confidence is not None]
        return {
            "total": len(self.records),
            "by_outcome": dict(by_outcome),
            "by_source": dict(by_source),
            "mean_confidence": (sum(confidences) / len(confidences)
                               if confidences else None),
        }

    def deduplicate(self) -> int:
        """Remove duplicate protocols, keeping the highest-confidence version.

        Duplicates are identified by (intervention_fingerprint, patient_fingerprint).
        Returns the number of records removed.
        """
        seen: dict[tuple[str, str], int] = {}
        to_remove: set[int] = set()
        for i, rec in enumerate(self.records):
            iv_fp = protocol_fingerprint(rec.intervention)
            pt_fp = protocol_fingerprint(rec.patient)
            key = (iv_fp, pt_fp)
            if key in seen:
                prev_idx = seen[key]
                prev_conf = self.records[prev_idx].confidence or 0.0
                curr_conf = rec.confidence or 0.0
                if curr_conf > prev_conf:
                    to_remove.add(prev_idx)
                    seen[key] = i
                else:
                    to_remove.add(i)
            else:
                seen[key] = i

        original_len = len(self.records)
        self.records = [r for i, r in enumerate(self.records)
                       if i not in to_remove]
        return original_len - len(self.records)
