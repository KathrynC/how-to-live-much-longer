"""protocol_record.py — Standardized schema for protocol dictionary entries.

Ported from rosetta-motion's DiscoveryRecord pattern. Each record carries
an intervention vector, patient context, simulation results, analytics,
enrichment fields, classification, and provenance metadata.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any

VALID_OUTCOME_CLASSES = frozenset({
    "thriving",      # final ATP > 0.8 AND het < 0.5
    "stable",        # final ATP > 0.5 AND het < 0.7
    "declining",     # final ATP > 0.2, het > 0.5
    "collapsed",     # final ATP < 0.2
    "paradoxical",   # worse than no-treatment on both ATP and het
})


def protocol_fingerprint(intervention: dict[str, Any]) -> str:
    """Return a short stable hash for an intervention dict.

    Analogous to rosetta-motion's _hash_weights(). Produces a 10-char
    hex string from the sorted (key, value) pairs.
    """
    items = sorted((str(k), float(v)) for k, v in intervention.items()
                   if v is not None)
    payload = json.dumps(items, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:10]


@dataclass
class ProtocolRecord:
    """A single protocol entry in the protocol dictionary.

    Mirrors rosetta-motion's DiscoveryRecord with domain-specific fields:
    - intervention: 6D intervention vector (rapamycin, NAD, senolytic, etc.)
    - patient: 6D patient context (age, heteroplasmy, NAD, etc.)
    - source: which tool generated this (dark_matter, ea_optimizer, character_seed, etc.)
    - method: specific generation method (random_sample, cma_es, llm_offer, etc.)
    - outcome_class: thriving/stable/declining/collapsed/paradoxical
    - confidence: 0.0-1.0, agreement across labeling methods
    - analytics: 4-pillar health analytics dict
    - enrichment: computed fields (complexity, clinical_signature, prototype, etc.)
    - simulation: raw simulation result summary (final states, trajectory shape)
    - classifications: per-method classification results for audit
    - meta: arbitrary provenance metadata
    """
    intervention: dict[str, Any]
    patient: dict[str, Any]
    source: str = "unknown"
    method: str = ""
    outcome_class: str | None = None
    confidence: float | None = None
    analytics: dict[str, Any] = field(default_factory=dict)
    enrichment: dict[str, Any] = field(default_factory=dict)
    simulation: dict[str, Any] = field(default_factory=dict)
    classifications: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProtocolRecord:
        """Deserialize from a plain dict."""
        return cls(
            intervention=d.get("intervention", {}),
            patient=d.get("patient", {}),
            source=d.get("source", "unknown"),
            method=d.get("method", ""),
            outcome_class=d.get("outcome_class"),
            confidence=d.get("confidence"),
            analytics=d.get("analytics", {}),
            enrichment=d.get("enrichment", {}),
            simulation=d.get("simulation", {}),
            classifications=d.get("classifications", {}),
            meta=d.get("meta", {}),
        )
