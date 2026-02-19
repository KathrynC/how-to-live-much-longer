# Protocol Dictionary Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port rosetta-motion's most valuable patterns — protocol dictionary, multi-labeler classification, rewrite rules, pattern language pipeline, and review governance — to create a unified protocol curation system for the mitochondrial aging simulator.

**Architecture:** A `ProtocolDictionary` (analogous to rosetta-motion's `MotionDiscovery`) serves as the central catalog of intervention protocols with enrichment fields, multi-method classification, and provenance tracking. A pattern language DAG defines the pipeline stages (ingest → analytics → robustness → classify → review → report), and a rewrite rules engine normalizes protocols before classification. Low-confidence protocols are routed to a review queue with audit trail.

**Tech Stack:** Python 3.11, numpy 1.26.4, matplotlib (Agg), JSON artifacts. No new dependencies.

---

## Background: Why These Patterns Matter

The rosetta-motion project developed a sophisticated curation pipeline for classifying robot gaits. The mito project has ~30 analysis scripts that each discover protocols independently (dark_matter.py, reachable_set.py, competing_evaluators.py, EA optimizers, character seeds, etc.) but there is **no unified catalog** of discovered protocols. Results are scattered across JSON artifacts with incompatible schemas.

### What We're Porting (ranked by ROI)

| Rosetta-Motion Innovation | Mito Equivalent | Why It Matters |
|---|---|---|
| `MotionDiscovery` class | `ProtocolDictionary` | Central catalog unifying all discovered protocols |
| `DiscoveryRecord` dataclass | `ProtocolRecord` | Standardized schema for protocol entries |
| Controller simplicity + sensory signature | Protocol complexity + clinical signature | Computed enrichment fields per protocol |
| Rule-based + metric-fit + LLM labeling | Multi-labeler classification pipeline | Multiple methods agree/disagree on protocol quality |
| Rewrite rules engine | Protocol normalization rules | Grid snapping, dose validation, clinical flags |
| Pattern language DAG + orchestrator | Analysis pipeline stages | Auditable, ordered pipeline execution |
| Review governance + override curation | Clinical review queue | Low-confidence routing with audit trail |
| `run_pipeline.py` orchestrator | `run_protocol_pipeline.py` | End-to-end pipeline runner |

### What We're NOT Porting

- Beer-framework analytics (PyBullet-specific; we already have 4-pillar health analytics)
- Word-gait loss functions (too domain-specific; our `competing_evaluators.py` fills this role)
- Thompson motif indexing (nice-to-have, defer to future work)
- 3D visualization server (we use Matplotlib Agg; viz-tools handles interactive views)

---

## Concept Mapping

| Rosetta-Motion | Mito | Notes |
|---|---|---|
| Motion gait entry | Intervention protocol | 6D intervention vector + 6D patient context |
| Observed label (crawl, sprint) | Clinical outcome class | thriving/stable/declining/collapsed/paradoxical |
| Confidence (0–1) | Classification confidence | Agreement across labeling methods |
| Beer 4-pillar analytics | Health 4-pillar analytics | Already exists in `analytics.py` |
| Controller simplicity (weight L1/L2) | Protocol complexity (total dose, parameter count) | How "heavy" is the intervention? |
| Sensory signature | Clinical signature | Energy trajectory shape, damage dynamics |
| Robustness (gravity/friction stress) | Robustness (biological stress) | Already exists via cramer-toolkit |
| Prototype descriptor | Protocol archetype | Nearest champion protocol |
| Metric-fit score | Analytics-fit score | Mahalanobis distance to class prototype |

---

## Task 1: Protocol Record Schema

**Files:**
- Create: `protocol_record.py`
- Test: `tests/test_protocol_record.py`

**Context:** This is the foundational data structure — a standardized schema for protocol entries in the dictionary. Analogous to rosetta-motion's `DiscoveryRecord` dataclass. Every protocol record carries its intervention vector, patient context, simulation results, analytics, enrichment fields, classification, and provenance.

**Step 1: Write the failing test**

```python
"""tests/test_protocol_record.py"""
import pytest


class TestProtocolRecord:
    """Test the ProtocolRecord dataclass and factory functions."""

    def test_create_minimal_record(self):
        """A record can be created with just intervention and patient dicts."""
        from protocol_record import ProtocolRecord

        rec = ProtocolRecord(
            intervention={"rapamycin_dose": 0.5, "nad_supplement": 0.75},
            patient={"baseline_age": 70.0, "baseline_heteroplasmy": 0.30},
        )
        assert rec.intervention["rapamycin_dose"] == 0.5
        assert rec.patient["baseline_age"] == 70.0
        assert rec.source == "unknown"
        assert rec.confidence is None
        assert rec.outcome_class is None

    def test_create_full_record(self):
        """A record can be created with all fields populated."""
        from protocol_record import ProtocolRecord

        rec = ProtocolRecord(
            intervention={"rapamycin_dose": 0.5},
            patient={"baseline_age": 70.0},
            source="dark_matter",
            method="random_sample",
            outcome_class="thriving",
            confidence=0.95,
            analytics={"energy": {"final_atp": 0.85}},
            enrichment={"complexity": {"total_dose": 1.25}},
            meta={"seed": 42, "sim_index": 7},
        )
        assert rec.source == "dark_matter"
        assert rec.outcome_class == "thriving"
        assert rec.confidence == 0.95
        assert rec.analytics["energy"]["final_atp"] == 0.85

    def test_record_to_dict(self):
        """A record can be serialized to a plain dict."""
        from protocol_record import ProtocolRecord

        rec = ProtocolRecord(
            intervention={"rapamycin_dose": 0.5},
            patient={"baseline_age": 70.0},
            source="test",
        )
        d = rec.to_dict()
        assert isinstance(d, dict)
        assert d["intervention"]["rapamycin_dose"] == 0.5
        assert d["source"] == "test"

    def test_record_from_dict(self):
        """A record can be deserialized from a plain dict."""
        from protocol_record import ProtocolRecord

        d = {
            "intervention": {"rapamycin_dose": 0.5},
            "patient": {"baseline_age": 70.0},
            "source": "test",
            "outcome_class": "stable",
            "confidence": 0.8,
        }
        rec = ProtocolRecord.from_dict(d)
        assert rec.intervention["rapamycin_dose"] == 0.5
        assert rec.outcome_class == "stable"
        assert rec.confidence == 0.8

    def test_record_roundtrip(self):
        """to_dict → from_dict preserves all fields."""
        from protocol_record import ProtocolRecord

        original = ProtocolRecord(
            intervention={"rapamycin_dose": 0.5, "nad_supplement": 0.75},
            patient={"baseline_age": 70.0, "baseline_heteroplasmy": 0.30},
            source="ea_optimizer",
            method="cma_es",
            outcome_class="thriving",
            confidence=0.92,
            analytics={"energy": {"final_atp": 0.88}},
            enrichment={"complexity": {"total_dose": 1.25}},
            meta={"budget": 500},
        )
        restored = ProtocolRecord.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_valid_outcome_classes(self):
        """VALID_OUTCOME_CLASSES contains the expected set."""
        from protocol_record import VALID_OUTCOME_CLASSES

        assert "thriving" in VALID_OUTCOME_CLASSES
        assert "stable" in VALID_OUTCOME_CLASSES
        assert "declining" in VALID_OUTCOME_CLASSES
        assert "collapsed" in VALID_OUTCOME_CLASSES
        assert "paradoxical" in VALID_OUTCOME_CLASSES

    def test_protocol_fingerprint(self):
        """Two records with same intervention produce same fingerprint."""
        from protocol_record import protocol_fingerprint

        iv = {"rapamycin_dose": 0.5, "nad_supplement": 0.75,
              "senolytic_dose": 0.0, "yamanaka_intensity": 0.0,
              "transplant_rate": 0.0, "exercise_level": 0.5}
        fp1 = protocol_fingerprint(iv)
        fp2 = protocol_fingerprint(iv)
        assert fp1 == fp2
        assert isinstance(fp1, str)
        assert len(fp1) == 10  # short sha1 hex

    def test_different_interventions_different_fingerprints(self):
        """Different interventions produce different fingerprints."""
        from protocol_record import protocol_fingerprint

        fp1 = protocol_fingerprint({"rapamycin_dose": 0.5})
        fp2 = protocol_fingerprint({"rapamycin_dose": 0.75})
        assert fp1 != fp2
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && conda run -n mito-aging pytest tests/test_protocol_record.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'protocol_record'"

**Step 3: Write minimal implementation**

```python
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
    - confidence: 0.0–1.0, agreement across labeling methods
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
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && conda run -n mito-aging pytest tests/test_protocol_record.py -v`
Expected: 9 PASSED

**Step 5: Commit**

```bash
git add protocol_record.py tests/test_protocol_record.py
git commit -m "feat: add ProtocolRecord schema (rosetta-motion DiscoveryRecord port)"
```

---

## Task 2: Protocol Dictionary

**Files:**
- Create: `protocol_dictionary.py`
- Test: `tests/test_protocol_dictionary.py`

**Context:** The persistent dictionary class that stores, queries, and serializes protocol records. Analogous to rosetta-motion's `MotionDiscovery` class. Supports adding records, querying by outcome class or source, deduplication via fingerprinting, and JSON persistence.

**Step 1: Write the failing test**

```python
"""tests/test_protocol_dictionary.py"""
import json
import pytest
from pathlib import Path


@pytest.fixture
def tmp_dict_path(tmp_path):
    return tmp_path / "test_protocols.json"


@pytest.fixture
def sample_record():
    from protocol_record import ProtocolRecord
    return ProtocolRecord(
        intervention={"rapamycin_dose": 0.5, "nad_supplement": 0.75,
                       "senolytic_dose": 0.5, "yamanaka_intensity": 0.0,
                       "transplant_rate": 0.0, "exercise_level": 0.5},
        patient={"baseline_age": 70.0, "baseline_heteroplasmy": 0.30,
                 "baseline_nad_level": 0.6, "genetic_vulnerability": 1.0,
                 "metabolic_demand": 1.0, "inflammation_level": 0.25},
        source="test",
        outcome_class="thriving",
        confidence=0.9,
    )


class TestProtocolDictionary:
    """Test the ProtocolDictionary persistent catalog."""

    def test_create_empty(self, tmp_dict_path):
        from protocol_dictionary import ProtocolDictionary
        pd = ProtocolDictionary(tmp_dict_path)
        assert len(pd) == 0

    def test_add_and_len(self, tmp_dict_path, sample_record):
        from protocol_dictionary import ProtocolDictionary
        pd = ProtocolDictionary(tmp_dict_path)
        pd.add(sample_record)
        assert len(pd) == 1

    def test_save_and_load(self, tmp_dict_path, sample_record):
        from protocol_dictionary import ProtocolDictionary
        pd = ProtocolDictionary(tmp_dict_path)
        pd.add(sample_record)
        pd.save()
        assert tmp_dict_path.exists()

        pd2 = ProtocolDictionary(tmp_dict_path)
        assert len(pd2) == 1
        assert pd2.records[0].outcome_class == "thriving"

    def test_query_by_outcome(self, tmp_dict_path):
        from protocol_dictionary import ProtocolDictionary
        from protocol_record import ProtocolRecord
        pd = ProtocolDictionary(tmp_dict_path)
        pd.add(ProtocolRecord(
            intervention={"rapamycin_dose": 0.5}, patient={"baseline_age": 70.0},
            outcome_class="thriving", source="test"))
        pd.add(ProtocolRecord(
            intervention={"rapamycin_dose": 0.1}, patient={"baseline_age": 70.0},
            outcome_class="collapsed", source="test"))
        pd.add(ProtocolRecord(
            intervention={"rapamycin_dose": 0.75}, patient={"baseline_age": 70.0},
            outcome_class="thriving", source="test"))

        thriving = pd.query(outcome_class="thriving")
        assert len(thriving) == 2
        collapsed = pd.query(outcome_class="collapsed")
        assert len(collapsed) == 1

    def test_query_by_source(self, tmp_dict_path):
        from protocol_dictionary import ProtocolDictionary
        from protocol_record import ProtocolRecord
        pd = ProtocolDictionary(tmp_dict_path)
        pd.add(ProtocolRecord(
            intervention={"rapamycin_dose": 0.5}, patient={"baseline_age": 70.0},
            source="dark_matter"))
        pd.add(ProtocolRecord(
            intervention={"rapamycin_dose": 0.75}, patient={"baseline_age": 70.0},
            source="ea_optimizer"))

        dm = pd.query(source="dark_matter")
        assert len(dm) == 1
        assert dm[0].source == "dark_matter"

    def test_query_by_min_confidence(self, tmp_dict_path):
        from protocol_dictionary import ProtocolDictionary
        from protocol_record import ProtocolRecord
        pd = ProtocolDictionary(tmp_dict_path)
        pd.add(ProtocolRecord(
            intervention={"rapamycin_dose": 0.5}, patient={"baseline_age": 70.0},
            confidence=0.9, source="test"))
        pd.add(ProtocolRecord(
            intervention={"rapamycin_dose": 0.1}, patient={"baseline_age": 70.0},
            confidence=0.3, source="test"))

        high = pd.query(min_confidence=0.5)
        assert len(high) == 1

    def test_summary_stats(self, tmp_dict_path):
        from protocol_dictionary import ProtocolDictionary
        from protocol_record import ProtocolRecord
        pd = ProtocolDictionary(tmp_dict_path)
        for oc in ["thriving", "thriving", "stable", "collapsed"]:
            pd.add(ProtocolRecord(
                intervention={"rapamycin_dose": 0.5}, patient={"baseline_age": 70.0},
                outcome_class=oc, source="test"))

        stats = pd.summary()
        assert stats["total"] == 4
        assert stats["by_outcome"]["thriving"] == 2
        assert stats["by_outcome"]["stable"] == 1
        assert stats["by_outcome"]["collapsed"] == 1

    def test_deduplicate(self, tmp_dict_path):
        from protocol_dictionary import ProtocolDictionary
        from protocol_record import ProtocolRecord
        pd = ProtocolDictionary(tmp_dict_path)
        iv = {"rapamycin_dose": 0.5, "nad_supplement": 0.75}
        pt = {"baseline_age": 70.0}
        pd.add(ProtocolRecord(intervention=iv, patient=pt, source="a", confidence=0.8))
        pd.add(ProtocolRecord(intervention=iv, patient=pt, source="b", confidence=0.9))
        pd.add(ProtocolRecord(intervention={"rapamycin_dose": 0.1}, patient=pt, source="c"))

        removed = pd.deduplicate()
        assert removed == 1  # kept higher confidence duplicate
        assert len(pd) == 2
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && conda run -n mito-aging pytest tests/test_protocol_dictionary.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'protocol_dictionary'"

**Step 3: Write minimal implementation**

```python
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
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && conda run -n mito-aging pytest tests/test_protocol_dictionary.py -v`
Expected: 8 PASSED

**Step 5: Commit**

```bash
git add protocol_dictionary.py tests/test_protocol_dictionary.py
git commit -m "feat: add ProtocolDictionary persistent catalog (rosetta-motion MotionDiscovery port)"
```

---

## Task 3: Protocol Enrichment

**Files:**
- Create: `protocol_enrichment.py`
- Test: `tests/test_protocol_enrichment.py`

**Context:** Computed enrichment fields added to each protocol record, analogous to rosetta-motion's `controller_simplicity()`, `sensory_signature()`, `prototype_descriptor()`, and `prototype_strength()`. For the mito domain, we compute: protocol complexity (dose burden), clinical signature (trajectory shape), and prototype grouping (nearest champion protocol).

**Step 1: Write the failing test**

```python
"""tests/test_protocol_enrichment.py"""
import pytest


class TestProtocolComplexity:
    """Test protocol complexity enrichment (analogous to controller_simplicity)."""

    def test_total_dose(self):
        from protocol_enrichment import protocol_complexity
        iv = {"rapamycin_dose": 0.5, "nad_supplement": 0.75,
              "senolytic_dose": 0.5, "yamanaka_intensity": 0.0,
              "transplant_rate": 0.0, "exercise_level": 0.5}
        c = protocol_complexity(iv)
        assert abs(c["total_dose"] - 2.25) < 1e-6

    def test_active_count(self):
        from protocol_enrichment import protocol_complexity
        iv = {"rapamycin_dose": 0.5, "nad_supplement": 0.75,
              "senolytic_dose": 0.0, "yamanaka_intensity": 0.0,
              "transplant_rate": 0.0, "exercise_level": 0.5}
        c = protocol_complexity(iv)
        assert c["active_count"] == 3  # rapamycin, NAD, exercise

    def test_empty_intervention(self):
        from protocol_enrichment import protocol_complexity
        c = protocol_complexity({})
        assert c["total_dose"] == 0.0
        assert c["active_count"] == 0

    def test_max_dose(self):
        from protocol_enrichment import protocol_complexity
        iv = {"rapamycin_dose": 0.5, "nad_supplement": 1.0, "exercise_level": 0.25}
        c = protocol_complexity(iv)
        assert c["max_single_dose"] == 1.0


class TestClinicalSignature:
    """Test clinical signature extraction (analogous to sensory_signature)."""

    def test_from_analytics(self):
        from protocol_enrichment import clinical_signature
        analytics = {
            "energy": {"final_atp": 0.85, "mean_atp": 0.78, "atp_slope": -0.002, "min_atp": 0.65},
            "damage": {"final_het": 0.25, "time_to_cliff": 999, "het_acceleration": -0.001},
            "intervention": {"benefit_cost_ratio": 3.5},
        }
        sig = clinical_signature(analytics)
        assert sig["final_atp"] == 0.85
        assert sig["final_het"] == 0.25
        assert sig["energy_trend"] == "declining"  # negative slope
        assert sig["cliff_risk"] == "none"  # time_to_cliff > 30

    def test_improving_trend(self):
        from protocol_enrichment import clinical_signature
        analytics = {
            "energy": {"final_atp": 0.85, "mean_atp": 0.78, "atp_slope": 0.005, "min_atp": 0.65},
            "damage": {"final_het": 0.25, "time_to_cliff": 999, "het_acceleration": -0.001},
            "intervention": {"benefit_cost_ratio": 3.5},
        }
        sig = clinical_signature(analytics)
        assert sig["energy_trend"] == "improving"

    def test_high_cliff_risk(self):
        from protocol_enrichment import clinical_signature
        analytics = {
            "energy": {"final_atp": 0.4, "mean_atp": 0.5, "atp_slope": -0.01, "min_atp": 0.1},
            "damage": {"final_het": 0.68, "time_to_cliff": 5.0, "het_acceleration": 0.01},
            "intervention": {"benefit_cost_ratio": 0.5},
        }
        sig = clinical_signature(analytics)
        assert sig["cliff_risk"] == "imminent"  # time_to_cliff < 10


class TestPrototypeGrouping:
    """Test prototype grouping (analogous to prototype_descriptor)."""

    def test_assigns_group(self):
        from protocol_enrichment import prototype_group
        iv = {"rapamycin_dose": 0.5, "nad_supplement": 0.75, "exercise_level": 0.5,
              "senolytic_dose": 0.0, "yamanaka_intensity": 0.0, "transplant_rate": 0.0}
        pg = prototype_group(iv)
        assert pg["archetype"] == "cocktail"  # 3+ active interventions, no transplant/yamanaka
        assert "fingerprint" in pg

    def test_transplant_focused(self):
        from protocol_enrichment import prototype_group
        iv = {"rapamycin_dose": 0.5, "nad_supplement": 0.5, "transplant_rate": 0.75,
              "senolytic_dose": 0.0, "yamanaka_intensity": 0.0, "exercise_level": 0.0}
        pg = prototype_group(iv)
        assert pg["archetype"] == "transplant_focused"

    def test_no_treatment(self):
        from protocol_enrichment import prototype_group
        iv = {"rapamycin_dose": 0.0, "nad_supplement": 0.0, "senolytic_dose": 0.0,
              "yamanaka_intensity": 0.0, "transplant_rate": 0.0, "exercise_level": 0.0}
        pg = prototype_group(iv)
        assert pg["archetype"] == "no_treatment"


class TestEnrichRecord:
    """Test the top-level enrich_record function."""

    def test_enriches_all_fields(self):
        from protocol_enrichment import enrich_record
        from protocol_record import ProtocolRecord

        rec = ProtocolRecord(
            intervention={"rapamycin_dose": 0.5, "nad_supplement": 0.75,
                           "senolytic_dose": 0.5, "yamanaka_intensity": 0.0,
                           "transplant_rate": 0.0, "exercise_level": 0.5},
            patient={"baseline_age": 70.0, "baseline_heteroplasmy": 0.30},
            analytics={
                "energy": {"final_atp": 0.85, "mean_atp": 0.78, "atp_slope": -0.002, "min_atp": 0.65},
                "damage": {"final_het": 0.25, "time_to_cliff": 999, "het_acceleration": -0.001},
                "intervention": {"benefit_cost_ratio": 3.5},
            },
            source="test",
        )
        enriched = enrich_record(rec)
        assert "complexity" in enriched.enrichment
        assert "clinical_signature" in enriched.enrichment
        assert "prototype" in enriched.enrichment
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && conda run -n mito-aging pytest tests/test_protocol_enrichment.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'protocol_enrichment'"

**Step 3: Write minimal implementation**

```python
"""protocol_enrichment.py — Computed enrichment fields for protocol records.

Ported from rosetta-motion's controller_simplicity(), sensory_signature(),
prototype_descriptor(), and prototype_strength(). Adapted for the
mitochondrial aging intervention domain.
"""
from __future__ import annotations

import copy
from typing import Any

from protocol_record import ProtocolRecord, protocol_fingerprint

# Interventions considered "active" above this threshold
ACTIVE_THRESHOLD = 0.05


def protocol_complexity(intervention: dict[str, Any]) -> dict[str, Any]:
    """Compute protocol complexity metrics.

    Analogous to rosetta-motion's controller_simplicity(). Measures how
    "heavy" an intervention protocol is in terms of dose burden.
    """
    doses = [float(v) for v in intervention.values() if v is not None]
    if not doses:
        return {"total_dose": 0.0, "active_count": 0, "max_single_dose": 0.0,
                "mean_active_dose": 0.0, "param_count": 0}

    active = [d for d in doses if abs(d) > ACTIVE_THRESHOLD]
    return {
        "total_dose": sum(doses),
        "active_count": len(active),
        "max_single_dose": max(doses) if doses else 0.0,
        "mean_active_dose": (sum(active) / len(active)) if active else 0.0,
        "param_count": len(doses),
    }


def clinical_signature(analytics: dict[str, Any]) -> dict[str, Any]:
    """Extract clinical signature from 4-pillar analytics.

    Analogous to rosetta-motion's sensory_signature(). Captures the
    trajectory shape and clinical risk profile.
    """
    energy = analytics.get("energy", {})
    damage = analytics.get("damage", {})
    intervention = analytics.get("intervention", {})

    atp_slope = energy.get("atp_slope", 0.0)
    if atp_slope is None:
        atp_slope = 0.0
    if atp_slope > 0.001:
        energy_trend = "improving"
    elif atp_slope < -0.001:
        energy_trend = "declining"
    else:
        energy_trend = "stable"

    ttc = damage.get("time_to_cliff", 999)
    if ttc is None:
        ttc = 999
    if ttc < 10:
        cliff_risk = "imminent"
    elif ttc < 20:
        cliff_risk = "moderate"
    else:
        cliff_risk = "none"

    return {
        "final_atp": energy.get("final_atp"),
        "final_het": damage.get("final_het"),
        "energy_trend": energy_trend,
        "cliff_risk": cliff_risk,
        "benefit_cost_ratio": intervention.get("benefit_cost_ratio"),
    }


def prototype_group(intervention: dict[str, Any]) -> dict[str, Any]:
    """Assign protocol to an archetype group.

    Analogous to rosetta-motion's prototype_descriptor(). Groups protocols
    by their dominant intervention mechanism.
    """
    fp = protocol_fingerprint(intervention)
    active = {k: v for k, v in intervention.items()
              if v is not None and float(v) > ACTIVE_THRESHOLD}

    if not active:
        return {"archetype": "no_treatment", "fingerprint": fp}

    has_transplant = float(intervention.get("transplant_rate", 0)) > ACTIVE_THRESHOLD
    has_yamanaka = float(intervention.get("yamanaka_intensity", 0)) > ACTIVE_THRESHOLD
    transplant_dominant = (float(intervention.get("transplant_rate", 0))
                          >= max(float(v) for v in active.values()) * 0.8)

    if has_yamanaka and has_transplant:
        archetype = "full_experimental"
    elif has_transplant and transplant_dominant:
        archetype = "transplant_focused"
    elif has_yamanaka:
        archetype = "reprogramming"
    elif len(active) >= 3:
        archetype = "cocktail"
    elif len(active) == 2:
        archetype = "dual_therapy"
    else:
        archetype = "monotherapy"

    return {"archetype": archetype, "fingerprint": fp,
            "dominant": max(active, key=lambda k: float(active[k]))}


def enrich_record(record: ProtocolRecord) -> ProtocolRecord:
    """Apply all enrichment fields to a protocol record.

    Returns a new record with enrichment dict populated.
    """
    enriched = copy.deepcopy(record)
    enriched.enrichment = {
        "complexity": protocol_complexity(record.intervention),
        "clinical_signature": clinical_signature(record.analytics),
        "prototype": prototype_group(record.intervention),
    }
    return enriched
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && conda run -n mito-aging pytest tests/test_protocol_enrichment.py -v`
Expected: 12 PASSED

**Step 5: Commit**

```bash
git add protocol_enrichment.py tests/test_protocol_enrichment.py
git commit -m "feat: add protocol enrichment (complexity, clinical signature, prototype grouping)"
```

---

## Task 4: Multi-Labeler Classification Pipeline

**Files:**
- Create: `protocol_classifier.py`
- Test: `tests/test_protocol_classifier.py`

**Context:** A multi-method classification system analogous to rosetta-motion's `discover_label()` pipeline. Three labeling methods — rule-based (existing threshold logic from dark_matter.py), analytics-fit (Mahalanobis distance to class prototypes), and LLM-based (Ollama clinical assessment) — are tried in sequence. The final outcome class and confidence are determined by method agreement.

**Step 1: Write the failing test**

```python
"""tests/test_protocol_classifier.py"""
import pytest


class TestRuleClassifier:
    """Test rule-based outcome classification."""

    def test_thriving(self):
        from protocol_classifier import rule_classify
        result = rule_classify(final_atp=0.85, final_het=0.25, baseline_atp=0.6, baseline_het=0.30)
        assert result["outcome_class"] == "thriving"
        assert result["confidence"] >= 0.8

    def test_stable(self):
        from protocol_classifier import rule_classify
        result = rule_classify(final_atp=0.55, final_het=0.55, baseline_atp=0.6, baseline_het=0.30)
        assert result["outcome_class"] == "stable"

    def test_declining(self):
        from protocol_classifier import rule_classify
        result = rule_classify(final_atp=0.35, final_het=0.65, baseline_atp=0.6, baseline_het=0.30)
        assert result["outcome_class"] == "declining"

    def test_collapsed(self):
        from protocol_classifier import rule_classify
        result = rule_classify(final_atp=0.10, final_het=0.90, baseline_atp=0.6, baseline_het=0.30)
        assert result["outcome_class"] == "collapsed"

    def test_paradoxical(self):
        from protocol_classifier import rule_classify
        result = rule_classify(final_atp=0.50, final_het=0.40, baseline_atp=0.55, baseline_het=0.30)
        assert result["outcome_class"] == "paradoxical"


class TestAnalyticsFitClassifier:
    """Test analytics-based prototype distance classification."""

    def test_computes_distances(self):
        from protocol_classifier import analytics_fit_classify, CLASS_PROTOTYPES
        analytics = {
            "energy": {"final_atp": 0.85, "mean_atp": 0.80},
            "damage": {"final_het": 0.20},
        }
        result = analytics_fit_classify(analytics)
        assert result["outcome_class"] in CLASS_PROTOTYPES
        assert 0.0 <= result["confidence"] <= 1.0
        assert "distances" in result

    def test_thriving_closest_to_thriving(self):
        from protocol_classifier import analytics_fit_classify
        analytics = {
            "energy": {"final_atp": 0.90, "mean_atp": 0.85},
            "damage": {"final_het": 0.15},
        }
        result = analytics_fit_classify(analytics)
        assert result["outcome_class"] == "thriving"


class TestMultiClassify:
    """Test the multi-method classification pipeline."""

    def test_pipeline_returns_all_methods(self):
        from protocol_classifier import multi_classify
        result = multi_classify(
            final_atp=0.85, final_het=0.25,
            baseline_atp=0.6, baseline_het=0.30,
            analytics={
                "energy": {"final_atp": 0.85, "mean_atp": 0.80},
                "damage": {"final_het": 0.25},
            },
            pipeline=["rule", "analytics_fit"],
        )
        assert "outcome_class" in result
        assert "confidence" in result
        assert "rule" in result["methods"]
        assert "analytics_fit" in result["methods"]

    def test_agreement_boosts_confidence(self):
        from protocol_classifier import multi_classify
        result = multi_classify(
            final_atp=0.85, final_het=0.20,
            baseline_atp=0.6, baseline_het=0.30,
            analytics={
                "energy": {"final_atp": 0.85, "mean_atp": 0.80},
                "damage": {"final_het": 0.20},
            },
            pipeline=["rule", "analytics_fit"],
        )
        # Both methods should agree on "thriving", boosting confidence
        assert result["outcome_class"] == "thriving"
        assert result["confidence"] >= 0.85

    def test_rule_only_pipeline(self):
        from protocol_classifier import multi_classify
        result = multi_classify(
            final_atp=0.85, final_het=0.20,
            baseline_atp=0.6, baseline_het=0.30,
            pipeline=["rule"],
        )
        assert result["outcome_class"] == "thriving"
        assert "rule" in result["methods"]
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && conda run -n mito-aging pytest tests/test_protocol_classifier.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'protocol_classifier'"

**Step 3: Write minimal implementation**

```python
"""protocol_classifier.py — Multi-method protocol classification pipeline.

Ported from rosetta-motion's discover_label() multi-labeler pattern.
Three classification methods:
  1. rule_classify: threshold-based (from dark_matter.py logic)
  2. analytics_fit_classify: prototype distance (from motion_discovery.py metric_fit_score)
  3. llm_classify: LLM clinical assessment (from motion_discovery.py llm_based_label)

Pipeline runs methods in sequence, aggregates results, and determines
final outcome_class + confidence from method agreement.
"""
from __future__ import annotations

import math
from typing import Any

# Class prototypes: mean values for key metrics per outcome class.
# Built from dark_matter.py classification thresholds and reachable_set observations.
CLASS_PROTOTYPES = {
    "thriving":    {"final_atp": 0.85, "mean_atp": 0.80, "final_het": 0.20},
    "stable":      {"final_atp": 0.60, "mean_atp": 0.55, "final_het": 0.50},
    "declining":   {"final_atp": 0.35, "mean_atp": 0.40, "final_het": 0.65},
    "collapsed":   {"final_atp": 0.10, "mean_atp": 0.15, "final_het": 0.85},
}

# Metric keys used for prototype distance computation
FIT_KEYS = ["final_atp", "mean_atp", "final_het"]

# Standard deviations for z-scoring (estimated from population simulations)
FIT_STD = {"final_atp": 0.25, "mean_atp": 0.20, "final_het": 0.20}


def rule_classify(
    final_atp: float,
    final_het: float,
    baseline_atp: float,
    baseline_het: float,
) -> dict[str, Any]:
    """Rule-based outcome classification using threshold logic.

    Matches dark_matter.py's classify_outcome() thresholds:
      thriving:    final ATP > 0.8 AND het < 0.5
      stable:      final ATP > 0.5 AND het < 0.7
      declining:   final ATP > 0.2, het > 0.5
      collapsed:   final ATP < 0.2
      paradoxical: worse than baseline on BOTH ATP and het
    """
    # Check paradoxical first (intervention made things worse)
    if final_atp < baseline_atp and final_het > baseline_het:
        return {"outcome_class": "paradoxical", "confidence": 0.9, "method": "rule"}

    if final_atp > 0.8 and final_het < 0.5:
        return {"outcome_class": "thriving", "confidence": 0.95, "method": "rule"}
    if final_atp > 0.5 and final_het < 0.7:
        return {"outcome_class": "stable", "confidence": 0.85, "method": "rule"}
    if final_atp < 0.2:
        return {"outcome_class": "collapsed", "confidence": 0.95, "method": "rule"}
    # Remaining: declining
    return {"outcome_class": "declining", "confidence": 0.80, "method": "rule"}


def analytics_fit_classify(
    analytics: dict[str, Any],
) -> dict[str, Any]:
    """Analytics-fit classification via prototype distance.

    Analogous to rosetta-motion's metric_fit_score(). Computes z-scored
    Euclidean distance to each class prototype, returns closest class
    with exponential similarity as confidence.
    """
    energy = analytics.get("energy", {})
    damage = analytics.get("damage", {})
    metrics = {
        "final_atp": energy.get("final_atp"),
        "mean_atp": energy.get("mean_atp"),
        "final_het": damage.get("final_het"),
    }

    distances: dict[str, float] = {}
    for cls, proto in CLASS_PROTOTYPES.items():
        keys = [k for k in FIT_KEYS if metrics.get(k) is not None and k in proto]
        if not keys:
            continue
        z2 = sum(
            ((float(metrics[k]) - proto[k]) / FIT_STD.get(k, 1.0)) ** 2
            for k in keys
        )
        distances[cls] = math.sqrt(z2 / max(1, len(keys)))

    if not distances:
        return {"outcome_class": None, "confidence": 0.0, "method": "analytics_fit",
                "distances": {}}

    best = min(distances, key=distances.get)
    confidence = math.exp(-distances[best])

    return {
        "outcome_class": best,
        "confidence": float(confidence),
        "method": "analytics_fit",
        "distances": {k: round(v, 4) for k, v in distances.items()},
    }


def multi_classify(
    final_atp: float = 0.0,
    final_het: float = 0.0,
    baseline_atp: float = 0.0,
    baseline_het: float = 0.0,
    analytics: dict[str, Any] | None = None,
    pipeline: list[str] | None = None,
) -> dict[str, Any]:
    """Run a multi-method classification pipeline and aggregate results.

    Analogous to rosetta-motion's discover_label(). Runs methods in order,
    collects per-method results, determines final class by majority vote
    (ties broken by highest confidence), and boosts confidence when methods agree.

    Args:
        pipeline: List of method names to run. Default: ["rule", "analytics_fit"]
    """
    if pipeline is None:
        pipeline = ["rule", "analytics_fit"]

    methods: dict[str, dict[str, Any]] = {}
    for step in pipeline:
        if step == "rule":
            methods["rule"] = rule_classify(final_atp, final_het,
                                            baseline_atp, baseline_het)
        elif step == "analytics_fit" and analytics is not None:
            methods["analytics_fit"] = analytics_fit_classify(analytics)
        # "llm" step would go here (requires Ollama, optional)

    if not methods:
        return {"outcome_class": None, "confidence": 0.0, "methods": {}}

    # Majority vote
    votes: dict[str, list[float]] = {}
    for name, result in methods.items():
        cls = result.get("outcome_class")
        conf = result.get("confidence", 0.0)
        if cls is not None:
            votes.setdefault(cls, []).append(conf)

    if not votes:
        return {"outcome_class": None, "confidence": 0.0, "methods": methods}

    # Pick class with most votes, break ties by max confidence
    best_class = max(votes, key=lambda c: (len(votes[c]), max(votes[c])))
    agreement = len(votes[best_class]) / len(methods)
    base_conf = max(votes[best_class])

    # Agreement bonus: if all methods agree, boost confidence
    if agreement == 1.0 and len(methods) > 1:
        confidence = min(1.0, base_conf + 0.05 * (len(methods) - 1))
    else:
        confidence = base_conf * agreement

    return {
        "outcome_class": best_class,
        "confidence": round(confidence, 4),
        "agreement": round(agreement, 4),
        "methods": methods,
    }
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && conda run -n mito-aging pytest tests/test_protocol_classifier.py -v`
Expected: 10 PASSED

**Step 5: Commit**

```bash
git add protocol_classifier.py tests/test_protocol_classifier.py
git commit -m "feat: add multi-labeler protocol classifier (rule + analytics-fit pipeline)"
```

---

## Task 5: Rewrite Rules Engine

**Files:**
- Create: `protocol_rewrite_rules.py`
- Create: `patterns/protocol_rewrite_rules.v1.json`
- Test: `tests/test_protocol_rewrite_rules.py`

**Context:** A declarative rule engine for transforming protocol records before classification. Ported directly from rosetta-motion's `rewrite_rules.py`. Three layers: normalization (grid snapping, field defaults), clinical (dose safety flags), and semantic (low-confidence routing). Rules are priority-ordered and traced for audit.

**Step 1: Write the failing test**

```python
"""tests/test_protocol_rewrite_rules.py"""
import json
import pytest
from pathlib import Path


class TestRuleValidation:
    """Test rewrite rule schema validation."""

    def test_valid_rules(self):
        from protocol_rewrite_rules import validate_rules
        data = {
            "version": "v1",
            "rules": [{
                "id": "snap_doses",
                "layer": "normalization",
                "enabled": True,
                "priority": 100,
                "description": "Snap intervention doses to grid",
                "match": {"total_dose": {"gt": 0}},
                "rewrite": {"set": {"_grid_snapped": True}},
            }],
        }
        errors = validate_rules(data)
        assert errors == []

    def test_invalid_missing_version(self):
        from protocol_rewrite_rules import validate_rules
        data = {"rules": []}
        errors = validate_rules(data)
        assert len(errors) > 0

    def test_invalid_layer(self):
        from protocol_rewrite_rules import validate_rules
        data = {
            "version": "v1",
            "rules": [{
                "id": "bad_layer",
                "layer": "invalid_layer",
                "enabled": True,
                "priority": 100,
                "description": "Bad",
                "match": {"x": {"gt": 0}},
                "rewrite": {"set": {"y": 1}},
            }],
        }
        errors = validate_rules(data)
        assert any("invalid layer" in e for e in errors)


class TestRuleMatching:
    """Test match condition evaluation."""

    def test_gt_match(self):
        from protocol_rewrite_rules import matches_condition
        assert matches_condition({"x": 5}, "x", {"gt": 3}) is True
        assert matches_condition({"x": 2}, "x", {"gt": 3}) is False

    def test_lt_match(self):
        from protocol_rewrite_rules import matches_condition
        assert matches_condition({"x": 2}, "x", {"lt": 3}) is True

    def test_equals_match(self):
        from protocol_rewrite_rules import matches_condition
        assert matches_condition({"x": "foo"}, "x", {"equals": "foo"}) is True
        assert matches_condition({"x": "bar"}, "x", {"equals": "foo"}) is False

    def test_exists_match(self):
        from protocol_rewrite_rules import matches_condition
        assert matches_condition({"x": 1}, "x", {"exists": True}) is True
        assert matches_condition({}, "x", {"exists": True}) is False
        assert matches_condition({}, "x", {"exists": False}) is True


class TestRuleApplication:
    """Test rule application to protocol records."""

    def test_set_operation(self):
        from protocol_rewrite_rules import apply_rules
        record = {"total_dose": 2.5, "outcome_class": None}
        rules_data = {
            "version": "v1",
            "rules": [{
                "id": "flag_high_dose",
                "layer": "clinical",
                "enabled": True,
                "priority": 100,
                "description": "Flag high dose",
                "match": {"total_dose": {"gt": 2.0}},
                "rewrite": {"set": {"_high_dose_warning": True}},
            }],
        }
        updated, trace = apply_rules(record, rules_data)
        assert updated["_high_dose_warning"] is True
        assert len(trace) == 1
        assert trace[0]["rule_id"] == "flag_high_dose"

    def test_append_unique_operation(self):
        from protocol_rewrite_rules import apply_rules
        record = {"confidence": 0.3, "flags": []}
        rules_data = {
            "version": "v1",
            "rules": [{
                "id": "low_conf_review",
                "layer": "semantic",
                "enabled": True,
                "priority": 200,
                "description": "Route low confidence to review",
                "match": {"confidence": {"lt": 0.6}},
                "rewrite": {"append_unique": {"flags": ["needs_review"]}},
            }],
        }
        updated, trace = apply_rules(record, rules_data)
        assert "needs_review" in updated["flags"]

    def test_disabled_rule_skipped(self):
        from protocol_rewrite_rules import apply_rules
        record = {"total_dose": 5.0}
        rules_data = {
            "version": "v1",
            "rules": [{
                "id": "disabled_rule",
                "layer": "normalization",
                "enabled": False,
                "priority": 100,
                "description": "Disabled",
                "match": {"total_dose": {"gt": 0}},
                "rewrite": {"set": {"should_not_appear": True}},
            }],
        }
        updated, trace = apply_rules(record, rules_data)
        assert "should_not_appear" not in updated
        assert len(trace) == 0

    def test_priority_ordering(self):
        from protocol_rewrite_rules import apply_rules
        record = {"x": 1, "log": []}
        rules_data = {
            "version": "v1",
            "rules": [
                {"id": "second", "layer": "normalization", "enabled": True,
                 "priority": 200, "description": "Second",
                 "match": {"x": {"gt": 0}},
                 "rewrite": {"append_unique": {"log": ["second"]}}},
                {"id": "first", "layer": "normalization", "enabled": True,
                 "priority": 100, "description": "First",
                 "match": {"x": {"gt": 0}},
                 "rewrite": {"append_unique": {"log": ["first"]}}},
            ],
        }
        updated, trace = apply_rules(record, rules_data)
        assert updated["log"] == ["first", "second"]
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && conda run -n mito-aging pytest tests/test_protocol_rewrite_rules.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'protocol_rewrite_rules'"

**Step 3: Write minimal implementation**

```python
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
```

Also create the default rules file:

```json
{
  "version": "v1",
  "description": "Protocol rewrite rules for normalization, clinical safety, and semantic routing.",
  "rules": [
    {
      "id": "flag_high_yamanaka",
      "layer": "clinical",
      "enabled": true,
      "priority": 100,
      "description": "Flag protocols with high Yamanaka intensity (3-5 MU energy cost)",
      "match": {"yamanaka_intensity": {"gt": 0.5}},
      "rewrite": {"append_unique": {"flags": ["high_yamanaka_energy_cost"]}}
    },
    {
      "id": "flag_transplant_dominant",
      "layer": "clinical",
      "enabled": true,
      "priority": 110,
      "description": "Flag transplant-dominant protocols for experimental review",
      "match": {"transplant_rate": {"gt": 0.75}},
      "rewrite": {"append_unique": {"flags": ["transplant_dominant"]}}
    },
    {
      "id": "route_low_confidence",
      "layer": "semantic",
      "enabled": true,
      "priority": 200,
      "description": "Route low-confidence classifications to review queue",
      "match": {"confidence": {"lt": 0.6}},
      "rewrite": {"append_unique": {"flags": ["needs_review"]}}
    },
    {
      "id": "flag_paradoxical",
      "layer": "semantic",
      "enabled": true,
      "priority": 210,
      "description": "Flag paradoxical outcomes for investigation",
      "match": {"outcome_class": {"equals": "paradoxical"}},
      "rewrite": {"append_unique": {"flags": ["paradoxical_investigate"]}}
    }
  ]
}
```

Save this to `patterns/protocol_rewrite_rules.v1.json`.

**Step 4: Run test to verify it passes**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && conda run -n mito-aging pytest tests/test_protocol_rewrite_rules.py -v`
Expected: 11 PASSED

**Step 5: Commit**

```bash
mkdir -p patterns
git add protocol_rewrite_rules.py patterns/protocol_rewrite_rules.v1.json tests/test_protocol_rewrite_rules.py
git commit -m "feat: add rewrite rules engine for protocol normalization (rosetta-motion port)"
```

---

## Task 6: Pattern Language Pipeline

**Files:**
- Create: `protocol_pattern_language.py`
- Create: `patterns/protocol_pattern_language.v1.json`
- Test: `tests/test_protocol_pattern_language.py`

**Context:** The pattern language defines the analysis pipeline as a validated DAG (Directed Acyclic Graph). Ported from rosetta-motion's `pattern_language.py` and `pattern_orchestrator.py`. Each pattern represents a pipeline stage with problem/solution documentation. The orchestrator executes registered handler functions in topological order.

**Step 1: Write the failing test**

```python
"""tests/test_protocol_pattern_language.py"""
import json
import pytest
from pathlib import Path


class TestPatternValidation:
    """Test pattern language schema validation."""

    def test_valid_pattern_language(self):
        from protocol_pattern_language import validate_pattern_language
        data = {
            "version": "v1",
            "patterns": [
                {"id": "ingest", "name": "Ingest", "stage": "ingest",
                 "problem": "P", "solution": "S", "confidence_tier": "core",
                 "larger_patterns": [], "smaller_patterns": ["analytics"],
                 "order_hint": 0},
                {"id": "analytics", "name": "Analytics", "stage": "analytics",
                 "problem": "P", "solution": "S", "confidence_tier": "core",
                 "larger_patterns": ["ingest"], "smaller_patterns": [],
                 "order_hint": 10},
            ],
        }
        errors = validate_pattern_language(data)
        assert errors == []

    def test_cycle_detection(self):
        from protocol_pattern_language import validate_pattern_language
        data = {
            "version": "v1",
            "patterns": [
                {"id": "a", "name": "A", "stage": "ingest",
                 "problem": "P", "solution": "S", "confidence_tier": "core",
                 "larger_patterns": ["b"], "smaller_patterns": ["b"],
                 "order_hint": 0},
                {"id": "b", "name": "B", "stage": "analytics",
                 "problem": "P", "solution": "S", "confidence_tier": "core",
                 "larger_patterns": ["a"], "smaller_patterns": ["a"],
                 "order_hint": 10},
            ],
        }
        errors = validate_pattern_language(data)
        assert any("cycle" in e for e in errors)

    def test_unknown_link(self):
        from protocol_pattern_language import validate_pattern_language
        data = {
            "version": "v1",
            "patterns": [
                {"id": "a", "name": "A", "stage": "ingest",
                 "problem": "P", "solution": "S", "confidence_tier": "core",
                 "larger_patterns": [], "smaller_patterns": ["nonexistent"],
                 "order_hint": 0},
            ],
        }
        errors = validate_pattern_language(data)
        assert any("unknown" in e for e in errors)


class TestPatternSequence:
    """Test topological sequencing."""

    def test_build_sequence(self):
        from protocol_pattern_language import build_sequence
        data = {
            "version": "v1",
            "patterns": [
                {"id": "report", "name": "Report", "stage": "report",
                 "problem": "P", "solution": "S", "confidence_tier": "core",
                 "larger_patterns": ["classify"], "smaller_patterns": [],
                 "order_hint": 30},
                {"id": "ingest", "name": "Ingest", "stage": "ingest",
                 "problem": "P", "solution": "S", "confidence_tier": "core",
                 "larger_patterns": [], "smaller_patterns": ["analytics"],
                 "order_hint": 0},
                {"id": "analytics", "name": "Analytics", "stage": "analytics",
                 "problem": "P", "solution": "S", "confidence_tier": "core",
                 "larger_patterns": ["ingest"], "smaller_patterns": ["classify"],
                 "order_hint": 10},
                {"id": "classify", "name": "Classify", "stage": "classify",
                 "problem": "P", "solution": "S", "confidence_tier": "core",
                 "larger_patterns": ["analytics"], "smaller_patterns": ["report"],
                 "order_hint": 20},
            ],
        }
        seq = build_sequence(data)
        ids = [s["id"] for s in seq]
        assert ids.index("ingest") < ids.index("analytics")
        assert ids.index("analytics") < ids.index("classify")
        assert ids.index("classify") < ids.index("report")


class TestOrchestrator:
    """Test pattern orchestration."""

    def test_orchestrate_record(self):
        from protocol_pattern_language import orchestrate_record

        sequence = [
            {"id": "ingest", "stage": "ingest", "confidence_tier": "core"},
            {"id": "analytics", "stage": "analytics", "confidence_tier": "core"},
        ]

        def ingest_handler(record, _):
            if not record.get("_ingested"):
                record["_ingested"] = True
                return True
            return False

        registry = {
            "ingest": ingest_handler,
            "analytics": lambda r, _: False,  # no-op
        }

        record = {"intervention": {"rapamycin_dose": 0.5}}
        updated, trace = orchestrate_record(record, sequence, registry=registry)
        assert updated["_ingested"] is True
        assert len(trace) == 2
        assert trace[0]["status"] == "applied"
        assert trace[1]["status"] == "noop"

    def test_pattern_refs_populated(self):
        from protocol_pattern_language import orchestrate_record

        sequence = [
            {"id": "ingest", "stage": "ingest", "confidence_tier": "core"},
            {"id": "classify", "stage": "classify", "confidence_tier": "core"},
        ]
        registry = {"ingest": lambda r, _: False, "classify": lambda r, _: False}

        record = {}
        updated, _ = orchestrate_record(record, sequence, registry=registry)
        assert "ingest" in updated["pattern_refs"]
        assert "classify" in updated["pattern_refs"]


class TestDefaultPatternLanguage:
    """Test loading the shipped pattern language file."""

    def test_load_default(self):
        from protocol_pattern_language import load_default_pattern_language
        data = load_default_pattern_language()
        assert "patterns" in data
        assert len(data["patterns"]) >= 6

    def test_default_validates(self):
        from protocol_pattern_language import (
            load_default_pattern_language, validate_pattern_language,
        )
        data = load_default_pattern_language()
        errors = validate_pattern_language(data)
        assert errors == [], f"Validation errors: {errors}"

    def test_default_sequence_builds(self):
        from protocol_pattern_language import (
            load_default_pattern_language, build_sequence,
        )
        data = load_default_pattern_language()
        seq = build_sequence(data)
        assert len(seq) >= 6
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && conda run -n mito-aging pytest tests/test_protocol_pattern_language.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the pattern language JSON and Python module**

First, `patterns/protocol_pattern_language.v1.json`:

```json
{
  "version": "v1",
  "description": "Protocol curation pipeline pattern language. Stages proceed from ingest through analytics, robustness, classification, review, and reporting.",
  "patterns": [
    {
      "id": "protocol_program",
      "name": "Protocol Program",
      "stage": "global",
      "problem": "Protocol classification can drift when analysis stages are run ad hoc.",
      "solution": "Establish a global pipeline structure that all downstream stages must follow.",
      "confidence_tier": "core",
      "larger_patterns": [],
      "smaller_patterns": ["ingest_protocols", "review_governance"],
      "order_hint": 0
    },
    {
      "id": "ingest_protocols",
      "name": "Ingest Protocols",
      "stage": "ingest",
      "problem": "Protocol records arrive from diverse sources with varying completeness.",
      "solution": "Normalize all records to ProtocolRecord schema with provenance metadata.",
      "confidence_tier": "core",
      "larger_patterns": ["protocol_program"],
      "smaller_patterns": ["analytics_profile"],
      "order_hint": 10
    },
    {
      "id": "analytics_profile",
      "name": "Analytics Profile",
      "stage": "analytics",
      "problem": "Classification is unreliable without simulation analytics.",
      "solution": "Compute 4-pillar health analytics and enrichment fields before classification.",
      "confidence_tier": "core",
      "larger_patterns": ["ingest_protocols"],
      "smaller_patterns": ["robustness_assessment", "classify_outcomes"],
      "order_hint": 20
    },
    {
      "id": "robustness_assessment",
      "name": "Robustness Assessment",
      "stage": "robustness",
      "problem": "Protocols that perform well under baseline conditions may fail under stress.",
      "solution": "Evaluate biological stress resilience via cramer-toolkit before final classification.",
      "confidence_tier": "adaptable",
      "larger_patterns": ["analytics_profile"],
      "smaller_patterns": ["review_governance"],
      "order_hint": 30
    },
    {
      "id": "classify_outcomes",
      "name": "Classify Outcomes",
      "stage": "classify",
      "problem": "Single-method classification misses edge cases and ambiguity.",
      "solution": "Run multi-method pipeline (rule + analytics-fit) and aggregate with confidence.",
      "confidence_tier": "core",
      "larger_patterns": ["analytics_profile"],
      "smaller_patterns": ["review_governance", "report_synthesis"],
      "order_hint": 40
    },
    {
      "id": "review_governance",
      "name": "Review Governance",
      "stage": "review",
      "problem": "Low-confidence or paradoxical protocols need human clinical judgment.",
      "solution": "Route flagged protocols to review queue with structured audit metadata.",
      "confidence_tier": "core",
      "larger_patterns": ["protocol_program", "robustness_assessment", "classify_outcomes"],
      "smaller_patterns": ["report_synthesis"],
      "order_hint": 50
    },
    {
      "id": "report_synthesis",
      "name": "Report Synthesis",
      "stage": "report",
      "problem": "Progress is hard to evaluate without integrated quality diagnostics.",
      "solution": "Generate summary reports with outcome distributions, confidence histograms, and coverage.",
      "confidence_tier": "core",
      "larger_patterns": ["classify_outcomes", "review_governance"],
      "smaller_patterns": [],
      "order_hint": 60
    }
  ]
}
```

Then `protocol_pattern_language.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && conda run -n mito-aging pytest tests/test_protocol_pattern_language.py -v`
Expected: 9 PASSED

**Step 5: Commit**

```bash
git add protocol_pattern_language.py patterns/protocol_pattern_language.v1.json tests/test_protocol_pattern_language.py
git commit -m "feat: add pattern language DAG pipeline (rosetta-motion port)"
```

---

## Task 7: Review Governance

**Files:**
- Create: `protocol_review.py`
- Test: `tests/test_protocol_review.py`

**Context:** Low-confidence and paradoxical protocols are routed to a review queue (JSONL file). Reviewers can accept, override classification, or flag for expert review. All decisions carry provenance (reason, timestamp). Ported from rosetta-motion's `review_queue_append()` and `_flag_review_needed()`.

**Step 1: Write the failing test**

```python
"""tests/test_protocol_review.py"""
import json
import pytest
from pathlib import Path


@pytest.fixture
def review_path(tmp_path):
    return tmp_path / "review_queue.jsonl"


class TestReviewQueue:
    """Test review queue operations."""

    def test_append_to_queue(self, review_path):
        from protocol_review import append_to_review_queue
        from protocol_record import ProtocolRecord

        rec = ProtocolRecord(
            intervention={"rapamycin_dose": 0.5},
            patient={"baseline_age": 70.0},
            outcome_class="stable",
            confidence=0.4,
            source="test",
        )
        append_to_review_queue(review_path, rec, reason="low_confidence")
        assert review_path.exists()
        lines = review_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["reason"] == "low_confidence"
        assert entry["status"] == "pending"

    def test_multiple_appends(self, review_path):
        from protocol_review import append_to_review_queue
        from protocol_record import ProtocolRecord

        for i in range(3):
            rec = ProtocolRecord(
                intervention={"rapamycin_dose": 0.1 * i},
                patient={"baseline_age": 70.0},
                source="test",
            )
            append_to_review_queue(review_path, rec, reason=f"reason_{i}")

        lines = review_path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_load_review_queue(self, review_path):
        from protocol_review import append_to_review_queue, load_review_queue
        from protocol_record import ProtocolRecord

        rec = ProtocolRecord(
            intervention={"rapamycin_dose": 0.5},
            patient={"baseline_age": 70.0},
            source="test",
        )
        append_to_review_queue(review_path, rec, reason="paradoxical")
        queue = load_review_queue(review_path)
        assert len(queue) == 1
        assert queue[0]["reason"] == "paradoxical"

    def test_needs_review(self):
        from protocol_review import needs_review
        assert needs_review(confidence=0.3, outcome_class="stable") is True
        assert needs_review(confidence=0.9, outcome_class="thriving") is False
        assert needs_review(confidence=0.9, outcome_class="paradoxical") is True

    def test_resolve_review(self, review_path):
        from protocol_review import (
            append_to_review_queue, load_review_queue, resolve_review,
        )
        from protocol_record import ProtocolRecord

        rec = ProtocolRecord(
            intervention={"rapamycin_dose": 0.5},
            patient={"baseline_age": 70.0},
            outcome_class="stable",
            confidence=0.4,
            source="test",
        )
        append_to_review_queue(review_path, rec, reason="low_confidence")
        resolved = resolve_review(
            review_path, index=0,
            decision="accept",
            reviewer="human",
            notes="Looks correct on inspection",
        )
        assert resolved["status"] == "resolved"
        assert resolved["decision"] == "accept"
        assert resolved["reviewer"] == "human"

        queue = load_review_queue(review_path)
        assert queue[0]["status"] == "resolved"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && conda run -n mito-aging pytest tests/test_protocol_review.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
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
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && conda run -n mito-aging pytest tests/test_protocol_review.py -v`
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add protocol_review.py tests/test_protocol_review.py
git commit -m "feat: add review governance for uncertain protocol classifications"
```

---

## Task 8: Pipeline Runner + Ingest Adapters

**Files:**
- Create: `run_protocol_pipeline.py`
- Test: `tests/test_protocol_pipeline.py`

**Context:** The end-to-end pipeline runner that ties everything together. Analogous to rosetta-motion's `run_pipeline.py`. Includes ingest adapters that import protocols from existing experiment artifacts (dark_matter.json, reachable_set, EA optimizer outputs). The pipeline runs: ingest → enrich → classify → rewrite rules → review routing → save + report.

**Step 1: Write the failing test**

```python
"""tests/test_protocol_pipeline.py"""
import json
import pytest
from pathlib import Path


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "pipeline_output"


class TestIngestAdapters:
    """Test importing protocols from existing artifact formats."""

    def test_ingest_from_dark_matter(self, tmp_path):
        from run_protocol_pipeline import ingest_dark_matter
        artifact = {
            "moderate_patient": {
                "total_sampled": 2,
                "by_category": {
                    "thriving": {
                        "count": 1,
                        "examples": [{
                            "intervention": {"rapamycin_dose": 0.5, "nad_supplement": 0.75,
                                             "senolytic_dose": 0.5, "yamanaka_intensity": 0.0,
                                             "transplant_rate": 0.0, "exercise_level": 0.5},
                            "final_atp": 0.85, "final_het": 0.20,
                        }],
                    },
                    "collapsed": {
                        "count": 1,
                        "examples": [{
                            "intervention": {"rapamycin_dose": 0.0, "nad_supplement": 0.0,
                                             "senolytic_dose": 0.0, "yamanaka_intensity": 1.0,
                                             "transplant_rate": 0.0, "exercise_level": 0.0},
                            "final_atp": 0.05, "final_het": 0.92,
                        }],
                    },
                },
            },
        }
        path = tmp_path / "dark_matter.json"
        path.write_text(json.dumps(artifact))
        records = ingest_dark_matter(path)
        assert len(records) == 2
        assert records[0].source == "dark_matter"

    def test_ingest_from_simulation(self):
        from run_protocol_pipeline import ingest_from_simulation
        from protocol_record import ProtocolRecord

        intervention = {"rapamycin_dose": 0.5, "nad_supplement": 0.75,
                        "senolytic_dose": 0.5, "yamanaka_intensity": 0.0,
                        "transplant_rate": 0.0, "exercise_level": 0.5}
        patient = {"baseline_age": 70.0, "baseline_heteroplasmy": 0.30,
                   "baseline_nad_level": 0.6, "genetic_vulnerability": 1.0,
                   "metabolic_demand": 1.0, "inflammation_level": 0.25}
        record = ingest_from_simulation(intervention, patient, source="manual")
        assert isinstance(record, ProtocolRecord)
        assert record.source == "manual"
        # Should have run simulation and computed analytics
        assert "energy" in record.analytics
        assert "damage" in record.analytics


class TestPipelineRunner:
    """Test the end-to-end pipeline."""

    def test_run_pipeline_minimal(self, output_dir):
        from run_protocol_pipeline import run_pipeline
        from protocol_record import ProtocolRecord

        records = [
            ProtocolRecord(
                intervention={"rapamycin_dose": 0.5, "nad_supplement": 0.75,
                               "senolytic_dose": 0.5, "yamanaka_intensity": 0.0,
                               "transplant_rate": 0.0, "exercise_level": 0.5},
                patient={"baseline_age": 70.0, "baseline_heteroplasmy": 0.30,
                         "baseline_nad_level": 0.6, "genetic_vulnerability": 1.0,
                         "metabolic_demand": 1.0, "inflammation_level": 0.25},
                source="test",
            ),
        ]
        result = run_pipeline(records, output_dir=output_dir)
        assert result["total_processed"] == 1
        assert (output_dir / "protocol_dictionary.json").exists()

    def test_pipeline_classifies(self, output_dir):
        from run_protocol_pipeline import run_pipeline, ingest_from_simulation

        record = ingest_from_simulation(
            intervention={"rapamycin_dose": 0.5, "nad_supplement": 0.75,
                           "senolytic_dose": 0.5, "yamanaka_intensity": 0.0,
                           "transplant_rate": 0.0, "exercise_level": 0.5},
            patient={"baseline_age": 70.0, "baseline_heteroplasmy": 0.30,
                     "baseline_nad_level": 0.6, "genetic_vulnerability": 1.0,
                     "metabolic_demand": 1.0, "inflammation_level": 0.25},
            source="test",
        )
        result = run_pipeline([record], output_dir=output_dir)
        assert result["total_processed"] == 1

        # Load and check the dictionary
        dict_path = output_dir / "protocol_dictionary.json"
        data = json.loads(dict_path.read_text())
        assert len(data["records"]) == 1
        assert data["records"][0]["outcome_class"] is not None
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && conda run -n mito-aging pytest tests/test_protocol_pipeline.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
"""run_protocol_pipeline.py — End-to-end protocol curation pipeline.

Ported from rosetta-motion's run_pipeline.py. Orchestrates:
  1. Ingest protocols from various sources
  2. Run simulation + compute analytics (if not already present)
  3. Enrich records (complexity, clinical signature, prototype)
  4. Classify outcomes (multi-method pipeline)
  5. Apply rewrite rules (normalization, clinical flags)
  6. Route low-confidence to review queue
  7. Save dictionary + report

Usage:
    python run_protocol_pipeline.py --source dark_matter
    python run_protocol_pipeline.py --intervention '{"rapamycin_dose":0.5}'
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from protocol_record import ProtocolRecord
from protocol_dictionary import ProtocolDictionary
from protocol_enrichment import enrich_record
from protocol_classifier import multi_classify
from protocol_rewrite_rules import apply_rules
from protocol_review import needs_review, append_to_review_queue

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "artifacts" / "protocol_pipeline"
DEFAULT_RULES = ROOT / "patterns" / "protocol_rewrite_rules.v1.json"


def ingest_from_simulation(
    intervention: dict[str, Any],
    patient: dict[str, Any],
    source: str = "manual",
    method: str = "",
) -> ProtocolRecord:
    """Create a ProtocolRecord by running a simulation and computing analytics."""
    from simulator import simulate
    from analytics import compute_all

    result = simulate(intervention=intervention, patient=patient)
    baseline = simulate(patient=patient)
    analytics = compute_all(result, baseline)

    final_states = result["states"][-1]
    het = result["heteroplasmy"][-1]

    return ProtocolRecord(
        intervention=intervention,
        patient=patient,
        source=source,
        method=method,
        analytics=analytics,
        simulation={
            "final_atp": float(final_states[2]),
            "final_het": float(het),
            "final_nad": float(final_states[4]),
            "final_sen": float(final_states[5]),
        },
    )


def ingest_dark_matter(path: Path) -> list[ProtocolRecord]:
    """Import protocols from dark_matter.py JSON artifact."""
    data = json.loads(path.read_text())
    records = []
    for patient_key, patient_data in data.items():
        categories = patient_data.get("by_category", {})
        for category, cat_data in categories.items():
            for example in cat_data.get("examples", []):
                records.append(ProtocolRecord(
                    intervention=example.get("intervention", {}),
                    patient={},  # dark_matter doesn't store full patient dict
                    source="dark_matter",
                    method="random_sample",
                    outcome_class=category,
                    simulation={
                        "final_atp": example.get("final_atp"),
                        "final_het": example.get("final_het"),
                    },
                    meta={"patient_key": patient_key},
                ))
    return records


def run_pipeline(
    records: list[ProtocolRecord],
    output_dir: Path | None = None,
    rules_path: Path | None = None,
    classify_pipeline: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full protocol curation pipeline.

    Steps:
      1. Simulate (if analytics missing)
      2. Enrich (complexity, clinical signature, prototype)
      3. Classify (multi-method)
      4. Apply rewrite rules
      5. Route to review if needed
      6. Save dictionary + report
    """
    out_dir = Path(output_dir or DEFAULT_OUTPUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    rules_file = Path(rules_path or DEFAULT_RULES)

    dict_path = out_dir / "protocol_dictionary.json"
    review_path = out_dir / "review_queue.jsonl"
    report_path = out_dir / "pipeline_report.json"

    pd = ProtocolDictionary(dict_path)
    classify_pipe = classify_pipeline or ["rule", "analytics_fit"]

    # Load rewrite rules if available
    rules_data = None
    if rules_file.exists():
        rules_data = json.loads(rules_file.read_text())

    processed = 0
    reviewed = 0

    for rec in records:
        # Step 1: Simulate if needed
        if not rec.analytics and rec.intervention:
            try:
                rec = ingest_from_simulation(
                    rec.intervention, rec.patient,
                    source=rec.source, method=rec.method,
                )
            except Exception:
                pass  # Keep record without analytics

        # Step 2: Enrich
        rec = enrich_record(rec)

        # Step 3: Classify
        sim = rec.simulation
        final_atp = sim.get("final_atp", 0.0) or 0.0
        final_het = sim.get("final_het", 0.0) or 0.0
        baseline_atp = 0.6  # approximate no-treatment baseline
        baseline_het = rec.patient.get("baseline_heteroplasmy", 0.30)

        cls_result = multi_classify(
            final_atp=final_atp, final_het=final_het,
            baseline_atp=baseline_atp, baseline_het=baseline_het,
            analytics=rec.analytics,
            pipeline=classify_pipe,
        )
        rec.outcome_class = cls_result.get("outcome_class")
        rec.confidence = cls_result.get("confidence")
        rec.classifications = cls_result.get("methods", {})

        # Step 4: Apply rewrite rules
        if rules_data is not None:
            flat = rec.to_dict()
            # Merge simulation fields into flat dict for rule matching
            flat.update(rec.simulation)
            flat.update(rec.intervention)
            updated, _trace = apply_rules(flat, rules_data)
            # Extract flags back
            if "flags" in updated:
                rec.meta["flags"] = updated["flags"]

        # Step 5: Route to review if needed
        if needs_review(rec.confidence, rec.outcome_class):
            reason = "paradoxical" if rec.outcome_class == "paradoxical" else "low_confidence"
            append_to_review_queue(review_path, rec, reason=reason)
            reviewed += 1

        pd.add(rec)
        processed += 1

    pd.save()

    # Step 6: Generate report
    report = {
        "total_processed": processed,
        "total_reviewed": reviewed,
        "summary": pd.summary(),
    }
    report_path.write_text(json.dumps(report, indent=2))

    return report


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Protocol curation pipeline")
    ap.add_argument("--source", choices=["dark_matter", "manual"],
                    default="manual", help="Data source to ingest")
    ap.add_argument("--artifact", type=str, default=None,
                    help="Path to source artifact (e.g. dark_matter.json)")
    ap.add_argument("--intervention", type=str, default=None,
                    help="JSON intervention dict for manual mode")
    ap.add_argument("--patient", type=str, default=None,
                    help="JSON patient dict for manual mode")
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                    help="Output directory")
    args = ap.parse_args()

    records: list[ProtocolRecord] = []

    if args.source == "dark_matter" and args.artifact:
        records = ingest_dark_matter(Path(args.artifact))
    elif args.intervention:
        iv = json.loads(args.intervention)
        pt = json.loads(args.patient) if args.patient else {
            "baseline_age": 70.0, "baseline_heteroplasmy": 0.30,
            "baseline_nad_level": 0.6, "genetic_vulnerability": 1.0,
            "metabolic_demand": 1.0, "inflammation_level": 0.25,
        }
        rec = ingest_from_simulation(iv, pt, source="manual", method="cli")
        records = [rec]

    if not records:
        print("No records to process. Use --source or --intervention.")
        return

    result = run_pipeline(records, output_dir=Path(args.output))
    print(f"Pipeline complete: {result['total_processed']} processed, "
          f"{result['total_reviewed']} sent to review")
    print(f"Output: {args.output}/protocol_dictionary.json")


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && conda run -n mito-aging pytest tests/test_protocol_pipeline.py -v`
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add run_protocol_pipeline.py tests/test_protocol_pipeline.py
git commit -m "feat: add protocol curation pipeline runner with ingest adapters"
```

---

## Task 9: Integration Test + CLAUDE.md Update

**Files:**
- Create: `tests/test_protocol_pipeline_integration.py`
- Modify: `CLAUDE.md`

**Context:** End-to-end integration test that runs the full pipeline: simulate a batch of protocols → enrich → classify → rewrite rules → review routing → save dictionary. Also update CLAUDE.md with the new modules.

**Step 1: Write the integration test**

```python
"""tests/test_protocol_pipeline_integration.py — End-to-end pipeline integration test."""
import json
import pytest
from pathlib import Path

from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT


class TestFullPipeline:
    """End-to-end integration test for the protocol curation pipeline."""

    def test_multi_protocol_pipeline(self, tmp_path):
        """Process multiple protocols through the full pipeline."""
        from run_protocol_pipeline import run_pipeline, ingest_from_simulation

        # Generate a range of protocols
        protocols = []
        for rap in [0.0, 0.25, 0.5, 0.75]:
            for nad in [0.0, 0.5, 1.0]:
                rec = ingest_from_simulation(
                    intervention={
                        "rapamycin_dose": rap, "nad_supplement": nad,
                        "senolytic_dose": 0.0, "yamanaka_intensity": 0.0,
                        "transplant_rate": 0.0, "exercise_level": 0.0,
                    },
                    patient=dict(DEFAULT_PATIENT),
                    source="integration_test",
                    method="grid_sweep",
                )
                protocols.append(rec)

        result = run_pipeline(protocols, output_dir=tmp_path / "output")

        # Verify pipeline ran
        assert result["total_processed"] == 12

        # Verify dictionary was saved
        dict_path = tmp_path / "output" / "protocol_dictionary.json"
        assert dict_path.exists()
        data = json.loads(dict_path.read_text())
        assert len(data["records"]) == 12

        # Verify all records are classified
        for rec in data["records"]:
            assert rec["outcome_class"] is not None
            assert rec["confidence"] is not None
            assert rec["enrichment"] != {}

        # Verify at least some diversity in outcomes
        classes = set(rec["outcome_class"] for rec in data["records"])
        assert len(classes) >= 2, f"Expected outcome diversity, got: {classes}"

    def test_pipeline_report_generated(self, tmp_path):
        """Verify pipeline report is written."""
        from run_protocol_pipeline import run_pipeline, ingest_from_simulation

        rec = ingest_from_simulation(
            intervention=dict(DEFAULT_INTERVENTION),
            patient=dict(DEFAULT_PATIENT),
            source="test",
        )
        run_pipeline([rec], output_dir=tmp_path / "output")

        report_path = tmp_path / "output" / "pipeline_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert "total_processed" in report
        assert "summary" in report
```

**Step 2: Run test to verify it passes**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && conda run -n mito-aging pytest tests/test_protocol_pipeline_integration.py -v`
Expected: 2 PASSED

**Step 3: Update CLAUDE.md**

Add a new section after the Precision Medicine Expansion section in CLAUDE.md documenting the protocol dictionary pipeline, its modules, and the pattern language.

**Step 4: Run full test suite**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && conda run -n mito-aging pytest tests/ -v`
Expected: All tests pass (previous ~385 + new ~60 = ~445 tests)

**Step 5: Commit**

```bash
git add tests/test_protocol_pipeline_integration.py CLAUDE.md
git commit -m "feat: add integration test + update CLAUDE.md with protocol dictionary pipeline"
```

---

## Summary

| Task | New Files | Tests | Rosetta-Motion Source |
|------|-----------|-------|---------------------|
| 1. Protocol Record Schema | `protocol_record.py` | 9 | `DiscoveryRecord` |
| 2. Protocol Dictionary | `protocol_dictionary.py` | 8 | `MotionDiscovery` |
| 3. Protocol Enrichment | `protocol_enrichment.py` | 12 | `controller_simplicity()`, `sensory_signature()`, `prototype_descriptor()` |
| 4. Multi-Labeler Classifier | `protocol_classifier.py` | 10 | `discover_label()`, `rule_based_label()`, `metric_fit_score()` |
| 5. Rewrite Rules Engine | `protocol_rewrite_rules.py` + `patterns/` | 11 | `rewrite_rules.py` |
| 6. Pattern Language Pipeline | `protocol_pattern_language.py` + `patterns/` | 9 | `pattern_language.py`, `pattern_orchestrator.py`, `pattern_registry.py` |
| 7. Review Governance | `protocol_review.py` | 5 | `review_queue_append()`, `_flag_review_needed()` |
| 8. Pipeline Runner | `run_protocol_pipeline.py` | 4 | `run_pipeline.py` |
| 9. Integration + Docs | `tests/test_protocol_pipeline_integration.py` | 2 | — |
| **Total** | **9 new files + 2 JSON** | **~70** | — |

## Future Work (not in this plan)

- **LLM classifier** (`llm_classify` step): Send protocol + analytics to Ollama for clinical assessment. Deferred because it requires Ollama running and is optional.
- **Thompson motif indexing**: Protocol fingerprinting for cross-run retrieval. Nice to have but the `protocol_fingerprint()` function covers basic deduplication.
- **Offline HTML dashboard**: Static D3.js visualization of the protocol dictionary. Deferred to viz-tools integration.
- **Additional ingest adapters**: Import from reachable_set.py, EA optimizer, character_seed_experiment, competing_evaluators artifacts.
- **Loss function alignment scoring**: Composite loss measuring protocol-to-patient-description alignment (rosetta-motion's word_gait_loss pattern). Could be useful for the archetype_matchmaker.
