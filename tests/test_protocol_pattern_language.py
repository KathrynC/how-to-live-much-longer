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
