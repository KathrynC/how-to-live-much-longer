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
