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
