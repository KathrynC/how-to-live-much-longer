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
        """to_dict -> from_dict preserves all fields."""
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
