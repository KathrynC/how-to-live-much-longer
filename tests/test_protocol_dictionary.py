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
