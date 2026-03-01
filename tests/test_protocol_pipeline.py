"""tests/test_protocol_pipeline.py"""
import json
import pytest
from pathlib import Path
import numpy as np


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "pipeline_output"


class TestIngestAdapters:
    """Test importing protocols from existing artifact formats."""

    def test_ingest_from_dark_matter(self, tmp_path):
        from run_protocol_pipeline import ingest_dark_matter
        artifact = {
            "experiment": "dark_matter",
            "results": [
                {
                    "trial": 1, "patient": "moderate_60",
                    "intervention": {"rapamycin_dose": 0.5, "nad_supplement": 0.75,
                                     "senolytic_dose": 0.5, "yamanaka_intensity": 0.0,
                                     "transplant_rate": 0.0, "exercise_level": 0.5},
                    "final_atp": 0.85, "final_het": 0.20, "category": "thriving",
                },
                {
                    "trial": 2, "patient": "near_cliff_75",
                    "intervention": {"rapamycin_dose": 0.0, "nad_supplement": 0.0,
                                     "senolytic_dose": 0.0, "yamanaka_intensity": 1.0,
                                     "transplant_rate": 0.0, "exercise_level": 0.0},
                    "final_atp": 0.05, "final_het": 0.92, "category": "collapsed",
                },
            ],
        }
        path = tmp_path / "dark_matter.json"
        path.write_text(json.dumps(artifact))
        records = ingest_dark_matter(path)
        assert len(records) == 2
        assert records[0].source == "dark_matter"
        assert records[0].patient["baseline_age"] == 60.0
        assert records[1].patient["baseline_age"] == 75.0

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

    def test_pipeline_caches_patient_baseline_simulation(self, output_dir, monkeypatch):
        from run_protocol_pipeline import run_pipeline
        from protocol_record import ProtocolRecord
        import simulator

        call_count = {"n": 0}

        def fake_simulate(*args, **kwargs):
            call_count["n"] += 1
            patient = kwargs.get("patient", {}) or {}
            base_het = float(patient.get("baseline_heteroplasmy", 0.3))
            return {
                "states": np.array([
                    [1.0, 0.1, 1.0, 0.1, 0.6, 0.1, 1.0, 0.1],
                    [1.0, 0.1, 0.8, 0.1, 0.6, 0.1, 1.0, 0.1],
                ], dtype=float),
                "heteroplasmy": np.array([base_het, base_het + 0.05], dtype=float),
                "time": np.array([0.0, 30.0], dtype=float),
            }

        monkeypatch.setattr(simulator, "simulate", fake_simulate)

        patient = {
            "baseline_age": 70.0,
            "baseline_heteroplasmy": 0.30,
            "baseline_nad_level": 0.6,
            "genetic_vulnerability": 1.0,
            "metabolic_demand": 1.0,
            "inflammation_level": 0.25,
        }
        records = [
            ProtocolRecord(
                intervention={"rapamycin_dose": 0.5},
                patient=dict(patient),
                analytics={"energy": {"atp_final": 0.8}, "damage": {"het_final": 0.3}},
                simulation={"final_atp": 0.8, "final_het": 0.3},
                source="test",
            ),
            ProtocolRecord(
                intervention={"rapamycin_dose": 0.25},
                patient=dict(patient),
                analytics={"energy": {"atp_final": 0.75}, "damage": {"het_final": 0.32}},
                simulation={"final_atp": 0.75, "final_het": 0.32},
                source="test",
            ),
        ]

        result = run_pipeline(records, output_dir=output_dir)
        assert result["total_processed"] == 2
        # Same patient should trigger one cached baseline simulation.
        assert call_count["n"] == 1

    def test_pipeline_records_structured_errors(self, output_dir):
        from run_protocol_pipeline import run_pipeline
        from protocol_record import ProtocolRecord

        # Invalid intervention payload triggers ingest simulation failure.
        bad_record = ProtocolRecord(
            intervention={"invalid_intervention_key": "bad"},
            patient={
                "baseline_age": 70.0,
                "baseline_heteroplasmy": 0.30,
                "baseline_nad_level": 0.6,
                "genetic_vulnerability": 1.0,
                "metabolic_demand": 1.0,
                "inflammation_level": 0.25,
            },
            source="test",
        )

        result = run_pipeline([bad_record], output_dir=output_dir)
        assert result["total_processed"] == 1

        data = json.loads((output_dir / "protocol_dictionary.json").read_text())
        record_meta = data["records"][0].get("meta", {})
        assert "errors" in record_meta
        assert len(record_meta["errors"]) >= 1
        assert any(e.get("stage") == "ingest_from_simulation" for e in record_meta["errors"])
