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
