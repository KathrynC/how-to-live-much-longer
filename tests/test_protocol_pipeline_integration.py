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

        # Use diverse patients to ensure outcome diversity across the grid:
        # - healthy young patient (should produce thriving/stable outcomes)
        # - default patient (moderate outcomes)
        # - near-cliff elderly patient (should produce declining/collapsed outcomes)
        patients = [
            {"baseline_age": 30.0, "baseline_heteroplasmy": 0.05,
             "baseline_nad_level": 0.9, "genetic_vulnerability": 0.5,
             "metabolic_demand": 1.0, "inflammation_level": 0.05},
            dict(DEFAULT_PATIENT),
            {"baseline_age": 85.0, "baseline_heteroplasmy": 0.75,
             "baseline_nad_level": 0.3, "genetic_vulnerability": 1.5,
             "metabolic_demand": 1.5, "inflammation_level": 0.75},
        ]

        # Generate a range of protocols: 4 rapamycin doses x 3 patients = 12
        protocols = []
        for rap in [0.0, 0.25, 0.5, 0.75]:
            for patient in patients:
                rec = ingest_from_simulation(
                    intervention={
                        "rapamycin_dose": rap, "nad_supplement": 0.0,
                        "senolytic_dose": 0.0, "yamanaka_intensity": 0.0,
                        "transplant_rate": 0.0, "exercise_level": 0.0,
                    },
                    patient=dict(patient),
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
