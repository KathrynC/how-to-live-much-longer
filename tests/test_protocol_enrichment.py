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
