"""Tests for genetics module — genotype-to-parameter mapping."""
import pytest


class TestGenotypeModifiers:
    def test_non_carrier_returns_neutral(self):
        from genetics_module import compute_genetic_modifiers
        mods = compute_genetic_modifiers(apoe_genotype=0, foxo3_protective=0, cd38_risk=0)
        assert mods['vulnerability'] == pytest.approx(1.0)
        assert mods['inflammation'] == pytest.approx(1.0)

    def test_apoe4_het_increases_vulnerability(self):
        from genetics_module import compute_genetic_modifiers
        mods = compute_genetic_modifiers(apoe_genotype=1, foxo3_protective=0, cd38_risk=0)
        assert mods['vulnerability'] > 1.0

    def test_apoe4_hom_worse_than_het(self):
        from genetics_module import compute_genetic_modifiers
        het = compute_genetic_modifiers(apoe_genotype=1, foxo3_protective=0, cd38_risk=0)
        hom = compute_genetic_modifiers(apoe_genotype=2, foxo3_protective=0, cd38_risk=0)
        assert hom['vulnerability'] > het['vulnerability']

    def test_foxo3_reduces_vulnerability(self):
        from genetics_module import compute_genetic_modifiers
        without = compute_genetic_modifiers(apoe_genotype=0, foxo3_protective=0, cd38_risk=0)
        with_foxo = compute_genetic_modifiers(apoe_genotype=0, foxo3_protective=1, cd38_risk=0)
        assert with_foxo['vulnerability'] < without['vulnerability']

    def test_cd38_reduces_nad_efficiency(self):
        from genetics_module import compute_genetic_modifiers
        mods = compute_genetic_modifiers(apoe_genotype=0, foxo3_protective=0, cd38_risk=1)
        assert mods['nad_efficiency'] < 1.0

    def test_combined_apoe4_foxo3(self):
        from genetics_module import compute_genetic_modifiers
        mods = compute_genetic_modifiers(apoe_genotype=1, foxo3_protective=1, cd38_risk=0)
        apoe_only = compute_genetic_modifiers(apoe_genotype=1, foxo3_protective=0, cd38_risk=0)
        assert mods['vulnerability'] < apoe_only['vulnerability']


class TestSexModifiers:
    def test_male_returns_neutral(self):
        from genetics_module import compute_sex_modifiers
        mods = compute_sex_modifiers(sex='M', menopause_status='pre', estrogen_therapy=0)
        assert mods['inflammation_delta'] == pytest.approx(0.0)
        assert mods['heteroplasmy_multiplier'] == pytest.approx(1.0)

    def test_postmenopausal_increases_inflammation(self):
        from genetics_module import compute_sex_modifiers
        mods = compute_sex_modifiers(sex='F', menopause_status='post', estrogen_therapy=0)
        assert mods['inflammation_delta'] > 0

    def test_estrogen_therapy_reduces_menopause_effect(self):
        from genetics_module import compute_sex_modifiers
        without = compute_sex_modifiers(sex='F', menopause_status='post', estrogen_therapy=0)
        with_hrt = compute_sex_modifiers(sex='F', menopause_status='post', estrogen_therapy=1)
        assert with_hrt['inflammation_delta'] < without['inflammation_delta']


class TestApplyGeneticModifiers:
    def test_applies_to_patient_dict(self):
        from genetics_module import apply_genetic_modifiers
        patient = {
            'baseline_age': 63.0,
            'baseline_heteroplasmy': 0.5,
            'baseline_nad_level': 0.6,
            'genetic_vulnerability': 1.0,
            'metabolic_demand': 1.0,
            'inflammation_level': 0.25,
        }
        expanded = {
            'apoe_genotype': 1,
            'foxo3_protective': 0,
            'cd38_risk': 0,
            'sex': 'F',
            'menopause_status': 'post',
            'estrogen_therapy': 0,
        }
        result = apply_genetic_modifiers(patient, expanded)
        assert result['genetic_vulnerability'] > 1.0
        assert result['inflammation_level'] > 0.25
