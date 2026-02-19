"""Tests for expansion constants (precision medicine upgrade)."""
import pytest


class TestGriefConstants:
    def test_grief_ros_factor_range(self):
        from constants import GRIEF_ROS_FACTOR
        assert 0 < GRIEF_ROS_FACTOR <= 1.0

    def test_grief_nad_decay_range(self):
        from constants import GRIEF_NAD_DECAY
        assert 0 < GRIEF_NAD_DECAY <= 1.0

    def test_love_buffer_factor_range(self):
        from constants import LOVE_BUFFER_FACTOR
        assert 0 < LOVE_BUFFER_FACTOR <= 1.0


class TestGenotypeMultipliers:
    def test_genotype_multipliers_has_three_profiles(self):
        from constants import GENOTYPE_MULTIPLIERS
        assert 'apoe4_het' in GENOTYPE_MULTIPLIERS
        assert 'apoe4_hom' in GENOTYPE_MULTIPLIERS
        assert 'foxo3_protective' in GENOTYPE_MULTIPLIERS
        assert 'cd38_risk' in GENOTYPE_MULTIPLIERS

    def test_apoe4_het_mitophagy_less_than_baseline(self):
        from constants import GENOTYPE_MULTIPLIERS
        assert GENOTYPE_MULTIPLIERS['apoe4_het']['mitophagy_efficiency'] < 1.0

    def test_apoe4_hom_worse_than_het(self):
        from constants import GENOTYPE_MULTIPLIERS
        het = GENOTYPE_MULTIPLIERS['apoe4_het']
        hom = GENOTYPE_MULTIPLIERS['apoe4_hom']
        assert hom['mitophagy_efficiency'] < het['mitophagy_efficiency']
        assert hom['inflammation'] > het['inflammation']

    def test_foxo3_protective_improves_mitophagy(self):
        from constants import GENOTYPE_MULTIPLIERS
        assert GENOTYPE_MULTIPLIERS['foxo3_protective']['mitophagy_efficiency'] > 1.0

    def test_cd38_risk_reduces_nad(self):
        from constants import GENOTYPE_MULTIPLIERS
        assert GENOTYPE_MULTIPLIERS['cd38_risk']['nad_efficiency'] < 1.0


class TestAlcoholConstants:
    def test_alcohol_apoe4_synergy_above_one(self):
        from constants import ALCOHOL_APOE4_SYNERGY
        assert ALCOHOL_APOE4_SYNERGY > 1.0


class TestCoffeeConstants:
    def test_coffee_max_beneficial_cups(self):
        from constants import COFFEE_MAX_BENEFICIAL_CUPS
        assert COFFEE_MAX_BENEFICIAL_CUPS == 3

    def test_coffee_preparation_multipliers(self):
        from constants import COFFEE_PREPARATION_MULTIPLIERS
        assert COFFEE_PREPARATION_MULTIPLIERS['filtered'] >= COFFEE_PREPARATION_MULTIPLIERS['unfiltered']
        assert COFFEE_PREPARATION_MULTIPLIERS['unfiltered'] >= COFFEE_PREPARATION_MULTIPLIERS['instant']


class TestSupplementConstants:
    def test_all_half_max_positive(self):
        from constants import (
            NR_HALF_MAX, DHA_HALF_MAX, COQ10_HALF_MAX,
            RESVERATROL_HALF_MAX, PQQ_HALF_MAX, ALA_HALF_MAX,
            VITAMIN_D_HALF_MAX, B_COMPLEX_HALF_MAX,
            MAGNESIUM_HALF_MAX, ZINC_HALF_MAX, SELENIUM_HALF_MAX,
        )
        for hm in [NR_HALF_MAX, DHA_HALF_MAX, COQ10_HALF_MAX,
                    RESVERATROL_HALF_MAX, PQQ_HALF_MAX, ALA_HALF_MAX,
                    VITAMIN_D_HALF_MAX, B_COMPLEX_HALF_MAX,
                    MAGNESIUM_HALF_MAX, ZINC_HALF_MAX, SELENIUM_HALF_MAX]:
            assert hm > 0

    def test_nr_max_effect_is_largest(self):
        from constants import MAX_NR_EFFECT, MAX_DHA_EFFECT, MAX_COQ10_EFFECT
        assert MAX_NR_EFFECT > MAX_DHA_EFFECT
        assert MAX_NR_EFFECT > MAX_COQ10_EFFECT


class TestMEF2Constants:
    def test_mef2_induction_rate_positive(self):
        from constants import MEF2_INDUCTION_RATE
        assert MEF2_INDUCTION_RATE > 0

    def test_ha_decay_slower_than_induction(self):
        from constants import HA_INDUCTION_RATE, HA_DECAY_RATE
        assert HA_DECAY_RATE < HA_INDUCTION_RATE


class TestAmyloidTauConstants:
    def test_tau_toxicity_greater_than_amyloid(self):
        from constants import AMYLOID_TOXICITY, TAU_TOXICITY
        assert TAU_TOXICITY > AMYLOID_TOXICITY

    def test_resilience_weights_sum_to_one(self):
        from constants import RESILIENCE_WEIGHTS
        total = sum(RESILIENCE_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.01)


class TestCognitiveReserveConstants:
    def test_cr_growth_rates_ordered(self):
        from constants import CR_GROWTH_RATE_BY_ACTIVITY
        assert CR_GROWTH_RATE_BY_ACTIVITY['collaborative_novel'] > CR_GROWTH_RATE_BY_ACTIVITY['solitary_routine']
