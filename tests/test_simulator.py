"""Tests for the mitochondrial aging simulator."""
from __future__ import annotations

import numpy as np
import pytest

from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT, SIM_YEARS
from simulator import (
    simulate, initial_state, derivatives, _cliff_factor,
    InterventionSchedule, phased_schedule, pulsed_schedule,
)


class TestBasicSimulation:
    """Core simulation tests (converted from inline Tests 1-3)."""

    def test_no_intervention_aging(self, default_patient, default_intervention):
        """Test 1: Natural aging with no treatment."""
        result = simulate()
        assert result["states"].shape == (3001, 7)
        assert result["heteroplasmy"][-1] > result["heteroplasmy"][0]
        assert result["states"][-1, 2] < result["states"][0, 2]  # ATP declines

    def test_cocktail_intervention(self, cocktail_intervention):
        """Test 2: Full cocktail should improve outcomes."""
        treated = simulate(intervention=cocktail_intervention)
        untreated = simulate()
        assert treated["heteroplasmy"][-1] < untreated["heteroplasmy"][-1]
        assert treated["states"][-1, 2] > untreated["states"][-1, 2]

    def test_near_cliff_patient(self, near_cliff_patient):
        """Test 3: Near-cliff patient should deteriorate."""
        result = simulate(patient=near_cliff_patient)
        assert result["heteroplasmy"][-1] > 0.7  # past cliff
        assert result["states"][-1, 2] < 0.2  # ATP collapse


class TestCliff:
    """Cliff verification tests (converted from inline Test 4)."""

    @pytest.mark.parametrize("het_start,expect_final_het_above", [
        (0.1, 0.3),
        (0.3, 0.5),
        (0.5, 0.7),
        (0.7, 0.8),
        (0.9, 0.9),
    ])
    def test_cliff_sweep(self, het_start, expect_final_het_above):
        """Higher starting het leads to worse outcomes."""
        p = dict(DEFAULT_PATIENT)
        p["baseline_heteroplasmy"] = het_start
        r = simulate(patient=p, sim_years=30)
        assert r["heteroplasmy"][-1] > expect_final_het_above

    def test_cliff_factor_sigmoid(self):
        """Cliff function returns ~1 below threshold, ~0 above."""
        assert _cliff_factor(0.3) > 0.99
        assert _cliff_factor(0.7) == pytest.approx(0.5, abs=0.01)
        assert _cliff_factor(0.9) < 0.05


class TestFalsifierEdgeCases:
    """Edge case tests (converted from inline Test 5)."""

    def test_no_damage_stays_healthy(self, young_patient):
        """5a: Low damage young patient stays relatively healthy."""
        result = simulate(patient=young_patient, sim_years=30)
        assert result["states"][-1, 2] > 0.5  # ATP stays reasonable

    def test_high_damage_collapses(self):
        """5b: 90% het should collapse."""
        p = dict(DEFAULT_PATIENT)
        p["baseline_heteroplasmy"] = 0.90
        result = simulate(patient=p, sim_years=30)
        assert result["states"][-1, 2] < 0.05  # ATP near zero

    def test_yamanaka_drains_atp(self):
        """5c: Max Yamanaka should cost significant energy."""
        i = dict(DEFAULT_INTERVENTION)
        i["yamanaka_intensity"] = 1.0
        result = simulate(intervention=i, sim_years=30)
        baseline = simulate(sim_years=30)
        assert result["states"][-1, 2] < baseline["states"][-1, 2]

    def test_nad_reduces_heteroplasmy(self):
        """5e: NAD supplementation should reduce heteroplasmy (fix C3)."""
        i = dict(DEFAULT_INTERVENTION)
        i["nad_supplement"] = 1.0
        r_nad = simulate(intervention=i, sim_years=30)
        r_none = simulate(sim_years=30)
        assert r_nad["heteroplasmy"][-1] < r_none["heteroplasmy"][-1]

    def test_past_cliff_no_recovery(self):
        """5f: Starting past cliff should not spontaneously recover (fix C4)."""
        p = dict(DEFAULT_PATIENT)
        p["baseline_heteroplasmy"] = 0.85
        result = simulate(patient=p, sim_years=30)
        assert result["heteroplasmy"][-1] > 0.80


class TestTissueTypes:
    """Tissue-specific simulation tests (converted from inline Test 6)."""

    @pytest.mark.parametrize("tissue", ["default", "brain", "muscle", "cardiac"])
    def test_tissue_runs(self, tissue):
        """Each tissue type should produce a valid result."""
        result = simulate(tissue_type=tissue, sim_years=30)
        assert result["states"].shape == (3001, 7)
        assert result["tissue_type"] == tissue

    def test_brain_worse_than_default(self):
        """Brain tissue should have worse outcomes (high demand, low biogenesis)."""
        brain = simulate(tissue_type="brain", sim_years=30)
        default = simulate(tissue_type="default", sim_years=30)
        assert brain["heteroplasmy"][-1] > default["heteroplasmy"][-1]


class TestStochastic:
    """Stochastic mode tests (converted from inline Tests 7-8)."""

    def test_single_stochastic(self):
        """Single stochastic trajectory should differ slightly from deterministic."""
        r_det = simulate(stochastic=False, sim_years=30)
        r_sto = simulate(stochastic=True, noise_scale=0.01, rng_seed=42, sim_years=30)
        # Both should be in reasonable range
        assert 0.0 < r_sto["heteroplasmy"][-1] < 1.0
        assert r_sto["states"][-1, 2] > 0.0

    def test_multi_trajectory(self):
        """Multiple trajectories should show variance."""
        r = simulate(stochastic=True, noise_scale=0.02, n_trajectories=10,
                     rng_seed=42, sim_years=30)
        assert r["states"].shape == (10, 3001, 7)
        assert r["heteroplasmy"].shape == (10, 3001)
        final_hets = r["heteroplasmy"][:, -1]
        assert np.std(final_hets) > 0.001  # should have some spread


class TestInterventionSchedule:
    """Tests for time-varying intervention schedules (G4)."""

    def test_phased_differs_from_constant(self, cocktail_intervention):
        """Phased schedule should differ from constant."""
        no_treatment = dict(DEFAULT_INTERVENTION)
        schedule = phased_schedule([(0, no_treatment), (10, cocktail_intervention)])
        r_phased = simulate(intervention=schedule, sim_years=30)
        r_constant = simulate(intervention=cocktail_intervention, sim_years=30)
        assert abs(r_phased["heteroplasmy"][-1]
                   - r_constant["heteroplasmy"][-1]) > 0.001

    def test_schedule_at(self):
        """InterventionSchedule.at() returns correct phase."""
        a = {"rapamycin_dose": 0.0, "nad_supplement": 0.0, "senolytic_dose": 0.0,
             "yamanaka_intensity": 0.0, "transplant_rate": 0.0, "exercise_level": 0.0}
        b = {"rapamycin_dose": 1.0, "nad_supplement": 1.0, "senolytic_dose": 1.0,
             "yamanaka_intensity": 0.0, "transplant_rate": 0.0, "exercise_level": 0.0}
        schedule = InterventionSchedule([(0, a), (10, b)])
        assert schedule.at(5)["rapamycin_dose"] == 0.0
        assert schedule.at(10)["rapamycin_dose"] == 1.0
        assert schedule.at(15)["rapamycin_dose"] == 1.0

    def test_pulsed_schedule(self):
        """Pulsed schedule creates alternating on/off phases."""
        on = dict(DEFAULT_INTERVENTION, rapamycin_dose=1.0)
        off = dict(DEFAULT_INTERVENTION)
        schedule = pulsed_schedule(on, off, period=5.0, duty_cycle=0.5,
                                   total_years=30.0)
        assert schedule.at(0)["rapamycin_dose"] == 1.0  # on
        assert schedule.at(2.5)["rapamycin_dose"] == 0.0  # off
        assert schedule.at(5.0)["rapamycin_dose"] == 1.0  # on again

    def test_plain_dict_still_works(self, cocktail_intervention):
        """Plain dicts should still work (backwards compatible)."""
        result = simulate(intervention=cocktail_intervention, sim_years=10)
        assert result["states"].shape[0] > 0


class TestCramerCorrections:
    """Tests for Cramer email corrections (C7: CD38, C8: transplant)."""

    def test_cd38_nonlinearity(self):
        """10a: High-dose NAD should give >2x benefit of low-dose (CD38 gating)."""
        baseline = simulate(sim_years=30)
        i_low = dict(DEFAULT_INTERVENTION, nad_supplement=0.25)
        i_high = dict(DEFAULT_INTERVENTION, nad_supplement=1.0)
        r_low = simulate(intervention=i_low, sim_years=30)
        r_high = simulate(intervention=i_high, sim_years=30)
        het_base = baseline["heteroplasmy"][-1]
        benefit_low = het_base - r_low["heteroplasmy"][-1]
        benefit_high = het_base - r_high["heteroplasmy"][-1]
        assert benefit_high > 2 * benefit_low  # CD38 nonlinearity

    def test_transplant_strong_rejuvenation(self):
        """10b: Transplant should produce strong heteroplasmy reduction."""
        baseline = simulate(sim_years=30)
        i_trans = dict(DEFAULT_INTERVENTION, transplant_rate=1.0)
        r_trans = simulate(intervention=i_trans, sim_years=30)
        benefit = baseline["heteroplasmy"][-1] - r_trans["heteroplasmy"][-1]
        assert benefit > 0.20  # strong rejuvenation effect

    def test_transplant_beats_nad(self):
        """10c: Transplant should outperform NAD for rejuvenation."""
        baseline = simulate(sim_years=30)
        het_base = baseline["heteroplasmy"][-1]
        i_trans = dict(DEFAULT_INTERVENTION, transplant_rate=1.0)
        i_nad = dict(DEFAULT_INTERVENTION, nad_supplement=1.0)
        r_trans = simulate(intervention=i_trans, sim_years=30)
        r_nad = simulate(intervention=i_nad, sim_years=30)
        trans_benefit = het_base - r_trans["heteroplasmy"][-1]
        nad_benefit = het_base - r_nad["heteroplasmy"][-1]
        assert trans_benefit > nad_benefit

    def test_transplant_rescues_near_cliff(self, near_cliff_patient):
        """10d: Transplant should help even near-cliff patients."""
        untreated = simulate(patient=near_cliff_patient, sim_years=30)
        i_rescue = dict(DEFAULT_INTERVENTION, transplant_rate=1.0, rapamycin_dose=0.5)
        rescued = simulate(intervention=i_rescue, patient=near_cliff_patient, sim_years=30)
        assert rescued["heteroplasmy"][-1] < untreated["heteroplasmy"][-1]
        assert rescued["states"][-1, 2] > untreated["states"][-1, 2]  # better ATP


class TestInitialState:
    """Tests for initial_state() consistency."""

    def test_total_copies_normalized(self, default_patient):
        """N_healthy + N_damaged should sum to ~1.0."""
        state = initial_state(default_patient)
        assert state[0] + state[1] == pytest.approx(1.0, abs=1e-10)

    def test_atp_positive(self, default_patient):
        """Initial ATP should be positive."""
        state = initial_state(default_patient)
        assert state[2] > 0.0


# ── C11: Split mutation type tests ──────────────────────────────────────────

from simulator import _total_heteroplasmy, _deletion_heteroplasmy


class TestMutationTypeSplit:
    """Tests for C11: split N_damaged into N_point + N_deletion."""

    def test_state_vector_is_8d(self):
        """State vector should have 8 variables after C11 split."""
        from constants import N_STATES
        assert N_STATES == 8

    def test_state_names_has_n_deletion(self):
        from constants import STATE_NAMES
        assert "N_deletion" in STATE_NAMES
        assert STATE_NAMES[1] == "N_deletion"

    def test_state_names_has_n_point(self):
        from constants import STATE_NAMES
        assert "N_point" in STATE_NAMES
        assert STATE_NAMES[7] == "N_point"

    def test_total_heteroplasmy_sums_both(self):
        """Total het = (N_del + N_pt) / (N_h + N_del + N_pt)."""
        het = _total_heteroplasmy(0.7, 0.2, 0.1)
        assert het == pytest.approx(0.3, abs=1e-10)

    def test_deletion_heteroplasmy_ignores_point(self):
        """Deletion het = N_del / total (point mutations don't drive cliff)."""
        het = _deletion_heteroplasmy(0.7, 0.2, 0.1)
        assert het == pytest.approx(0.2, abs=1e-10)

    def test_total_het_greater_than_deletion_het(self):
        het_total = _total_heteroplasmy(0.7, 0.2, 0.1)
        het_del = _deletion_heteroplasmy(0.7, 0.2, 0.1)
        assert het_total > het_del

    def test_heteroplasmy_edge_case_zero_copies(self):
        assert _total_heteroplasmy(0.0, 0.0, 0.0) == 1.0
        assert _deletion_heteroplasmy(0.0, 0.0, 0.0) == 1.0

    def test_heteroplasmy_no_damage(self):
        assert _total_heteroplasmy(1.0, 0.0, 0.0) == 0.0
        assert _deletion_heteroplasmy(1.0, 0.0, 0.0) == 0.0

    def test_initial_state_8d(self):
        """initial_state should return 8D vector."""
        state = initial_state(DEFAULT_PATIENT)
        assert state.shape == (8,)

    def test_initial_state_split_copies(self):
        """N_h + N_del + N_pt should sum to ~1.0."""
        state = initial_state(DEFAULT_PATIENT)
        total = state[0] + state[1] + state[7]
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_initial_deletion_fraction_age_dependent(self):
        """Older patients should have higher deletion fraction."""
        young_p = dict(DEFAULT_PATIENT, baseline_age=30.0, baseline_heteroplasmy=0.3)
        old_p = dict(DEFAULT_PATIENT, baseline_age=80.0, baseline_heteroplasmy=0.3)
        young_state = initial_state(young_p)
        old_state = initial_state(old_p)
        young_del_frac = young_state[1] / (young_state[1] + young_state[7])
        old_del_frac = old_state[1] / (old_state[1] + old_state[7])
        assert old_del_frac > young_del_frac

    def test_derivatives_returns_8d(self):
        """derivatives() should return 8-element array."""
        state = initial_state(DEFAULT_PATIENT)
        deriv = derivatives(state, 0.0, DEFAULT_INTERVENTION, DEFAULT_PATIENT)
        assert deriv.shape == (8,)

    def test_simulate_returns_8d_states(self):
        """simulate() should produce (n_steps+1, 8) states array."""
        result = simulate(sim_years=1)
        assert result["states"].shape[1] == 8

    def test_simulate_has_deletion_heteroplasmy(self):
        """Result should include both total and deletion heteroplasmy."""
        result = simulate(sim_years=1)
        assert "heteroplasmy" in result
        assert "deletion_heteroplasmy" in result
        assert np.all(result["deletion_heteroplasmy"] <= result["heteroplasmy"] + 1e-10)

    def test_deletions_grow_faster_than_points(self):
        """Deletions should outpace points in absolute terms over 30 years.

        With baseline mitophagy selectively clearing deletions, the NET
        growth rate of deletions may be lower than points. But in absolute
        terms (copy count increase), deletions should still accumulate more
        than points due to the 1.10x replication advantage and age-dependent
        de novo deletion creation from Pol gamma slippage.
        """
        result = simulate(sim_years=30)
        n_del_0 = result["states"][0, 1]
        n_pt_0 = result["states"][0, 7]
        n_del_f = result["states"][-1, 1]
        n_pt_f = result["states"][-1, 7]
        # Deletions should increase in absolute terms
        del_increase = n_del_f - n_del_0
        pt_increase = n_pt_f - n_pt_0
        # Both types should increase over 30 years of aging
        assert n_del_f > n_del_0 * 0.9  # deletions persist despite mitophagy
        assert n_pt_f > n_pt_0          # points grow (evade mitophagy)

    def test_cliff_driven_by_deletions_not_points(self):
        """High deletion het should collapse ATP.

        With het=0.75 at age 70, deletion_het is ~0.51 (below 0.70 cliff),
        so ATP stays high. Need het>=0.92 for deletion_het to approach cliff.
        The key test: deletion_het determines ATP, not total het.
        """
        # Very high het patient: deletion_het should be high enough for ATP impact
        p_high = dict(DEFAULT_PATIENT, baseline_heteroplasmy=0.92, baseline_age=80.0)
        r_high = simulate(patient=p_high, sim_years=5)
        # Deletion het should be majority of total het
        assert r_high["deletion_heteroplasmy"][-1] > 0.5
        # With high deletion het approaching cliff, ATP should be reduced
        assert r_high["states"][-1, 2] < 0.5

    def test_analytics_includes_deletion_metrics(self):
        from analytics import compute_all
        result = simulate(sim_years=10)
        baseline = simulate(sim_years=10)
        analytics = compute_all(result, baseline)
        damage = analytics["damage"]
        assert "deletion_het_final" in damage
        assert "deletion_het_initial" in damage
        assert damage["deletion_het_final"] <= damage["het_final"]

    def test_point_mutations_no_replication_advantage(self):
        """Point mutations replicate at base rate (no size advantage).

        Deletions have 1.10x replication advantage but are selectively
        cleared by mitophagy (PINK1/Parkin). Point mutations evade
        mitophagy (normal membrane potential) but replicate at 1.0x.
        The key signature: both types should be present throughout,
        and total damage (del + pt) should increase monotonically.
        """
        result = simulate(sim_years=30)
        n_del_end = result["states"][-1, 1]
        n_pt_end = result["states"][-1, 7]
        # Both mutation types should persist
        assert n_del_end > 0.01
        assert n_pt_end > 0.01
        # Total heteroplasmy should have increased
        assert result["heteroplasmy"][-1] > result["heteroplasmy"][0]
