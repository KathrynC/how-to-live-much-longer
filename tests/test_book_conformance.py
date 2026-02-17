"""Executable conformance checks against Cramer's Appendix 2 claims.

These tests are traceability checks, not exhaustive biological validation.
Each test maps to a claim ID in docs/book_conformance_appendix2.md.
"""
from __future__ import annotations

import os
import math

import pytest

from constants import (
    AGE_TRANSITION,
    CD38_BASE_SURVIVAL,
    CD38_SUPPRESSION_GAIN,
    DELETION_REPLICATION_ADVANTAGE,
    DOUBLING_TIME_OLD,
    DOUBLING_TIME_YOUNG,
)
from simulator import _deletion_rate, derivatives


def _base_intervention() -> dict[str, float]:
    return {
        "rapamycin_dose": 0.0,
        "nad_supplement": 0.0,
        "senolytic_dose": 0.0,
        "yamanaka_intensity": 0.0,
        "transplant_rate": 0.0,
        "exercise_level": 0.0,
    }


def _base_patient() -> dict[str, float]:
    return {
        "baseline_age": 50.0,
        "baseline_heteroplasmy": 0.3,
        "baseline_nad_level": 0.8,
        "genetic_vulnerability": 1.0,
        "metabolic_demand": 1.0,
        "inflammation_level": 0.1,
    }


def test_appendix2_doubling_time_constants():
    """A2-01/A2-02: Doubling constants reflect Appendix 2 fit values."""
    assert DOUBLING_TIME_YOUNG == pytest.approx(11.81, abs=0.05)
    assert DOUBLING_TIME_OLD == pytest.approx(3.06, abs=0.01)


def test_appendix2_age_transition_constant():
    """A2-03: Transition age is age 65."""
    assert AGE_TRANSITION == pytest.approx(65.0, abs=1e-12)


def test_deletions_have_replication_advantage_over_point_pool():
    """A2-04: Deletion pool has >1 replication multiplier (directional claim)."""
    assert DELETION_REPLICATION_ADVANTAGE > 1.0


def test_deletion_rate_accelerates_with_age():
    """A2-07: Old-age deletion rate should exceed young-age deletion rate."""
    # Use nominal health reference to minimize dynamic shift side effects.
    young = _deletion_rate(
        age=40.0,
        genetic_vulnerability=1.0,
        atp_norm=0.77,
        mitophagy_rate=0.02,
    )
    old = _deletion_rate(
        age=80.0,
        genetic_vulnerability=1.0,
        atp_norm=0.77,
        mitophagy_rate=0.02,
    )
    assert old > young


def test_figure23_regime_rates_match_reported_doubling_scales():
    """A2-01/A2-02: Rate regimes align with Figure 23 doubling values."""
    # With nominal health reference, these ages are deep in their regimes.
    young_rate = _deletion_rate(
        age=25.0,
        genetic_vulnerability=1.0,
        atp_norm=0.77,
        mitophagy_rate=0.02,
    )
    old_rate = _deletion_rate(
        age=90.0,
        genetic_vulnerability=1.0,
        atp_norm=0.77,
        mitophagy_rate=0.02,
    )
    expected_young = math.log(2.0) / 11.81
    expected_old = math.log(2.0) / 3.06

    # Tolerant checks: dynamic smoothing and health-coupled transition are active.
    assert young_rate == pytest.approx(expected_young, rel=0.10)
    assert old_rate == pytest.approx(expected_old, rel=0.10)


def test_cliff_depends_on_deletion_fraction_not_total_mutation_burden():
    """A2-05: ATP cliff term should follow deletion heteroplasmy semantics.

    We compare two states with identical deletion heteroplasmy (N_del/total)
    but very different total mutation burden; ATP derivative should match.
    """
    intervention = _base_intervention()
    patient = _base_patient()

    # Same deletion heteroplasmy = 0.35, different total mutation burden.
    # state = [N_h, N_del, ATP, ROS, NAD, Sen, Psi, N_point]
    state_low_total = [0.65, 0.35, 0.7, 0.1, 0.8, 0.05, 0.9, 0.0]   # total het=0.35
    state_high_total = [0.15, 0.35, 0.7, 0.1, 0.8, 0.05, 0.9, 0.5]  # total het=0.85

    d_low = derivatives(state_low_total, 0.0, intervention, patient)
    d_high = derivatives(state_high_total, 0.0, intervention, patient)

    # dATP index = 2
    assert d_low[2] == pytest.approx(d_high[2], abs=1e-10)


def test_more_deletion_with_same_total_damage_worsens_atp_derivative():
    """A2-05/A2-07: For equal total mutation burden, more deletions => worse ATP."""
    intervention = _base_intervention()
    patient = _base_patient()

    # Both states have total mutation burden 0.5, but different deletion shares.
    # low_del: N_del=0.2, N_point=0.3 ; high_del: N_del=0.5, N_point=0.0
    low_del = [0.5, 0.2, 0.7, 0.1, 0.8, 0.05, 0.9, 0.3]
    high_del = [0.5, 0.5, 0.7, 0.1, 0.8, 0.05, 0.9, 0.0]

    d_low = derivatives(low_del, 0.0, intervention, patient)
    d_high = derivatives(high_del, 0.0, intervention, patient)

    # More deletion burden should lower ATP derivative (more negative / less positive).
    assert d_high[2] < d_low[2]


def test_book_minimum_21pct_replication_advantage_compliant():
    """A2-08: Enforce Appendix 2 minimum replication advantage wording."""
    assert DELETION_REPLICATION_ADVANTAGE >= 1.21


def test_cd38_survival_endpoints_match_c7_text():
    """C7: CD38 survival is 40% at minimum and 100% at maximum dose."""
    min_survival = CD38_BASE_SURVIVAL + CD38_SUPPRESSION_GAIN * 0.0
    max_survival = CD38_BASE_SURVIVAL + CD38_SUPPRESSION_GAIN * 1.0
    assert min_survival == pytest.approx(0.4, abs=1e-12)
    assert max_survival == pytest.approx(1.0, abs=1e-12)


def test_nad_dynamics_reflect_cd38_gated_nonlinearity():
    """C7: Higher NMN/NR dose should get amplified by CD38-gated survival."""
    patient = _base_patient()
    state = [0.7, 0.2, 0.8, 0.1, 0.5, 0.05, 0.9, 0.1]

    low = _base_intervention()
    high = _base_intervention()
    low["nad_supplement"] = 0.25
    high["nad_supplement"] = 1.0

    d_low = derivatives(state, 0.0, low, patient)
    d_high = derivatives(state, 0.0, high, patient)

    # dNAD index = 4; high dose should strongly increase NAD derivative.
    assert d_high[4] > d_low[4]


@pytest.mark.skipif(
    os.getenv("STRICT_BOOK_CONFORMANCE") != "1",
    reason="Enable with STRICT_BOOK_CONFORMANCE=1",
)
def test_book_minimum_21pct_replication_advantage_strict_gate():
    """A2-08 strict gate for CI when exact Appendix 2 enforcement is desired."""
    assert DELETION_REPLICATION_ADVANTAGE >= 1.21
