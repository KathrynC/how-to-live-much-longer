"""Downstream chain -- neuroplasticity and Alzheimer's pathology.

Integrates 6 ODEs (MEF2, HA, SS, CR, amyloid, tau) as post-processing
on the Cramer core ODE output. One-way dependency: reads ATP, ROS,
senescence from core trajectory, never feeds back.

ODE equations from handoff_batch3_unified_spec_2026-02-19.md:

dMEF2/dt = engagement * MEF2_INDUCTION_RATE * apoe_mult * (1-MEF2)
           - MEF2 * MEF2_DECAY_RATE * (1 - engagement*0.5)
dHA/dt   = MEF2 * HA_INDUCTION_RATE * (1-HA) - HA * HA_DECAY_RATE
dSS/dt   = LEARNING_RATE_BASE * engagement * plasticity * apoe_synaptic_mult
           * (1 - SS/MAX_SYNAPTIC_STRENGTH)
           - SYNAPTIC_DECAY_RATE * (SS - 1)
dCR/dt   = engagement * growth_rate * (1-CR) + education_boost * 0.05
dAb/dt   = (AMYLOID_PRODUCTION_BASE
           + AMYLOID_PRODUCTION_AGE_FACTOR*(age-63)
           - AMYLOID_CLEARANCE_BASE*apoe_clearance*Ab)
           * (1 + inflammation*AMYLOID_INFLAMMATION_SYNERGY)
dtau/dt  = TAU_SEEDING_RATE * Ab * TAU_SEEDING_FACTOR * apoe_tau_mult
           + inflammation * TAU_INFLAMMATION_FACTOR * apoe_tau_mult
           - TAU_CLEARANCE_BASE * tau

memory_index = BASELINE_MEMORY + (SS-1)*SYNAPSES_TO_MEMORY
               + MEF2*MEF2_MEMORY_BOOST + CR*0.2 - effective_pathology
effective_pathology = (Ab*AMYLOID_TOXICITY + tau*TAU_TOXICITY)
                      * (1 - resilience)
resilience = min(1.0, MEF2*0.3 + max(0, SS-1)*0.3 + CR*0.4)
"""
from __future__ import annotations

import numpy as np

from constants import (
    MEF2_INDUCTION_RATE, MEF2_DECAY_RATE, MEF2_MEMORY_BOOST,
    HA_INDUCTION_RATE, HA_DECAY_RATE,
    PLASTICITY_FACTOR_BASE, PLASTICITY_FACTOR_HA_MAX,
    LEARNING_RATE_BASE, SYNAPTIC_DECAY_RATE, MAX_SYNAPTIC_STRENGTH,
    SYNAPSES_TO_MEMORY, BASELINE_MEMORY,
    CR_GROWTH_RATE_BY_ACTIVITY,
    AMYLOID_PRODUCTION_BASE, AMYLOID_PRODUCTION_AGE_FACTOR,
    AMYLOID_CLEARANCE_BASE, AMYLOID_INFLAMMATION_SYNERGY,
    TAU_SEEDING_RATE, TAU_SEEDING_FACTOR, TAU_INFLAMMATION_FACTOR,
    TAU_CLEARANCE_BASE,
    AMYLOID_TOXICITY, TAU_TOXICITY,
    RESILIENCE_WEIGHTS,
)


# ── Education baseline mapping for cognitive reserve ────────────────────────

EDUCATION_BASELINE = {
    'high_school': 0.3,
    'bachelors': 0.4,
    'masters': 0.5,
    'doctoral': 0.6,
}


# ── Individual derivative functions ─────────────────────────────────────────

def mef2_derivative(
    mef2: float,
    engagement: float,
    apoe_mult: float,
) -> float:
    """MEF2 transcription factor activity derivative.

    MEF2 is induced by intellectual engagement (modulated by APOE genotype)
    and decays when engagement is absent. Saturates toward 1.0.

    Args:
        mef2: Current MEF2 activity level (0..1).
        engagement: Intellectual engagement level (0..1).
        apoe_mult: APOE genotype multiplier for MEF2 induction (>=1.0 for
            APOE4 carriers, reflecting increased need for MEF2 activity).

    Returns:
        dMEF2/dt.
    """
    induction = engagement * MEF2_INDUCTION_RATE * apoe_mult * (1.0 - mef2)
    decay = mef2 * MEF2_DECAY_RATE * (1.0 - engagement * 0.5)
    return induction - decay


def ha_derivative(
    ha: float,
    mef2: float,
) -> float:
    """Histone acetylation derivative.

    MEF2 induces histone acetylation (epigenetic memory of learning).
    Acetylation decays slowly in the absence of MEF2 activity.

    Args:
        ha: Current histone acetylation level (0..1).
        mef2: Current MEF2 activity level (0..1).

    Returns:
        dHA/dt.
    """
    induction = mef2 * HA_INDUCTION_RATE * (1.0 - ha)
    decay = ha * HA_DECAY_RATE
    return induction - decay


def synaptic_derivative(
    ss: float,
    ha: float,
    engagement: float,
    apoe_synaptic_mult: float = 1.0,
) -> float:
    """Synaptic strength derivative.

    Engagement + histone-acetylation-modulated plasticity strengthens
    synapses. Without engagement, synaptic strength decays toward
    baseline (1.0). APOE4 carriers have reduced synaptic maintenance
    efficiency (fewer dendritic spines, Dumanis et al. 2010).

    Args:
        ss: Current synaptic strength (baseline 1.0, max MAX_SYNAPTIC_STRENGTH).
        ha: Current histone acetylation level (0..1).
        engagement: Intellectual engagement level (0..1).
        apoe_synaptic_mult: APOE genotype multiplier for synaptic growth
            (<=1.0 for APOE4 carriers). Default 1.0 (non-carrier).

    Returns:
        dSS/dt.
    """
    plasticity = PLASTICITY_FACTOR_BASE + ha * (PLASTICITY_FACTOR_HA_MAX - PLASTICITY_FACTOR_BASE)
    growth = LEARNING_RATE_BASE * engagement * plasticity * apoe_synaptic_mult * (1.0 - ss / MAX_SYNAPTIC_STRENGTH)
    decay = SYNAPTIC_DECAY_RATE * (ss - 1.0)
    return growth - decay


def cr_derivative(
    cr: float,
    engagement: float,
    growth_rate: float,
    education_boost: float,
) -> float:
    """Cognitive reserve derivative.

    Engagement grows cognitive reserve; education provides a small
    ongoing boost. CR saturates toward 1.0.

    Args:
        cr: Current cognitive reserve (0..1).
        engagement: Intellectual engagement level (0..1).
        growth_rate: Activity-type-dependent growth rate.
        education_boost: Education-level-dependent boost (0 or 1).

    Returns:
        dCR/dt.
    """
    growth = engagement * growth_rate * (1.0 - cr)
    edu = education_boost * 0.05
    return growth + edu


def amyloid_derivative(
    amyloid: float,
    inflammation: float,
    age: float,
    apoe_clearance: float,
) -> float:
    """Amyloid-beta burden derivative.

    Amyloid accumulates with age and inflammation, and is cleared by
    glymphatic/microglial pathways (impaired by APOE4).

    Args:
        amyloid: Current amyloid-beta burden (arbitrary units, >=0).
        inflammation: Current inflammation level (0..1).
        age: Current age of the patient (years).
        apoe_clearance: APOE-dependent clearance multiplier (1.0 normal,
            0.7 for APOE4 het, 0.5 for APOE4 hom).

    Returns:
        dAb/dt.
    """
    production = AMYLOID_PRODUCTION_BASE + AMYLOID_PRODUCTION_AGE_FACTOR * (age - 63.0)
    clearance = AMYLOID_CLEARANCE_BASE * apoe_clearance * amyloid
    net = production - clearance
    inflammation_synergy = 1.0 + inflammation * AMYLOID_INFLAMMATION_SYNERGY
    return net * inflammation_synergy


def tau_derivative(
    tau: float,
    amyloid: float,
    inflammation: float,
    apoe_tau_mult: float = 1.0,
) -> float:
    """Tau pathology derivative.

    Tau is seeded by amyloid-beta and promoted by inflammation,
    cleared at a baseline rate. APOE4 exacerbates tau pathology
    independently of its amyloid effects (Shi et al. 2017, Nature;
    Therriault et al. 2020, JAMA Neurology).

    Args:
        tau: Current tau pathology burden (arbitrary units, >=0).
        amyloid: Current amyloid-beta burden.
        inflammation: Current inflammation level (0..1).
        apoe_tau_mult: APOE genotype multiplier for tau accumulation
            (>=1.0 for APOE4 carriers). Default 1.0 (non-carrier).

    Returns:
        dtau/dt.
    """
    seeding = TAU_SEEDING_RATE * amyloid * TAU_SEEDING_FACTOR * apoe_tau_mult
    infl = inflammation * TAU_INFLAMMATION_FACTOR * apoe_tau_mult
    clearance = TAU_CLEARANCE_BASE * tau
    return seeding + infl - clearance


def resilience(
    mef2: float,
    ss: float,
    cr: float,
) -> float:
    """Compute cognitive resilience from neuroprotective factors.

    Resilience buffers against amyloid/tau pathology. Composed of
    MEF2 activity, synaptic gain (above baseline), and cognitive reserve.

    Args:
        mef2: MEF2 activity (0..1).
        ss: Synaptic strength (baseline 1.0).
        cr: Cognitive reserve (0..1).

    Returns:
        Resilience score (0..1).
    """
    w = RESILIENCE_WEIGHTS
    raw = mef2 * w['MEF2'] + max(0.0, ss - 1.0) * w['synaptic_gain'] + cr * w['CR']
    return min(1.0, raw)


def memory_index(
    ss: float,
    mef2: float,
    cr: float,
    amyloid: float,
    tau: float,
) -> float:
    """Compute composite memory index.

    Positive contributors: synaptic strength above baseline, MEF2 activity,
    cognitive reserve. Negative contributor: effective pathology (amyloid +
    tau, buffered by resilience).

    Args:
        ss: Synaptic strength (baseline 1.0).
        mef2: MEF2 activity (0..1).
        cr: Cognitive reserve (0..1).
        amyloid: Amyloid-beta burden.
        tau: Tau pathology burden.

    Returns:
        Memory index (can go below 0 in extreme pathology).
    """
    res = resilience(mef2, ss, cr)
    effective_pathology = (amyloid * AMYLOID_TOXICITY + tau * TAU_TOXICITY) * (1.0 - res)
    mi = (
        BASELINE_MEMORY
        + (ss - 1.0) * SYNAPSES_TO_MEMORY
        + mef2 * MEF2_MEMORY_BOOST
        + cr * 0.2
        - effective_pathology
    )
    return mi


# ── Main entry point ────────────────────────────────────────────────────────

def compute_downstream(
    core_result: dict,
    patient_expanded: dict,
) -> list[dict]:
    """Run the downstream chain on a core simulation result.

    Integrates 6 ODEs (MEF2, HA, SS, CR, amyloid, tau) using Euler
    integration at the same timestep as the core simulation. Reads ATP,
    ROS, and senescence from the core trajectory as driving inputs.

    Args:
        core_result: Dict returned by simulator.simulate(), containing:
            - 'time': np.array of time points
            - 'states': np.array shape (n_steps+1, 8)
            - 'patient': the patient dict used
        patient_expanded: Dict with expanded patient parameters, must include:
            - 'intellectual_engagement': float (0..1)
            - 'baseline_age': float (years)
            Optional:
            - 'apoe_genotype': str (e.g. 'apoe4_het')
            - 'education_level': str (e.g. 'bachelors', 'doctoral')
            - 'activity_type': str (key into CR_GROWTH_RATE_BY_ACTIVITY)

    Returns:
        List of dicts (one per timestep), each containing:
            MEF2_activity, histone_acetylation, synaptic_strength,
            cognitive_reserve, amyloid_burden, tau_pathology,
            memory_index, resilience
    """
    # ── Extract core trajectory ─────────────────────────────────────────
    time = core_result['time']
    states = core_result['states']
    n_steps = len(time)

    # Core state indices: ATP=2, ROS=3, Senescent_fraction=5
    atp_trace = states[:, 2]
    ros_trace = states[:, 3]
    sen_trace = states[:, 5]

    # ── Extract patient parameters ──────────────────────────────────────
    engagement = patient_expanded.get('intellectual_engagement', 0.5)
    baseline_age = patient_expanded.get('baseline_age', 70.0)
    apoe_genotype = patient_expanded.get('apoe_genotype', None)
    education_level = patient_expanded.get('education_level', 'bachelors')
    activity_type = patient_expanded.get('activity_type', 'solitary_routine')

    # APOE multipliers
    from constants import GENOTYPE_MULTIPLIERS
    if apoe_genotype and apoe_genotype in GENOTYPE_MULTIPLIERS:
        geno = GENOTYPE_MULTIPLIERS[apoe_genotype]
        apoe_mult = geno.get('mef2_induction', 1.0)
        apoe_clearance = geno.get('amyloid_clearance', 1.0)
        apoe_tau_mult = geno.get('tau_pathology_sensitivity', 1.0)
        apoe_synaptic_mult = geno.get('synaptic_function', 1.0)
    else:
        apoe_mult = 1.0
        apoe_clearance = 1.0
        apoe_tau_mult = 1.0
        apoe_synaptic_mult = 1.0

    # Cognitive reserve growth rate from activity type
    growth_rate = CR_GROWTH_RATE_BY_ACTIVITY.get(activity_type, 0.03)

    # Education baseline
    education_baseline = EDUCATION_BASELINE.get(education_level, 0.4)

    # Education boost: small ongoing term (1.0 if educated, 0.0 otherwise)
    # We use a continuous value: higher education = higher boost
    education_boost = 1.0 if education_baseline >= 0.4 else 0.0

    # ── Compute dt from core time array ─────────────────────────────────
    if n_steps > 1:
        dt = time[1] - time[0]
    else:
        dt = 0.01  # fallback

    # ── Initialize downstream state ─────────────────────────────────────
    mef2_val = 0.2
    ha_val = 0.2
    ss_val = 1.0
    cr_val = education_baseline
    # Age-dependent amyloid baseline: zero before 40, accumulates after
    amyloid_val = max(0.0, 0.02 * (baseline_age - 40.0))
    tau_val = 0.0

    # ── Euler integration ───────────────────────────────────────────────
    output = []

    for i in range(n_steps):
        # Current age at this timestep
        current_age = baseline_age + time[i]

        # Estimate inflammation from ROS and senescence (from core)
        inflammation = min(1.0, 0.3 * ros_trace[i] + 0.5 * sen_trace[i])

        # Compute current resilience and memory index
        res = resilience(mef2_val, ss_val, cr_val)
        mi = memory_index(ss_val, mef2_val, cr_val, amyloid_val, tau_val)

        # Record current state
        output.append({
            'MEF2_activity': mef2_val,
            'histone_acetylation': ha_val,
            'synaptic_strength': ss_val,
            'cognitive_reserve': cr_val,
            'amyloid_burden': amyloid_val,
            'tau_pathology': tau_val,
            'memory_index': mi,
            'resilience': res,
        })

        # If this is the last point, no integration step needed
        if i == n_steps - 1:
            break

        # ── Compute derivatives ─────────────────────────────────────────
        d_mef2 = mef2_derivative(mef2_val, engagement, apoe_mult)
        d_ha = ha_derivative(ha_val, mef2_val)
        d_ss = synaptic_derivative(ss_val, ha_val, engagement, apoe_synaptic_mult)
        d_cr = cr_derivative(cr_val, engagement, growth_rate, education_boost)
        d_amyloid = amyloid_derivative(amyloid_val, inflammation, current_age, apoe_clearance)
        d_tau = tau_derivative(tau_val, amyloid_val, inflammation, apoe_tau_mult)

        # ── Euler step ──────────────────────────────────────────────────
        mef2_val += d_mef2 * dt
        ha_val += d_ha * dt
        ss_val += d_ss * dt
        cr_val += d_cr * dt
        amyloid_val += d_amyloid * dt
        tau_val += d_tau * dt

        # ── Clamp all variables ─────────────────────────────────────────
        mef2_val = max(0.0, min(1.0, mef2_val))
        ha_val = max(0.0, min(1.0, ha_val))
        ss_val = max(0.0, min(MAX_SYNAPTIC_STRENGTH, ss_val))
        cr_val = max(0.0, min(1.0, cr_val))
        amyloid_val = max(0.0, amyloid_val)
        tau_val = max(0.0, tau_val)

    return output
