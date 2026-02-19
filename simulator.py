"""Numpy-only RK4 ODE integrator for mitochondrial aging dynamics.

Simulates 8 coupled state variables over a configurable time horizon
(default 30 years, dt=0.01 yr ~ 3.65 days) using 4th-order Runge-Kutta.
Models the core biology from Cramer (2025): the ROS-damage vicious cycle,
the heteroplasmy cliff, age-dependent deletion doubling, and six
pharmacological/lifestyle intervention mechanisms.

C11 split (Cramer email 2026-02-17): The original 7-variable model had a
single "N_damaged" pool. C11 splits this into two biologically distinct
mutation types:

    N_deletion (index 1) — Large-scale deletions (>3 kbp removed from the
        16.5 kbp mtDNA ring). These shorter molecules replicate ~10-21%
        faster than wild-type (Vandiver et al. 2023), creating exponential
        clonal expansion. Because large deletions knock out multiple ETC
        genes, they are the PRIMARY driver of the heteroplasmy cliff: when
        deletion heteroplasmy exceeds ~70%, respiratory chain complexes
        can no longer assemble, and ATP collapses nonlinearly.

    N_point (index 7) — Single-nucleotide substitutions from Pol-gamma
        copying errors and ROS-induced oxidative damage (8-oxoguanine).
        These replicate at the SAME rate as wild-type (no size advantage)
        and are often functionally silent or mildly deleterious. They
        contribute to overall heteroplasmy and ROS production but do NOT
        drive the cliff.

This split is critical because it separates two fundamentally different
dynamics: exponential clonal expansion (deletions) vs. linear drift
(point mutations). The cliff — the central catastrophe of mitochondrial
aging — is caused specifically by deletions, not by the total burden of
all mutations.

Reference:
    Cramer, J.G. (forthcoming from Springer Verlag in 2026).
    *How to Live Much Longer: The Mitochondrial DNA Connection*.

Key biological mechanisms and their book sources:
    - ROS-damage vicious cycle: Ch. II.H pp.14-15, Appendix 2 pp.152-154
    - Heteroplasmy cliff: mitochondrial genetics literature (Rossignol 2003)
    - Deletion doubling times: Appendix 2 p.155, Fig. 23 (Va23 data)
    - Replication advantage: Appendix 2 pp.154-155 ("at least 21% faster")
    - Rapamycin → mTOR → mitophagy: Ch. VI.A.1 pp.71-72
    - NAD+ supplementation (NMN/NR): Ch. VI.A.3 pp.72-73 (Ca16)
    - Mitophagy (PINK1/Parkin): Ch. VI.B p.75
    - Senolytics (D+Q+F): Ch. VII.A.2 p.91
    - Yamanaka reprogramming: Ch. VII.B pp.92-95, energy cost Ch. VIII.A
      Table 3 p.100 (~3-5 MU, citing Ci24, Fo18)
    - Senescent cells: Ch. VII.A pp.89-90, Ch. VIII.F p.103 (~2x energy)
    - Membrane potential (ΔΨ): Ch. IV pp.46-47, Ch. VI.B p.75
    - Mitochondrial transplant: Ch. VIII.G pp.104-107 (mitlets)
    - ATP = 1 MU/day baseline: Ch. VIII.A Table 3 p.100

Corrections applied (chronological):
    C1  — Cliff feeds back into replication and apoptosis (2026-02-15)
    C2  — Copy number homeostasis: total N targets 1.0 (2026-02-15)
    C3  — NAD selectively benefits healthy mitos (2026-02-15)
    C4  — Bistability past cliff: damaged replication advantage (2026-02-15)
    C7  — CD38 degrades NMN/NR; survival gated by dose (Cramer email)
    C8  — Transplant is primary rejuvenation; doubled rate (Cramer email)
    C9  — AGE_TRANSITION restored to 65 (Cramer email)
    C10 — Deletion rate coupled to ATP + mitophagy (Cramer email)
    C11 — N_damaged split into N_deletion + N_point (Cramer email 2026-02-17)
    M1  — Yamanaka gated by ATP (2026-02-15)
    M5  — Exercise biogenesis gated by energy + copy pressure (2026-02-15)

State variables (8D, C11 layout):
    Index 0: N_healthy          — healthy (wild-type) mtDNA copies (normalized, 1.0 = full)
    Index 1: N_deletion         — deletion-mutated mtDNA (exponential growth, drives cliff)
    Index 2: ATP                — ATP production rate (MU/day, 1 MU = 10^8 ATP releases)
    Index 3: ROS                — reactive oxygen species level (normalized)
    Index 4: NAD                — NAD+ cofactor availability (normalized, 1.0 = young baseline)
    Index 5: Senescent_fraction — fraction of cells in irreversible growth arrest
    Index 6: Membrane_potential — mitochondrial inner membrane ΔΨ (normalized, 1.0 = ~180 mV)
    Index 7: N_point            — point-mutated mtDNA (linear growth, functionally mild)

Usage:
    from simulator import simulate
    result = simulate(intervention, patient)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from constants import (
    SIM_YEARS, DT, N_STATES,
    HETEROPLASMY_CLIFF, CLIFF_STEEPNESS,
    DOUBLING_TIME_YOUNG, DOUBLING_TIME_OLD, AGE_TRANSITION,
    BASELINE_ATP, BASELINE_ROS, ROS_PER_DAMAGED,
    BASELINE_NAD, NAD_DECLINE_RATE,
    BASELINE_MEMBRANE_POTENTIAL, BASELINE_SENESCENT, SENESCENCE_RATE,
    BASELINE_MITOPHAGY_RATE,
    CD38_BASE_SURVIVAL, CD38_SUPPRESSION_GAIN,
    TRANSPLANT_ADDITION_RATE, TRANSPLANT_DISPLACEMENT_RATE, TRANSPLANT_HEADROOM,
    DEFAULT_INTERVENTION, DEFAULT_PATIENT,
    TISSUE_PROFILES,
    # C11: Split mutation type constants
    POINT_ERROR_RATE, ROS_POINT_COEFF, POINT_MITOPHAGY_SELECTIVITY,
    DELETION_REPLICATION_ADVANTAGE,
    DELETION_FRACTION_YOUNG, DELETION_FRACTION_OLD,
)

# Default tissue modifiers: no tissue-specific scaling applied.
# When tissue_type is None, ROS sensitivity and biogenesis rate are 1.0x.
_DEFAULT_TISSUE_MODS = {"ros_sensitivity": 1.0, "biogenesis_rate": 1.0}


# ── Time-varying intervention schedules ─────────────────────────────────────

class InterventionSchedule:
    """Allow interventions to change over the simulation horizon.

    Holds a sorted list of (start_year, intervention_dict) phases.
    At any time t, returns the intervention dict of the most recent phase.
    This enables modeling realistic clinical protocols where treatment
    starts, stops, or changes intensity over a patient's lifetime.

    Usage:
        schedule = InterventionSchedule([
            (0, no_treatment),       # years 0-10: observation only
            (10, full_cocktail),     # years 10-30: aggressive treatment
        ])
        schedule.at(5)   # -> no_treatment
        schedule.at(15)  # -> full_cocktail
    """

    def __init__(self, phases: list[tuple[float, dict[str, float]]]) -> None:
        # Sort phases by start time so .at() can iterate chronologically
        self.phases = sorted(phases, key=lambda x: x[0])

    def at(self, t: float) -> dict[str, float]:
        """Return the intervention dict active at time t.

        Walks through phases in order; the last phase whose start_year <= t
        is the active one. This is a piecewise-constant (step function)
        interpolation — no blending between phases.
        """
        active = self.phases[0][1]
        for start, intervention in self.phases:
            if t >= start:
                active = intervention
            else:
                break
        return active


def phased_schedule(
    phases: list[tuple[float, dict[str, float]]],
) -> InterventionSchedule:
    """Convenience constructor for multi-phase intervention schedules.

    Example: no treatment for 10 years, then cocktail for 20 years:
        schedule = phased_schedule([(0, no_treatment), (10, cocktail)])

    Args:
        phases: List of (start_year, intervention_dict) tuples.

    Returns:
        InterventionSchedule instance.
    """
    return InterventionSchedule(phases)


def pulsed_schedule(
    on_intervention: dict[str, float],
    off_intervention: dict[str, float],
    period: float,
    duty_cycle: float = 0.5,
    total_years: float = 30.0,
) -> InterventionSchedule:
    """Create a pulsed on/off intervention schedule.

    Models intermittent dosing protocols (e.g., 5 years on rapamycin,
    5 years off) which may reduce side effects while maintaining benefit.

    Args:
        on_intervention: Intervention dict during "on" phases.
        off_intervention: Intervention dict during "off" phases.
        period: Cycle length in years (e.g., 10.0 for 10-year cycles).
        duty_cycle: Fraction of each period that is "on" (0.0-1.0).
        total_years: Total schedule duration.

    Returns:
        InterventionSchedule with alternating on/off phases.
    """
    phases: list[tuple[float, dict[str, float]]] = []
    t = 0.0
    while t < total_years:
        phases.append((t, on_intervention))
        t_off = t + period * duty_cycle
        if t_off < total_years:
            phases.append((t_off, off_intervention))
        t += period
    return InterventionSchedule(phases)


def _resolve_intervention(
    intervention: dict[str, float] | InterventionSchedule,
    t: float,
) -> dict[str, float]:
    """Resolve intervention at time t (handles both dict and schedule).

    If the intervention is a plain dict (constant protocol), returns it
    directly. If it is an InterventionSchedule (time-varying), calls .at(t)
    to get the phase-appropriate intervention dict.
    """
    if isinstance(intervention, InterventionSchedule):
        return intervention.at(t)
    return intervention


# ── Heteroplasmy computation functions ──────────────────────────────────────
#
# Why two heteroplasmy functions exist (C11):
#
# Before C11, there was a single N_damaged pool, and heteroplasmy was simply
# N_damaged / (N_healthy + N_damaged). After the C11 split, we need TWO
# different heteroplasmy measures for different purposes:
#
# 1. TOTAL heteroplasmy = (N_del + N_pt) / total
#    This is the overall fraction of mutated mtDNA. It determines:
#    - How much extra ROS the cell produces (all mutants leak electrons)
#    - The "headline number" reported in clinical mtDNA sequencing
#    - Total mutational burden for epidemiological comparison
#
# 2. DELETION heteroplasmy = N_del / total
#    This is the fraction of deletion-mutated mtDNA specifically. It determines:
#    - Whether the cell crosses the heteroplasmy cliff (only deletions matter
#      because they knock out ETC subunit genes; point mutations are usually
#      in non-coding regions or cause conservative amino acid changes)
#    - The cliff factor (sigmoid collapse of ATP production)
#    - The bistability threshold (Cramer's "point of no return")
#
# The biological rationale: a cell can have 80% point mutations and still
# function reasonably well (mild ETC dysfunction, some extra ROS). But a
# cell with 70% deletions is in catastrophic failure because the ETC cannot
# assemble at all. This distinction is the core insight of C11.

def _heteroplasmy_fraction(n_healthy: float, n_damaged: float) -> float:
    """Compute heteroplasmy as fraction of damaged copies.

    DEPRECATED (C11): Use _total_heteroplasmy() or _deletion_heteroplasmy()
    instead. Kept for backward compatibility with existing code that uses
    the 7-variable state vector (index 1 = N_damaged).

    Returns 1.0 if total copy number is near zero (complete depletion
    is treated as fully damaged for safety).
    """
    total = n_healthy + n_damaged
    if total < 1e-12:
        return 1.0  # If essentially no mtDNA remains, treat as fully damaged
    return n_damaged / total


def _total_heteroplasmy(n_healthy: float, n_deletion: float, n_point: float) -> float:
    """Total heteroplasmy: (N_del + N_pt) / total.

    This is the clinically reportable heteroplasmy — the fraction of ALL
    mtDNA copies that carry ANY mutation (deletion or point). Used for:
      - ROS production scaling (both mutation types cause electron leakage)
      - Output reporting (what a clinician would measure via sequencing)
      - Epidemiological comparison with published heteroplasmy data

    Returns 1.0 if total copy number is near zero (depletion guard).
    """
    total = n_healthy + n_deletion + n_point
    if total < 1e-12:
        return 1.0  # Complete mtDNA depletion: treat as fully mutated
    return (n_deletion + n_point) / total


def _deletion_heteroplasmy(n_healthy: float, n_deletion: float, n_point: float) -> float:
    """Deletion heteroplasmy: N_del / total. Drives the cliff factor.

    This is the biologically critical measure — the fraction of mtDNA that
    carries LARGE DELETIONS specifically. Only deletions drive the cliff
    because:
      1. Large deletions remove multiple ETC subunit genes (ND1-6, CO1-3,
         cytb, ATP6/8). Without these genes, respiratory complexes I, III,
         IV, and V cannot assemble.
      2. Below ~70% deletion heteroplasmy, the remaining wild-type copies
         can complement the defect (threshold effect). Above 70%, there
         aren't enough wild-type copies to sustain oxidative phosphorylation.
      3. Point mutations rarely knock out entire complexes — they cause
         single amino acid changes that may be neutral or mildly deleterious.

    Cramer Appendix 2 pp.152-155: the distinction between large deletions
    (>3 kbp, replication advantage) and point mutations (no advantage)
    is fundamental to the exponential accumulation model.

    Returns 1.0 if total copy number is near zero (depletion guard).
    """
    total = n_healthy + n_deletion + n_point
    if total < 1e-12:
        return 1.0  # Complete mtDNA depletion: treat as fully mutated
    return n_deletion / total


def _cliff_factor(heteroplasmy: float) -> float:
    """Sigmoid cliff function: ATP efficiency drops steeply at the threshold.

    Models the nonlinear collapse of oxidative phosphorylation when
    deletion heteroplasmy exceeds the critical threshold (~70%).

    Biology (Cramer Ch. V.K p.66, Rossignol et al. 2003):
        Mitochondrial genetics exhibits a "threshold effect" — cells
        can tolerate a high fraction of mutant mtDNA because wild-type
        copies complement the defect. But once the wild-type fraction
        drops below ~30% (i.e., deletion heteroplasmy > 70%), there are
        not enough functional ETC subunits to sustain ATP production.
        The transition is sharp because respiratory complex assembly
        requires stoichiometric amounts of multiple subunits.

    Mathematical form:
        cliff(h) = 1 / (1 + exp(k * (h - h_cliff)))

        where k = CLIFF_STEEPNESS = 15.0 (calibrated for ~10% ATP at
        h=0.6 → ~80% ATP drop at h=0.9) and h_cliff = 0.70.

    Returns:
        Float in (0, 1) where:
          - 1.0 = fully healthy (low deletion heteroplasmy)
          - ~0.5 = at the cliff edge (het ≈ 0.70)
          - ~0 = collapsed (het >> 0.70, ATP production near zero)

    NOTE: After C11, this function receives DELETION heteroplasmy only,
    not total heteroplasmy. Point mutations do not drive the cliff.
    """
    return 1.0 / (1.0 + np.exp(CLIFF_STEEPNESS * (heteroplasmy - HETEROPLASMY_CLIFF)))


def _deletion_rate(age: float, genetic_vulnerability: float,
                   atp_norm: float = 1.0, mitophagy_rate: float = BASELINE_MITOPHAGY_RATE) -> float:
    """ATP- and mitophagy-dependent mtDNA deletion rate.

    Computes the rate at which new large-scale deletions appear in mtDNA,
    primarily through Pol-gamma slippage at direct repeats during replication
    and double-strand break misrepair.

    Empirical data (Cramer Appendix 2, p.155, Fig. 23):
        Vandiver et al. (Va23, Aging Cell 22(6), 2023) measured deletion
        levels across human lifespan and found two distinct exponential
        growth regimes:
          - Before age ~65: doubling time = 11.81 years (slow accumulation)
          - After age ~65: doubling time = 3.06 years (accelerated damage)

        The transition reflects declining mitochondrial quality control:
        as the cell ages, mitophagy efficiency drops, DNA repair weakens,
        and the deletion rate accelerates.

    Correction history:
        C9  (2026-02-15): AGE_TRANSITION restored to 65 (was incorrectly 40).
        C10 (2026-02-15): Transition age is no longer fixed. It shifts
            earlier or later depending on cellular health:
              - High ATP + effective mitophagy -> shift UP to +10 years
                (cell maintains "young" repair capacity longer)
              - Low ATP + poor mitophagy -> shift DOWN to -15 years
                (early onset of accelerated damage)

    C10 calibration:
        NATURAL_HEALTH_REF = 0.77 was iteratively determined so that a
        naturally aging person (no intervention, default patient) produces
        shift ≈ 0 at age 65, matching the Va23 empirical data point.
        Natural aging yields ATP_norm ≈ 0.774 at age 65 with baseline
        mitophagy rate = 0.02; normalizing by 0.77 gives a residual shift
        of < 0.05 years (18 days), satisfying Cramer's requirement that
        the AVERAGE transition age is 65.

    Args:
        age: Current biological age in years (baseline_age + simulation time).
        genetic_vulnerability: Haplogroup-dependent susceptibility multiplier
            (1.0 = average, 2.0 = highly vulnerable, e.g., certain mtDNA
            haplogroups with more direct repeats).
        atp_norm: Normalized ATP level (0-1). Low ATP indicates cellular
            energy crisis, which impairs DNA repair and accelerates deletions.
        mitophagy_rate: Current mitophagy rate (baseline + rapamycin + NAD
            contributions). Higher mitophagy clears damaged organelles
            faster, slowing net deletion accumulation.

    Returns:
        Deletion rate in units of 1/year, scaled by genetic vulnerability.
        This rate feeds into the de novo deletion term in the dN_del equation.
    """
    # ── C10: Health-dependent transition age shift ──
    # The key insight is that the age-65 transition in Va23 data reflects
    # AVERAGE cellular health at age 65. Interventions that maintain
    # higher ATP and mitophagy effectively delay this transition.
    NATURAL_HEALTH_REF = 0.77  # calibrated: ATP_norm at age 65, no intervention

    # health_factor combines ATP level (energy for DNA repair) with
    # mitophagy efficiency (clearance of damaged organelles). Both are
    # required for maintaining "young" deletion dynamics.
    health_factor = atp_norm * (mitophagy_rate / BASELINE_MITOPHAGY_RATE)

    # shift > 0 means "healthier than average at this age" -> transition later
    # shift < 0 means "sicker than average" -> transition earlier
    # The factor of 10.0 scales the health deviation into years.
    # Asymmetric caps: +10 years (best case) vs -15 years (worst case)
    # reflect that deterioration is easier than preservation.
    shift = 10.0 * (health_factor / NATURAL_HEALTH_REF - 1.0)
    shift = max(-15.0, min(shift, 10.0))  # cap the shift range
    effective_transition = AGE_TRANSITION + shift

    # ── Smooth sigmoid blend instead of hard cutoff ──
    # Width parameter 2.5 gives a ~5-year transition window
    # (from 90% young-rate to 90% old-rate over ~10 years centered
    # on effective_transition). This avoids the discontinuity that
    # a hard if/else would create, which would cause numerical
    # artifacts in the RK4 integrator.
    blend = 1.0 / (1.0 + np.exp(-(age - effective_transition) / 2.5))

    # Weighted average of young and old doubling times
    # blend=0 -> DOUBLING_TIME_YOUNG (11.8 yr), blend=1 -> DOUBLING_TIME_OLD (3.06 yr)
    doubling_time = DOUBLING_TIME_YOUNG * (1.0 - blend) + DOUBLING_TIME_OLD * blend

    # Convert doubling time to exponential growth rate: rate = ln(2) / DT
    # Then scale by genetic vulnerability (haplogroup-dependent factor).
    # A vulnerability of 2.0 means deletions accumulate twice as fast
    # (e.g., haplogroups with more direct repeat sequences in mtDNA).
    return (np.log(2) / doubling_time) * genetic_vulnerability


def derivatives(
    state: npt.NDArray[np.float64],
    t: float,
    intervention: dict[str, float],
    patient: dict[str, float],
    tissue_mods: dict[str, float] | None = None,
) -> npt.NDArray[np.float64]:
    """Compute time derivatives of all 8 state variables.

    This is the ODE right-hand side: dstate/dt = f(state, t, params).
    Called 4 times per RK4 step (at t, t+dt/2, t+dt/2, t+dt) to achieve
    4th-order accuracy in the time integration.

    C11 split (Cramer email 2026-02-17, Appendix 2 pp.152-155):
      - N_damaged split into N_deletion (index 1) and N_point (index 7)
      - Deletions: exponential growth with 1.21x replication advantage
        (shorter rings replicate faster). Drives the cliff.
      - Point mutations: linear growth at same rate as healthy mtDNA.
        No replication advantage (same-length molecule).
      - ROS coupling weakened: ~33% of old coefficient. ROS-induced
        oxidative damage (8-oxoG) primarily creates point mutations,
        not large deletions (which require replication slippage or DSBs).
      - Cliff factor now based on deletion heteroplasmy only.

    Previous fixes preserved: C1-C4, C7-C8, C10, M1-M5.

    Args:
        state: np.array of shape (8,) -- current state vector:
            [N_h, N_del, ATP, ROS, NAD, Sen, ΔΨ, N_pt]
        t: Current time in years from simulation start (NOT calendar age;
            calendar age = patient["baseline_age"] + t).
        intervention: Dict with 6 intervention parameter values (0-1 each).
        patient: Dict with 6 patient parameter values.
        tissue_mods: Optional dict with tissue-specific modifiers:
            "ros_sensitivity" (multiplier on ROS damage to healthy mitos)
            "biogenesis_rate" (multiplier on exercise-induced biogenesis)

    Returns:
        np.array of shape (8,) -- time derivatives in same order as state:
            [dN_h/dt, dN_del/dt, dATP/dt, dROS/dt, dNAD/dt,
             dSen/dt, dΔΨ/dt, dN_pt/dt]
    """
    if tissue_mods is None:
        tissue_mods = _DEFAULT_TISSUE_MODS

    # ── Unpack state vector ────────────────────────────────────────────────
    # Each variable represents a different aspect of mitochondrial/cellular
    # health. The 8-element vector is the minimal representation needed to
    # capture the key feedback loops in mitochondrial aging.
    n_h, n_del, atp, ros, nad, sen, psi, n_pt = state
    # n_h:   Healthy (wild-type) mtDNA copy number, normalized so 1.0 = full
    #         complement. These copies encode all 13 ETC subunits correctly.
    # n_del: Deletion-mutated mtDNA copies. Missing >3 kbp of the 16.5 kbp
    #         genome, including ETC subunit genes. Shorter ring -> faster
    #         replication -> clonal expansion (Cramer Appendix 2 pp.154-155).
    # atp:   ATP production rate in metabolic units/day (1 MU = 10^8 ATP
    #         releases, Cramer Ch. VIII.A Table 3 p.100). Driven by cliff
    #         factor, NAD availability, and senescence burden.
    # ros:   Reactive oxygen species level (normalized). Produced by
    #         mitochondrial electron transport chain, elevated when damaged
    #         mitos have defective ETC (electron leakage -> superoxide).
    # nad:   NAD+ cofactor availability (normalized, 1.0 = young adult level).
    #         Declines with age (~1%/year after 30, Camacho-Pereira 2016).
    #         Required for ETC Complex I and sirtuins.
    # sen:   Fraction of cells in irreversible growth arrest (senescence).
    #         Senescent cells consume ~2x ATP (Cramer Ch. VIII.F p.103),
    #         secrete SASP inflammatory factors, and cannot be replaced.
    # psi:   Mitochondrial inner membrane potential (ΔΨ, normalized to 1.0
    #         = ~180 mV in healthy mitos). Low ΔΨ triggers PINK1
    #         accumulation on the outer membrane -> Parkin recruitment ->
    #         mitophagy (Cramer Ch. VI.B p.75).
    # n_pt:  Point-mutated mtDNA copies. Single nucleotide changes from
    #         Pol-gamma errors or ROS-induced 8-oxoguanine. Same-length
    #         molecule -> no replication advantage -> linear accumulation.

    # ── Clamp state variables to prevent negative values ───────────────────
    # Negative concentrations are biologically meaningless. Small negative
    # values can occur transiently during RK4 midpoint evaluations.
    # n_h and n_del are clamped to 1e-6 (not zero) to avoid division-by-zero
    # in heteroplasmy calculations.
    n_h = max(n_h, 1e-6)     # Must have some healthy copies (numerical guard)
    n_del = max(n_del, 1e-6) # Must have some deletions (numerical guard)
    n_pt = max(n_pt, 0.0)    # Point mutations can be zero (young patients)
    atp = max(atp, 0.0)      # ATP cannot be negative
    ros = max(ros, 0.0)      # ROS cannot be negative
    nad = max(nad, 0.0)      # NAD+ cannot be negative
    sen = max(sen, 0.0)      # Senescent fraction cannot be negative
    psi = max(psi, 0.0)      # Membrane potential cannot be negative

    # ── Unpack intervention parameters ─────────────────────────────────────
    # Each intervention modifies specific terms in the ODE system.

    # Rapamycin: mTOR inhibitor. Inhibiting mTOR upregulates autophagy
    # and specifically mitophagy (the selective degradation of damaged
    # mitochondria). Cramer Ch. VI.A.1 pp.71-72.
    rapa = intervention["rapamycin_dose"]

    # NAD+ supplement: NMN (nicotinamide mononucleotide) or NR (nicotinamide
    # riboside) that gets converted to NAD+. Restores the age-declining
    # NAD+ pool needed for ETC Complex I and sirtuin activity.
    # Cramer Ch. VI.A.3 pp.72-73 (citing Ca16 = Camacho-Pereira 2016).
    # IMPORTANT (C7): The actual NAD+ boost is gated by CD38 survival
    # (see cd38_survival below). At low doses, CD38 enzyme destroys most
    # of the NMN/NR before it can be converted to NAD+.
    nad_supp = intervention["nad_supplement"]

    # Senolytic drugs: dasatinib + quercetin + fisetin (D+Q+F) that
    # selectively kill senescent cells. Removes the metabolic burden
    # and SASP inflammatory signaling of zombie cells.
    # Cramer Ch. VII.A.2 p.91.
    seno = intervention["senolytic_dose"]

    # Yamanaka reprogramming: Transient expression of Oct4, Sox2, Klf4, Myc
    # (OSKM factors) to partially reprogram cell state. Can convert
    # damaged mtDNA back to healthy copies. MAJOR CAVEAT: costs 3-5 MU
    # of ATP per day (Cramer Ch. VIII.A Table 3 p.100). At high intensity,
    # the energy cost can exceed the cell's ATP budget, causing harm.
    yama = intervention["yamanaka_intensity"]

    # Mitochondrial transplant: Infusion of healthy mtDNA via platelet-
    # derived mitochondrial packages ("mitlets"). The ONLY method that
    # directly adds new healthy copies from an external source.
    # Cramer Ch. VIII.G pp.104-107. Per Cramer email (C8): this is the
    # primary rejuvenation mechanism -- NAD boosting and mitophagy can
    # slow decline, but only transplant can truly reverse it.
    transplant = intervention["transplant_rate"]

    # Exercise: Hormetic stimulus. Moderate exercise transiently increases
    # ROS (exercise_ros term), which upregulates endogenous antioxidant
    # defenses (defense_factor term) AND stimulates mitochondrial
    # biogenesis via PGC-1alpha (exercise_biogenesis term).
    # The net effect at moderate levels is beneficial despite the
    # transient ROS increase.
    exercise = intervention["exercise_level"]

    # ── CD38 survival factor (C7, Cramer email) ───────────────────────────
    # CD38 is an ectoenzyme that degrades NMN and NR in the bloodstream
    # before they can enter cells and be converted to NAD+.
    # Cramer Ch. VI.A.3 p.73: CD38 destroys most supplemented NMN/NR.
    #
    # At low nad_supplement doses (e.g., 0.25):
    #   cd38_survival = 0.4 + 0.6 * 0.25 = 0.55 (only 55% reaches cells)
    # At high nad_supplement doses (e.g., 1.0):
    #   cd38_survival = 0.4 + 0.6 * 1.0 = 1.0 (100% reaches cells)
    #
    # The rationale: high-dose protocols include apigenin (a CD38 inhibitor)
    # along with NMN/NR, effectively suppressing CD38 degradation.
    # Low-dose protocols do not include apigenin, so CD38 destroys most
    # of the supplement. This creates a nonlinear dose-response curve.
    cd38_survival = CD38_BASE_SURVIVAL + CD38_SUPPRESSION_GAIN * nad_supp

    # ── Unpack patient parameters ──────────────────────────────────────────

    # Current biological age = starting age + simulation time elapsed.
    # Age affects: deletion rate transition, NAD decline, senescence onset,
    # immune clearance of senescent cells.
    age = patient["baseline_age"] + t

    # Genetic vulnerability: haplogroup-dependent susceptibility to mtDNA
    # damage. Multiplies the deletion rate. Range 0.5 (resilient haplogroup)
    # to 2.0 (highly vulnerable, e.g., haplogroups with many direct repeats
    # that promote replication slippage).
    gen_vuln = patient["genetic_vulnerability"]

    # Metabolic demand: tissue-specific energy requirement. Brain = 2.0
    # (20% of body's ATP for 2% of mass), skin = 0.5.
    # Higher demand -> more ROS production (more electron transport needed).
    met_demand = patient["metabolic_demand"]

    # Chronic inflammation level: "inflammaging" (Cramer Ch. VII.A pp.89-90).
    # Amplifies ROS damage. Range 0.0 (no inflammation) to 1.0 (severe).
    inflammation = patient["inflammation_level"]

    # ── Derived quantities ────────────────────────────────────────────────
    # These intermediate values are computed from the current state and
    # used by multiple terms in the ODE equations below.

    # Total mtDNA copy number (all three pools). Should stay near 1.0
    # due to copy number homeostasis (C2). Guard against division by zero.
    total = max(n_h + n_del + n_pt, 1e-12)

    # Deletion heteroplasmy: fraction of total mtDNA that carries large
    # deletions. THIS is what drives the cliff -- not total heteroplasmy.
    # (C11: cliff based on deletion heteroplasmy only.)
    het_del = n_del / total

    # Total heteroplasmy: fraction of total mtDNA carrying ANY mutation.
    # Used for the ROS equation because both deletion and point mutations
    # cause some degree of ETC dysfunction and electron leakage.
    het_total = (n_del + n_pt) / total

    # Cliff factor: sigmoid function of deletion heteroplasmy.
    # 1.0 = healthy (het_del << 0.70), ~0 = collapsed (het_del >> 0.70).
    # C11: only deletion heteroplasmy drives the cliff.
    cliff = _cliff_factor(het_del)

    # Normalized ATP: current ATP as fraction of baseline (1.0 MU/day).
    # Capped at 1.0 because super-physiological ATP doesn't provide
    # additional repair capacity in this model.
    atp_norm = min(atp / BASELINE_ATP, 1.0)

    # Energy available: like atp_norm but floored at 0.05 to represent
    # minimal residual energy even in crisis (cells don't instantly die
    # at zero ATP -- they enter a low-energy survival state).
    energy_available = max(atp_norm, 0.05)

    # ── Copy number regulation (C2, now across 3 pools) ──────────────────
    # Homeostatic pressure to maintain total copy number near 1.0.
    # When total < 1.0: positive pressure -> replication encouraged.
    # When total > 1.0: negative pressure -> replication suppressed.
    # Cap at -0.5 to prevent excessive negative regulation from
    # transplant-induced overshoot (C8 allows headroom up to 1.5).
    #
    # Without this term, the copy number would drift unboundedly as
    # all pools replicate independently. The biology: cells sense total
    # mtDNA via TFAM (mitochondrial transcription factor A) availability.
    # TFAM coats and stabilizes mtDNA; when copies are low, free TFAM
    # promotes replication; when copies are high, TFAM is saturated.
    copy_number_pressure = max(1.0 - total, -0.5)

    # ── Age- and health-dependent deletion rate (C10) ────────────────────
    # Compute the current effective mitophagy rate INCLUDING drug effects.
    # Rapamycin boosts mitophagy by 0.08/yr at max dose (mTOR inhibition
    # -> upregulated autophagy, Cramer Ch. VI.A.1 pp.71-72).
    # NAD supplement indirectly boosts mitophagy by 0.03/yr at max dose
    # (NAD+ -> sirtuin activation -> PINK1/Parkin pathway) but this
    # boost is also subject to CD38 degradation.
    _current_mitophagy_rate = (BASELINE_MITOPHAGY_RATE
                               + rapa * 0.08
                               + nad_supp * 0.03 * cd38_survival)

    # The deletion rate depends on age, genetic vulnerability, current
    # ATP level, and mitophagy rate. See _deletion_rate() docstring for
    # the C9/C10 correction details and calibration.
    del_rate = _deletion_rate(age, gen_vuln, atp_norm=energy_available,
                              mitophagy_rate=_current_mitophagy_rate)

    # ══════════════════════════════════════════════════════════════════════
    # ODE EQUATIONS: 8 coupled differential equations
    # ══════════════════════════════════════════════════════════════════════

    # ── 1. dN_healthy/dt ─────────────────────────────────────────────────
    # Healthy mtDNA copies change due to:
    #   (+) Replication of existing healthy copies
    #   (-) ROS-induced point mutation damage (converts healthy -> point)
    #   (+) Transplant addition of new healthy copies from external source
    #   (+) Transplant displacement of damaged copies (competitive advantage)
    #   (+) Yamanaka repair of deletions back to healthy
    #   (+) Yamanaka repair of point mutations back to healthy
    #   (+) Exercise-stimulated biogenesis (new healthy copies via PGC-1alpha)
    #   (-) Apoptosis (cell death from energy crisis)

    # Base replication rate: 0.1/year = ~10% of healthy copies replicate
    # per year under optimal conditions. This is the "raw engine speed"
    # of the mtDNA replication machinery (Pol-gamma + helicase + SSB).
    base_replication_rate = 0.1

    # Healthy replication: proportional to existing copies (n_h), gated
    # by NAD+ (required for sirtuin-mediated replication licensing),
    # by energy (replication costs ATP), and by copy number pressure
    # (homeostasis: suppress replication when total copies are high).
    # The max(copy_number_pressure, 0.0) ensures replication stops when
    # total > 1.0 but never goes negative (copies don't destroy themselves
    # just because there are too many).
    replication_h = (base_replication_rate * n_h * nad
                     * energy_available * max(copy_number_pressure, 0.0))

    # C11: ROS-induced damage now creates POINT mutations only, not deletions.
    # Biological rationale: ROS causes 8-oxoguanine (a point mutation) by
    # oxidizing guanine bases. ROS does NOT cause large deletions -- those
    # require Pol-gamma slippage at direct repeats during replication.
    # The coefficient ROS_POINT_COEFF = 0.05 is ~33% of the old unified
    # damage rate of 0.15, reflecting that ROS damage is less catastrophic
    # per event than the old model assumed (most 8-oxoG is repaired by
    # OGG1 before it becomes a permanent mutation).
    # This term REMOVES copies from N_healthy and the corresponding
    # term in dN_pt ADDS them to N_point.
    ros_point_damage = (ROS_POINT_COEFF * ros * gen_vuln * n_h
                        * tissue_mods["ros_sensitivity"])

    # Transplant: external addition of healthy mtDNA copies (C8).
    # Cramer Ch. VIII.G pp.104-107: bioreactor-grown stem cells donate
    # mitochondria encapsulated in "mitlets" (platelet-derived packages).
    #
    # transplant_headroom: how much room there is for new copies before
    # hitting the ceiling. TRANSPLANT_HEADROOM = 1.5 allows transplant
    # to push total copies above the normal 1.0 (overshoot is OK during
    # active transplant therapy). The min(transplant_headroom, 1.0)
    # ensures the addition rate doesn't exceed the base rate even when
    # there's lots of headroom.
    transplant_headroom = max(TRANSPLANT_HEADROOM - total, 0.0)
    transplant_add = (transplant * TRANSPLANT_ADDITION_RATE
                      * min(transplant_headroom, 1.0))

    # Transplant displacement: healthy transplanted mitos OUTCOMPETE
    # deletion-mutated mitos for cellular resources (membrane space,
    # fission/fusion partners, TFAM binding). This is the mechanism by
    # which transplant can actually REDUCE deletion heteroplasmy, not
    # just add to total copies. The displacement rate is proportional
    # to n_del (more targets = more displacement) and energy_available
    # (the displacement process itself requires ATP).
    # The same amount displaced from N_del is NOT added to N_h (it's
    # a destruction of damaged copies, not conversion).
    transplant_displace = (transplant * TRANSPLANT_DISPLACEMENT_RATE
                           * n_del * energy_available)

    # Yamanaka repair: partial epigenetic reprogramming can "repair"
    # damaged mtDNA by activating DNA repair pathways and selectively
    # degrading/replacing damaged copies (Cramer Ch. VII.B pp.92-95).
    # M1 (fix): Yamanaka repair is GATED BY ATP -- reprogramming is
    # extremely energy-intensive. If ATP is low, the cell cannot afford
    # to run the reprogramming machinery.
    # Repair rates differ: deletions are harder to repair (0.05) than
    # point mutations (0.02) because large deletions require de novo
    # DNA synthesis to replace missing segments, while point mutations
    # only need base excision repair.
    repair_deletion = yama * 0.05 * n_del * energy_available
    repair_point = yama * 0.02 * n_pt * energy_available

    # Exercise-stimulated biogenesis (M5): moderate exercise activates
    # PGC-1alpha, the master regulator of mitochondrial biogenesis.
    # New healthy copies are synthesized de novo. This is gated by:
    #   - energy_available: biogenesis costs ATP
    #   - copy_number_pressure: homeostasis prevents overshoot
    #   - tissue_mods["biogenesis_rate"]: tissue-specific PGC-1alpha
    #     responsiveness (muscle = 1.5, brain = 0.3, cardiac = 0.5)
    exercise_biogenesis = (exercise * 0.03 * energy_available
                           * max(copy_number_pressure, 0.0)
                           * tissue_mods["biogenesis_rate"])

    # Apoptosis of healthy copies (C1: cliff feedback).
    # When energy collapses (cliff crossed), the cell activates
    # mitochondrial apoptosis pathways (cytochrome c release ->
    # caspase cascade). The (1 - cliff) factor means apoptosis is
    # negligible when healthy (cliff ~ 1.0) and maximal when collapsed
    # (cliff ~ 0.0). The (1 - energy_available) factor ensures
    # apoptosis only triggers during energy crisis.
    # 0.02 = base apoptosis rate constant.
    apoptosis_h = 0.02 * max(1.0 - energy_available, 0.0) * n_h * (1.0 - cliff)

    # Sum all terms for dN_h/dt
    dn_h = (replication_h - ros_point_damage + transplant_add
            + repair_deletion + repair_point + exercise_biogenesis - apoptosis_h)

    # ── 2. dN_deletion/dt (C11: REVISED from old dN_damaged) ─────────────
    # Deletion-mutated mtDNA copies change due to:
    #   (+) Replication with advantage (shorter rings -> faster replication)
    #   (+) De novo deletions from Pol-gamma slippage
    #   (-) Mitophagy: selective removal (deletions have low ΔΨ -> PINK1)
    #   (-) Yamanaka repair (converts deletions back to healthy)
    #   (-) Apoptosis (cell death from energy crisis)
    #   (-) Transplant displacement (healthy mitos outcompete damaged ones)

    # Deletion replication: DELETION_REPLICATION_ADVANTAGE = 1.21 means
    # deletion-mutated mtDNA replicates 21% faster than healthy copies.
    # Cramer Appendix 2 pp.154-155: "at least 21% faster" for deletions
    # >3 kbp (Va23 data). Our 1.21 enforces the Appendix 2 minimum.
    # would create even faster cliff approach.
    # This is the CORE MECHANISM of clonal expansion: even a small
    # replication advantage compounds exponentially over decades.
    # C4 (bistability): this advantage, combined with copy number pressure,
    # creates a positive feedback loop past the cliff. Once deletions
    # dominate, they out-replicate healthy copies and heteroplasmy
    # ratchets upward irreversibly.
    replication_del = (base_replication_rate * DELETION_REPLICATION_ADVANTAGE
                       * n_del * nad * energy_available
                       * max(copy_number_pressure, 0.0))

    # De novo deletions: new large deletions arising from Pol-gamma
    # replication errors. When Pol-gamma encounters direct repeats in
    # mtDNA, it can "slip" and skip a section, creating a deletion.
    # This is NOT driven by ROS (C11 insight: ROS causes point mutations,
    # not deletions). The deletion rate (del_rate) already incorporates
    # age dependence and genetic vulnerability.
    # The 0.05 * n_h factor means deletions arise from HEALTHY copies
    # (you can't delete from an already-deleted molecule), and the rate
    # is proportional to energy_available (replication requires ATP).
    age_deletions = del_rate * 0.05 * n_h * energy_available

    # Mitophagy: selective degradation of deletion-bearing mitochondria.
    # Biology (Cramer Ch. VI.B p.75): deletion-mutated mitos have
    # defective ETC -> low membrane potential ΔΨ -> PINK1 kinase
    # accumulates on outer membrane (normally cleaved by active import)
    # -> recruits Parkin E3 ligase -> ubiquitination -> autophagosome.
    # C3: this selectivity is the cell's quality control mechanism.
    # Deletions are HIGHLY selected against because they cause severe
    # ΔΨ loss (missing ETC subunits), unlike point mutations which may
    # have near-normal ΔΨ.
    mitophagy_del = _current_mitophagy_rate * n_del

    # Apoptosis of deletion copies: same mechanism as healthy apoptosis
    # (energy crisis -> cytochrome c release). Affects all copy pools
    # proportionally.
    apoptosis_del = (0.02 * max(1.0 - energy_available, 0.0)
                     * n_del * (1.0 - cliff))

    # Sum all terms for dN_del/dt
    dn_del = (replication_del + age_deletions
              - mitophagy_del - repair_deletion - apoptosis_del
              - transplant_displace)

    # ── 3. dN_point/dt (C11: NEW equation) ────────────────────────────────
    # Point-mutated mtDNA copies change due to:
    #   (+) Replication at SAME rate as healthy (no size advantage)
    #   (+) New point mutations from Pol-gamma copying errors
    #   (+) ROS-induced oxidative damage (8-oxoG converted to mutations)
    #   (-) Mitophagy: LOW selectivity (point mutants often have normal ΔΨ)
    #   (-) Yamanaka repair (converts point mutations back to healthy)
    #   (-) Apoptosis (energy crisis)

    # Point mutation replication: unlike deletions, point-mutated mtDNA
    # has the SAME length as wild-type (16,569 bp) and therefore the
    # SAME replication rate. No clonal expansion advantage. This is why
    # point mutations accumulate linearly, not exponentially.
    replication_pt = (base_replication_rate * n_pt * nad
                      * energy_available * max(copy_number_pressure, 0.0))

    # Pol-gamma copying errors: every time a healthy copy replicates,
    # there's a small chance (POINT_ERROR_RATE = 0.001 = 0.1%) that
    # a point mutation is introduced. This is the basal error rate of
    # the mitochondrial DNA polymerase gamma, which lacks the proofreading
    # fidelity of nuclear DNA polymerases.
    point_from_replication = POINT_ERROR_RATE * replication_h

    # Mitophagy of point mutants: MUCH LESS selective than for deletions.
    # POINT_MITOPHAGY_SELECTIVITY = 0.3 means point mutants are cleared
    # at only 30% the rate of deletions. Biological reason: point
    # mutations typically cause mild or no ETC dysfunction, so the
    # membrane potential stays near-normal, and the PINK1/Parkin
    # quality control pathway is not triggered.
    mitophagy_pt = (_current_mitophagy_rate * POINT_MITOPHAGY_SELECTIVITY
                    * n_pt)

    # Apoptosis of point mutation copies (same mechanism as above)
    apoptosis_pt = (0.02 * max(1.0 - energy_available, 0.0)
                    * n_pt * (1.0 - cliff))

    # Sum all terms for dN_pt/dt
    # Note: ros_point_damage appears here as a SOURCE (it was a SINK in
    # the dN_h equation -- healthy copies convert to point-mutated copies).
    dn_pt = (replication_pt + point_from_replication + ros_point_damage
             - mitophagy_pt - repair_point - apoptosis_pt)

    # ── 4. dATP/dt (equilibrium equation, uses DELETION cliff) ───────────
    # ATP production relaxes toward a target level with time constant ~1 year.
    # The target is determined by:
    #   - cliff: deletion heteroplasmy determines ETC functional capacity
    #   - NAD: required cofactor for Complex I (NADH -> NAD+ + 2e- + H+)
    #   - senescence: senescent cells consume ~2x ATP (Cramer Ch. VIII.F p.103)
    #     so more senescence -> less net ATP available
    #
    # The (0.6 + 0.4 * NAD) factor models that ATP production is ~60%
    # from substrate-level phosphorylation (glycolysis, TCA cycle) which
    # doesn't need NAD+ directly, and ~40% from oxidative phosphorylation
    # which requires NADH as electron donor.
    atp_target = (BASELINE_ATP * cliff
                  * (0.6 + 0.4 * min(nad, 1.0))
                  * (1.0 - 0.15 * sen))

    # Yamanaka energy cost: reprogramming is extremely expensive.
    # At intensity 1.0: cost = 0.15 + 0.20 = 0.35 MU/day.
    # The quadratic term (0.2 * yama) represents diminishing efficiency
    # at higher intensities -- pushing cells harder costs disproportionately
    # more energy. Cramer Ch. VIII.A Table 3 p.100: ~3-5 MU total.
    yama_cost = yama * (0.15 + 0.2 * yama)

    # Exercise metabolic cost: moderate steady-state expenditure.
    # At max exercise (1.0): 0.03 MU/day additional ATP consumption.
    # This is deliberately small because exercise benefits (biogenesis,
    # hormesis) outweigh the metabolic cost at moderate levels.
    exercise_cost = exercise * 0.03

    # Net ATP target after subtracting intervention energy costs.
    # Floor at 0.0 to prevent targeting negative ATP.
    atp_target = max(atp_target - yama_cost - exercise_cost, 0.0)

    # Relaxation dynamics: ATP approaches target with time constant 1 year.
    # The factor 1.0 * (target - current) is a first-order exponential
    # relaxation: d(ATP)/dt = (ATP_target - ATP) / tau, with tau = 1 year.
    # This models the lag between changing conditions and ATP adaptation
    # (mitochondria need time to adjust ETC assembly, membrane composition,
    # and metabolic enzyme levels).
    datp = 1.0 * (atp_target - atp)

    # ── 5. dROS/dt (equilibrium equation, uses TOTAL heteroplasmy) ───────
    # ROS relaxes toward an equilibrium determined by production vs defense.
    # C11: uses total heteroplasmy (both deletion and point mutations
    # cause ETC dysfunction and electron leakage -> superoxide production).

    # Baseline ROS: proportional to metabolic demand (more ETC activity
    # = more electron leakage). Even healthy mitochondria produce some
    # ROS as a byproduct of normal oxidative phosphorylation.
    ros_baseline = BASELINE_ROS * met_demand

    # Damage-dependent ROS: the vicious cycle (Cramer Ch. II.H pp.14-15).
    # Damaged mitochondria have defective ETC complexes that leak
    # electrons to O2, producing superoxide. The het_total^2 term
    # (quadratic in heteroplasmy) models positive feedback: more damage
    # -> more ROS -> more damage. The (1 + inflammation) factor models
    # how chronic inflammation (SASP, TNF-alpha, IL-6) amplifies
    # mitochondrial ROS production.
    ros_from_damage = (ROS_PER_DAMAGED * het_total * het_total
                       * (1.0 + inflammation))

    # Antioxidant defense: NAD-dependent (sirtuins, SOD2 upregulation)
    # plus exercise-induced hormesis (Nrf2 pathway, catalase, GPx).
    # defense_factor > 1.0 means the cell's antioxidant capacity
    # exceeds baseline, reducing equilibrium ROS.
    defense_factor = 1.0 + 0.4 * min(nad, 1.0)
    defense_factor += exercise * 0.2

    # Exercise-generated ROS: transient increase during physical activity.
    # This is the "hormetic signal" that triggers adaptive upregulation
    # of antioxidant defenses. At moderate exercise, the net effect is
    # beneficial (defense_factor increase > exercise_ros increase).
    exercise_ros = exercise * 0.03

    # ROS equilibrium and relaxation (same time constant as ATP, ~1 year)
    ros_eq = (ros_baseline + ros_from_damage + exercise_ros) / defense_factor
    dros = 1.0 * (ros_eq - ros)

    # ── 6. dNAD/dt (equilibrium + drains) ────────────────────────────────
    # NAD+ is a critical cofactor that declines with age. The equation
    # models age-dependent synthesis capacity, supplementation, and
    # consumption by ROS defense and Yamanaka reprogramming.

    # Age-dependent NAD synthesis capacity: declines at 1%/year after age 30.
    # Cramer Ch. VI.A.3 pp.72-73 (citing Ca16 = Camacho-Pereira 2016):
    # NAD+ levels decline ~50% between ages 40-60 in human tissues.
    # Floor at 0.2 (cells maintain minimal NAD+ for survival).
    age_factor = max(1.0 - NAD_DECLINE_RATE * max(age - 30, 0), 0.2)

    # NAD supplementation boost (C7: gated by CD38 survival).
    # At max dose with full CD38 suppression: 0.25 * 1.0 = 0.25 MU boost.
    # At low dose without CD38 suppression: 0.25 * 0.55 = 0.14 MU boost.
    # The nonlinear dose-response from CD38 is the key C7 correction.
    nad_boost = nad_supp * 0.25 * cd38_survival

    # NAD target: age-limited synthesis + supplement boost, capped at 1.2
    # (slight super-physiological level achievable with supplementation).
    nad_target = BASELINE_NAD * age_factor + nad_boost
    nad_target = min(nad_target, 1.2)

    # ROS drain: elevated ROS consumes NAD+ via PARP activation.
    # Poly(ADP-ribose) polymerase uses NAD+ to repair oxidative DNA damage;
    # high ROS -> high PARP activity -> NAD+ depletion.
    ros_drain = 0.03 * ros

    # Yamanaka drain: reprogramming consumes NAD+ (sirtuin competition
    # for the NAD+ pool during epigenetic remodeling).
    yama_drain = yama * 0.03

    # NAD relaxation: approaches target with time constant ~3 years
    # (0.3/year), minus constant drains from ROS and Yamanaka.
    dnad = 0.3 * (nad_target - nad) - ros_drain - yama_drain

    # ── 7. dSenescent_fraction/dt (accumulation - clearance) ─────────────
    # Senescent cells are cells that have entered irreversible growth
    # arrest due to DNA damage, telomere shortening, or metabolic stress.
    # They secrete SASP (senescence-associated secretory phenotype)
    # inflammatory factors and consume ~2x normal ATP.
    # Cramer Ch. VII.A pp.89-92, Ch. VIII.F p.103.

    # Energy stress: how far ATP is below normal. High stress -> more
    # cells pushed into senescence.
    energy_stress = max(1.0 - energy_available, 0.0)

    # New senescent cells: driven by ROS damage (2x coefficient),
    # energy stress, and age (1% additional per year after 40).
    # The age scaling reflects declining DNA repair and checkpoint
    # function with age.
    new_sen = (SENESCENCE_RATE * (1.0 + 2.0 * ros + energy_stress)
               * (1.0 + 0.01 * max(age - 40, 0)))

    # Senolytic clearance: drug-induced killing of senescent cells.
    # Proportional to both drug dose and current senescent fraction
    # (more targets = more clearance).
    clearance = seno * 0.2 * sen

    # Immune clearance: natural immune surveillance of senescent cells
    # via NK cells and cytotoxic T cells. Declines with age (1%/year
    # after 50, reflecting immunosenescence), with a floor of 10%
    # residual immune function.
    immune_clear = 0.01 * sen * max(1.0 - 0.01 * max(age - 50, 0), 0.1)

    # Cap: senescent fraction cannot exceed 1.0 (all cells senescent).
    # If we're already at the cap, no new senescent cells are generated.
    if sen >= 1.0:
        new_sen = 0.0
    dsen = new_sen - clearance - immune_clear

    # ── 8. dMembrane_potential/dt (equilibrium, slave variable) ──────────
    # Membrane potential (ΔΨ) is a "slave" variable that tracks the
    # other state variables with minimal independent dynamics. It
    # reflects the proton gradient across the inner mitochondrial
    # membrane (~180 mV in healthy mitochondria).
    # Cramer Ch. IV pp.46-47: the proton motive force drives ATP synthase.
    # Cramer Ch. VI.B p.75: low ΔΨ triggers PINK1 accumulation -> mitophagy.
    #
    # ΔΨ equilibrium depends on:
    #   - cliff: ETC function determines proton pumping capacity
    #   - NAD: Complex I requires NADH (capped at 1.0)
    #   - senescence: senescent cells have impaired membrane maintenance
    # Relaxation time constant: ~2 years (0.5/year), reflecting the
    # relatively slow turnover of inner membrane lipid composition.
    psi_eq = cliff * min(nad, 1.0) * (1.0 - 0.3 * sen)
    psi_eq = min(psi_eq, BASELINE_MEMBRANE_POTENTIAL)
    dpsi = 0.5 * (psi_eq - psi)

    # ── Return derivative vector ─────────────────────────────────────────
    # Order matches state vector: [N_h, N_del, ATP, ROS, NAD, Sen, ΔΨ, N_pt]
    # The RK4 integrator will use this to advance the state by dt.
    return np.array([dn_h, dn_del, datp, dros, dnad, dsen, dpsi, dn_pt])


def _rk4_step(
    state: npt.NDArray[np.float64],
    t: float,
    dt: float,
    intervention: dict[str, float],
    patient: dict[str, float],
    tissue_mods: dict[str, float] | None = None,
) -> npt.NDArray[np.float64]:
    """Single 4th-order Runge-Kutta step.

    RK4 evaluates the derivative at 4 points within each timestep:
        k1 = f(t,        y)             -- slope at start
        k2 = f(t + dt/2, y + dt/2 * k1) -- slope at midpoint using k1
        k3 = f(t + dt/2, y + dt/2 * k2) -- slope at midpoint using k2
        k4 = f(t + dt,   y + dt * k3)   -- slope at end using k3

    The weighted average (k1 + 2*k2 + 2*k3 + k4) / 6 achieves O(dt^5)
    local error and O(dt^4) global error. This is the standard method
    for non-stiff ODEs and is more than adequate for our biological
    timescales (dt = 0.01 year ~ 3.65 days).

    Note: intervention and patient are held constant within each step
    (they change only between steps for InterventionSchedule).
    """
    k1 = derivatives(state, t, intervention, patient, tissue_mods)
    k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt, intervention, patient, tissue_mods)
    k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt, intervention, patient, tissue_mods)
    k4 = derivatives(state + dt * k3, t + dt, intervention, patient, tissue_mods)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def initial_state(patient: dict[str, float]) -> npt.NDArray[np.float64]:
    """Compute initial state vector from patient parameters.

    Constructs the 8-element state vector at t=0 (simulation start) from
    the patient's baseline characteristics. The key challenge is splitting
    the patient's total baseline_heteroplasmy into deletion vs. point
    mutation fractions using the C11 age-dependent model.

    C11 split logic:
        Young adults (age ~20): damage is mostly point mutations (~60%)
        because deletions haven't had time for exponential expansion.
        DELETION_FRACTION_YOUNG = 0.4 -> 40% of initial damage is deletions.

        Older adults (age ~90): damage is dominated by deletions (~80%)
        because their replication advantage has had decades to compound.
        DELETION_FRACTION_OLD = 0.8 -> 80% of initial damage is deletions.

        The transition is linear in age: deletion_frac interpolates from
        0.4 at age 20 to 0.8 at age 90. This matches the empirical
        observation (Cramer Appendix 2 pp.152-155) that deletion burden
        increases superlinearly with age while point mutations accumulate
        more linearly.

    Total copy number: N_h + N_del + N_pt = 1.0 (normalized). This is
    the homeostatic setpoint maintained by TFAM-mediated copy control.

    State vector (8D):
        [0] N_healthy, [1] N_deletion, [2] ATP, [3] ROS, [4] NAD,
        [5] Senescent_fraction, [6] Membrane_potential, [7] N_point

    Args:
        patient: Dict with patient parameter values including:
            - baseline_heteroplasmy: total fraction of mutated mtDNA (0-0.95)
            - baseline_nad_level: starting NAD+ level (0.2-1.0)
            - metabolic_demand: tissue-specific energy need (0.5-2.0)
            - baseline_age: patient age in years (20-90)

    Returns:
        np.array of shape (8,) -- initial state. All values are
        biologically consistent (e.g., ATP already reflects the cliff
        factor from the initial deletion heteroplasmy).
    """
    het0 = patient["baseline_heteroplasmy"]
    nad0 = patient["baseline_nad_level"]
    met_demand = patient["metabolic_demand"]
    age = patient["baseline_age"]

    # ── C11: Age-dependent deletion fraction of total damage ──────────────
    # age_frac: normalized age (0 at age 20, 1 at age 90, clamped).
    # This interpolation assumes that the balance between deletion and
    # point mutations shifts monotonically with age due to the cumulative
    # effect of deletion replication advantage.
    age_frac = min(max(age - 20.0, 0.0) / 70.0, 1.0)
    deletion_frac = (DELETION_FRACTION_YOUNG
                     + (DELETION_FRACTION_OLD - DELETION_FRACTION_YOUNG) * age_frac)

    # Split total heteroplasmy into three pools
    # n_h0: healthy copies = 1 - total_heteroplasmy
    # n_del0: deletion copies = total_het * deletion_fraction_for_age
    # n_pt0: point mutation copies = total_het * (1 - deletion_fraction)
    n_h0 = 1.0 - het0
    n_del0 = het0 * deletion_frac
    n_pt0 = het0 * (1.0 - deletion_frac)

    # Compute deletion-specific heteroplasmy for cliff factor calculation.
    # This determines the initial ATP and membrane potential.
    het_del0 = _deletion_heteroplasmy(n_h0, n_del0, n_pt0)
    cliff0 = _cliff_factor(het_del0)

    # ── Senescence: age-dependent initial burden ──────────────────────────
    # Starts at 0 for young patients, accumulates at 0.5%/year after age 40.
    # Capped at 0.5 (starting with more than half senescent cells is
    # unrealistic even for very elderly patients).
    sen0 = BASELINE_SENESCENT + 0.005 * max(age - 40, 0)
    sen0 = min(sen0, 0.5)

    # ── ATP: initial level from deletion cliff ────────────────────────────
    # Same formula as the equilibrium target in derivatives(), evaluated
    # at initial conditions. A patient starting at 80% deletion heteroplasmy
    # will have severely depressed initial ATP.
    atp0 = (BASELINE_ATP * cliff0
            * (0.6 + 0.4 * min(nad0, 1.0))
            * (1.0 - 0.15 * sen0))

    # ── ROS: initial level from total heteroplasmy ────────────────────────
    # Uses TOTAL heteroplasmy (het0, not het_del0) because both mutation
    # types contribute to ETC dysfunction and electron leakage.
    ros0 = (BASELINE_ROS * met_demand
            + ROS_PER_DAMAGED * het0 * het0) / (1.0 + 0.4 * min(nad0, 1.0))

    # ── Membrane potential: initial level from deletion cliff ─────────────
    # Tracks cliff factor and NAD, same as the equilibrium in derivatives().
    psi0 = cliff0 * min(nad0, 1.0) * (1.0 - 0.3 * sen0)
    psi0 = min(psi0, BASELINE_MEMBRANE_POTENTIAL)

    return np.array([n_h0, n_del0, atp0, ros0, nad0, min(sen0, 1.0), psi0, n_pt0])


def simulate(
    intervention: dict[str, float] | InterventionSchedule | None = None,
    patient: dict[str, float] | None = None,
    sim_years: float | None = None,
    dt: float | None = None,
    tissue_type: str | None = None,
    stochastic: bool = False,
    noise_scale: float = 0.01,
    n_trajectories: int = 1,
    rng_seed: int | None = None,
    resolver=None,
) -> dict:
    """Run the full mitochondrial aging simulation.

    This is the main entry point for the simulator. It:
    1. Resolves defaults for intervention, patient, and time parameters
    2. Optionally applies tissue-specific modifiers
    3. Computes the initial state from patient parameters
    4. Integrates the ODE system forward in time using either:
       - Deterministic RK4 (default): single trajectory, O(dt^4) accuracy
       - Stochastic Euler-Maruyama: multiple trajectories with noise on
         ROS and damage accumulation for confidence intervals
    5. Records total and deletion heteroplasmy at each timestep

    Args:
        intervention: Dict of 6 intervention params (defaults to no treatment).
            Can also be an InterventionSchedule for time-varying protocols.
        patient: Dict of 6 patient params (defaults to typical 70yo).
        sim_years: Override simulation horizon (default: constants.SIM_YEARS = 30).
        dt: Override timestep (default: constants.DT = 0.01 years ~ 3.65 days).
        tissue_type: Optional tissue type ("brain", "muscle", "cardiac",
            "default"). Overrides metabolic_demand from TISSUE_PROFILES
            and passes tissue-specific ROS sensitivity and biogenesis
            rate to the ODE. Default None = no tissue modification.
        stochastic: If True, use Euler-Maruyama with additive noise on
            ROS generation and damage accumulation. Default False.
        noise_scale: Standard deviation of Wiener process increments
            (only used if stochastic=True). Default 0.01.
        n_trajectories: Number of stochastic trajectories to run
            (only used if stochastic=True). Default 1.
        rng_seed: Optional RNG seed for reproducibility.

    Returns:
        Dict with:
            "time": np.array of time points (years from start), shape (n_steps+1,)
            "states": np.array of shape (n_steps+1, 8) -- full trajectory
                (or (n_trajectories, n_steps+1, 8) if stochastic with
                n_trajectories > 1)
            "heteroplasmy": np.array -- TOTAL heteroplasmy at each step
                (N_del + N_pt) / total. Shape (n_steps+1,) or
                (n_trajectories, n_steps+1) if stochastic.
            "deletion_heteroplasmy": np.array -- deletion-only heteroplasmy
                N_del / total. This is the value that drives the cliff.
                Same shape as heteroplasmy.
            "intervention": the intervention dict (or schedule) used
            "patient": the patient dict used
            "tissue_type": tissue type used (or None)
    """
    # ── Resolve defaults ──────────────────────────────────────────────────
    if intervention is None:
        intervention = dict(DEFAULT_INTERVENTION)  # Copy to avoid mutation
    if patient is None:
        patient = dict(DEFAULT_PATIENT)             # Copy to avoid mutation
    if sim_years is None:
        sim_years = SIM_YEARS   # 30 years
    if dt is None:
        dt = DT                 # 0.01 years ~ 3.65 days

    # ── Apply tissue profile ──────────────────────────────────────────────
    # If a tissue type is specified, override the patient's metabolic_demand
    # and pass tissue-specific ROS sensitivity and biogenesis rate modifiers.
    tissue_mods = None
    if tissue_type is not None:
        profile = TISSUE_PROFILES[tissue_type]
        patient = dict(patient)  # copy to avoid mutating caller's dict
        patient["metabolic_demand"] = profile["metabolic_demand"]
        tissue_mods = {
            "ros_sensitivity": profile["ros_sensitivity"],
            "biogenesis_rate": profile["biogenesis_rate"],
        }

    n_steps = int(sim_years / dt)

    # ── Dispatch to stochastic integrator if requested ────────────────────
    if stochastic and n_trajectories > 1:
        return _simulate_stochastic(
            intervention, patient, n_steps, dt, tissue_mods,
            noise_scale, n_trajectories, rng_seed, resolver=resolver)

    # ── Compute initial state from patient parameters ─────────────────────
    state = initial_state(patient)

    # ── Pre-allocate output arrays ────────────────────────────────────────
    time_arr = np.zeros(n_steps + 1)
    states = np.zeros((n_steps + 1, N_STATES))      # (3001, 8) for default
    het_arr = np.zeros(n_steps + 1)                  # total heteroplasmy
    del_het_arr = np.zeros(n_steps + 1)              # deletion heteroplasmy

    # ── Record initial conditions ─────────────────────────────────────────
    states[0] = state
    # state[0] = N_h, state[1] = N_del, state[7] = N_pt
    het_arr[0] = _total_heteroplasmy(state[0], state[1], state[7])
    del_het_arr[0] = _deletion_heteroplasmy(state[0], state[1], state[7])

    # Set up RNG for single-trajectory stochastic mode
    if stochastic:
        rng = np.random.default_rng(rng_seed)

    # ── Main integration loop ─────────────────────────────────────────────
    for i in range(n_steps):
        t = i * dt
        # Resolve time-varying intervention (handles both dict and schedule)
        if resolver is not None:
            current_intervention, current_patient = resolver.resolve(t)
        else:
            current_intervention = _resolve_intervention(intervention, t)
            current_patient = patient

        if stochastic:
            # ── Euler-Maruyama integration ────────────────────────────────
            # For stochastic mode: deterministic drift + Brownian noise.
            # This is a first-order stochastic integrator (O(dt^0.5) for
            # the noise term), less accurate than RK4 but necessary for
            # modeling biological variability in ROS and mutation rates.
            deriv = derivatives(state, t, current_intervention, current_patient, tissue_mods)

            # Wiener increments: dW ~ N(0, sqrt(dt)) for each state variable
            dW = rng.normal(0, np.sqrt(dt), N_STATES)

            # Multiplicative noise on selected channels only:
            # - Index 1 (N_del): stochastic variation in deletion replication
            # - Index 3 (ROS): stochastic fluctuations in electron leakage
            # - Index 7 (N_pt): stochastic variation in point mutation rate
            # Other channels (ATP, NAD, senescence, ΔΨ, N_h) are NOT noisy
            # because they are driven by equilibrium dynamics that average
            # over many molecular events.
            noise = np.zeros(N_STATES)
            noise[1] = noise_scale * state[1] * dW[1]  # deletion noise
            noise[3] = noise_scale * state[3] * dW[3]  # ROS noise
            noise[7] = noise_scale * state[7] * dW[7]  # point mutation noise
            state = state + dt * deriv + noise
        else:
            # ── Deterministic RK4 integration ─────────────────────────────
            state = _rk4_step(state, t, dt, current_intervention, current_patient, tissue_mods)

        # ── Post-step constraints ─────────────────────────────────────────
        # Enforce non-negativity: all biological quantities must be >= 0.
        # Small negative values can arise from RK4 midpoint evaluations or
        # stochastic noise overshooting zero.
        state = np.maximum(state, 0.0)

        # Cap senescent fraction at 1.0 (100% senescent is the maximum;
        # you cannot have more senescent cells than total cells).
        state[5] = min(state[5], 1.0)

        # ── Record state and heteroplasmy ─────────────────────────────────
        time_arr[i + 1] = (i + 1) * dt
        states[i + 1] = state
        het_arr[i + 1] = _total_heteroplasmy(state[0], state[1], state[7])
        del_het_arr[i + 1] = _deletion_heteroplasmy(state[0], state[1], state[7])

    return {
        "time": time_arr,
        "states": states,
        "heteroplasmy": het_arr,               # total (N_del + N_pt) / total
        "deletion_heteroplasmy": del_het_arr,   # deletion only: N_del / total
        "intervention": intervention,
        "patient": patient,
        "tissue_type": tissue_type,
    }


def _simulate_stochastic(intervention, patient, n_steps, dt, tissue_mods,
                          noise_scale, n_trajectories, rng_seed, resolver=None):
    """Run multiple stochastic trajectories for confidence intervals.

    Uses Euler-Maruyama integration with multiplicative noise on three
    channels: N_deletion (index 1), ROS (index 3), and N_point (index 7).

    The multiplicative noise model (noise ~ state * dW) means that:
    - When a variable is near zero, noise is small (can't go very negative)
    - When a variable is large, noise is proportionally larger
    - This prevents the biologically impossible scenario of large random
      fluctuations creating negative copy numbers

    Why these three channels have noise:
    - N_deletion: mtDNA replication is a stochastic process (random
      segregation during mitochondrial fission means daughter organelles
      get variable numbers of deleted copies). This is "genetic drift"
      at the mitochondrial level.
    - ROS: electron leakage from the ETC is inherently stochastic
      (individual electron tunneling events). ROS levels fluctuate on
      short timescales.
    - N_point: Pol-gamma copying errors are stochastic (each base pair
      has an independent error probability per replication event).

    Why other channels do NOT have noise:
    - ATP, NAD, senescence, ΔΨ are aggregate properties of thousands
      of mitochondria per cell and millions of cells per tissue. Their
      fluctuations average out at the tissue level (law of large numbers).
    - N_healthy: driven by the noisy channels through the ODE coupling;
      adding separate noise would double-count the variability.

    Args:
        intervention, patient: as in simulate().
        n_steps: number of integration steps.
        dt: timestep size.
        tissue_mods: tissue-specific modifiers (or None).
        noise_scale: standard deviation scaling factor for Wiener increments.
        n_trajectories: number of independent stochastic paths to simulate.
        rng_seed: seed for reproducibility.

    Returns:
        Dict with arrays of shape (n_trajectories, n_steps+1, ...).
        Each trajectory starts from the same initial state but diverges
        due to stochastic noise. The ensemble provides confidence intervals.
    """
    rng = np.random.default_rng(rng_seed)
    state0 = initial_state(patient)

    time_arr = np.arange(n_steps + 1) * dt
    all_states = np.zeros((n_trajectories, n_steps + 1, N_STATES))
    all_het = np.zeros((n_trajectories, n_steps + 1))
    all_del_het = np.zeros((n_trajectories, n_steps + 1))

    for traj in range(n_trajectories):
        state = state0.copy()
        all_states[traj, 0] = state
        all_het[traj, 0] = _total_heteroplasmy(state[0], state[1], state[7])
        all_del_het[traj, 0] = _deletion_heteroplasmy(state[0], state[1], state[7])

        for i in range(n_steps):
            t = i * dt
            if resolver is not None:
                current_intervention, current_patient = resolver.resolve(t)
            else:
                current_intervention = _resolve_intervention(intervention, t)
                current_patient = patient
            deriv = derivatives(state, t, current_intervention, current_patient, tissue_mods)

            # Wiener increments for this step
            dW = rng.normal(0, np.sqrt(dt), N_STATES)

            # Multiplicative noise on N_del, ROS, N_pt only
            noise = np.zeros(N_STATES)
            noise[1] = noise_scale * state[1] * dW[1]  # deletion noise
            noise[3] = noise_scale * state[3] * dW[3]  # ROS noise
            noise[7] = noise_scale * state[7] * dW[7]  # point mutation noise

            state = state + dt * deriv + noise
            state = np.maximum(state, 0.0)   # enforce non-negativity
            state[5] = min(state[5], 1.0)    # cap senescent fraction

            all_states[traj, i + 1] = state
            all_het[traj, i + 1] = _total_heteroplasmy(state[0], state[1], state[7])
            all_del_het[traj, i + 1] = _deletion_heteroplasmy(state[0], state[1], state[7])

    return {
        "time": time_arr,
        "states": all_states,
        "heteroplasmy": all_het,
        "deletion_heteroplasmy": all_del_het,
        "intervention": intervention,
        "patient": patient,
        "n_trajectories": n_trajectories,
    }


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Mitochondrial Aging Simulator — Standalone Test")
    print("=" * 70)

    # Test 1: No intervention (natural aging of 70-year-old)
    print("\n--- Test 1: No intervention (70yo, 30% het) ---")
    result = simulate()
    final = result["states"][-1]
    print(f"  Final state at year {SIM_YEARS}:")
    for j, name in enumerate(["N_healthy", "N_deletion", "ATP", "ROS",
                               "NAD", "Senescent", "ΔΨ", "N_point"]):
        print(f"    {name:20s} = {final[j]:.4f}")
    print(f"  Final heteroplasmy: {result['heteroplasmy'][-1]:.4f}")

    # Test 2: Full cocktail intervention
    print("\n--- Test 2: Full cocktail (rapamycin + NAD + senolytics + exercise) ---")
    cocktail = {
        "rapamycin_dose": 0.5,
        "nad_supplement": 0.75,
        "senolytic_dose": 0.5,
        "yamanaka_intensity": 0.0,  # skip Yamanaka (high energy cost)
        "transplant_rate": 0.0,
        "exercise_level": 0.5,
    }
    result2 = simulate(intervention=cocktail)
    final2 = result2["states"][-1]
    print(f"  Final state at year {SIM_YEARS}:")
    for j, name in enumerate(["N_healthy", "N_deletion", "ATP", "ROS",
                               "NAD", "Senescent", "ΔΨ", "N_point"]):
        print(f"    {name:20s} = {final2[j]:.4f}")
    print(f"  Final heteroplasmy: {result2['heteroplasmy'][-1]:.4f}")

    # Test 3: Near-cliff patient
    print("\n--- Test 3: Near-cliff patient (80yo, 65% het) ---")
    near_cliff_patient = {
        "baseline_age": 80.0,
        "baseline_heteroplasmy": 0.65,
        "baseline_nad_level": 0.4,
        "genetic_vulnerability": 1.0,
        "metabolic_demand": 1.0,
        "inflammation_level": 0.5,
    }
    result3 = simulate(patient=near_cliff_patient)
    print(f"  Initial heteroplasmy: {result3['heteroplasmy'][0]:.4f}")
    print(f"  Final heteroplasmy:   {result3['heteroplasmy'][-1]:.4f}")
    print(f"  Initial ATP:          {result3['states'][0, 2]:.4f}")
    print(f"  Final ATP:            {result3['states'][-1, 2]:.4f}")

    # Test 4: Heteroplasmy cliff verification
    print("\n--- Test 4: Cliff verification (sweep 0→0.95, 30yr) ---")
    for het_start in [0.1, 0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]:
        p = dict(DEFAULT_PATIENT)
        p["baseline_heteroplasmy"] = het_start
        r = simulate(patient=p, sim_years=30, dt=0.01)
        print(f"  het={het_start:.2f} → final_ATP={r['states'][-1, 2]:.4f}  "
              f"final_het={r['heteroplasmy'][-1]:.4f}  "
              f"N_total={r['states'][-1, 0] + r['states'][-1, 1] + r['states'][-1, 7]:.3f}")

    # Test 5: Falsifier edge cases
    print("\n--- Test 5: Falsifier edge cases ---")

    # 5a: No damage — should stay healthy
    p5a = dict(DEFAULT_PATIENT)
    p5a["baseline_heteroplasmy"] = 0.01
    p5a["baseline_age"] = 30.0
    p5a["baseline_nad_level"] = 1.0
    r5a = simulate(patient=p5a, sim_years=30)
    print(f"  [5a] het=0.01, age=30: final_het={r5a['heteroplasmy'][-1]:.4f}  "
          f"final_ATP={r5a['states'][-1, 2]:.4f}")

    # 5b: Near-total damage — should collapse
    p5b = dict(DEFAULT_PATIENT)
    p5b["baseline_heteroplasmy"] = 0.90
    r5b = simulate(patient=p5b, sim_years=30)
    print(f"  [5b] het=0.90, age=70: final_het={r5b['heteroplasmy'][-1]:.4f}  "
          f"final_ATP={r5b['states'][-1, 2]:.4f}")

    # 5c: Yamanaka at max — should drain ATP
    i5c = dict(DEFAULT_INTERVENTION)
    i5c["yamanaka_intensity"] = 1.0
    r5c = simulate(intervention=i5c, sim_years=30)
    print(f"  [5c] Yamanaka=1.0: final_ATP={r5c['states'][-1, 2]:.4f}  "
          f"final_het={r5c['heteroplasmy'][-1]:.4f}")

    # 5d: All interventions max
    i5d = {k: 1.0 for k in DEFAULT_INTERVENTION}
    r5d = simulate(intervention=i5d, sim_years=30)
    print(f"  [5d] All max: final_ATP={r5d['states'][-1, 2]:.4f}  "
          f"final_het={r5d['heteroplasmy'][-1]:.4f}  "
          f"N_total={r5d['states'][-1, 0] + r5d['states'][-1, 1] + r5d['states'][-1, 7]:.3f}")

    # 5e: NAD supplementation should REDUCE heteroplasmy (fix C3)
    i_nad = dict(DEFAULT_INTERVENTION)
    i_nad["nad_supplement"] = 1.0
    r_nad = simulate(intervention=i_nad, sim_years=30)
    r_none = simulate(sim_years=30)
    print(f"  [5e] NAD=1.0: het={r_nad['heteroplasmy'][-1]:.4f}  "
          f"vs no-treatment: het={r_none['heteroplasmy'][-1]:.4f}  "
          f"({'PASS: reduced' if r_nad['heteroplasmy'][-1] < r_none['heteroplasmy'][-1] else 'FAIL: increased'})")

    # 5f: Patient starting past cliff should NOT spontaneously recover (fix C4)
    p5f = dict(DEFAULT_PATIENT)
    p5f["baseline_heteroplasmy"] = 0.85
    r5f = simulate(patient=p5f, sim_years=30)
    print(f"  [5f] het=0.85 start: final_het={r5f['heteroplasmy'][-1]:.4f}  "
          f"({'PASS: stayed high' if r5f['heteroplasmy'][-1] > 0.80 else 'FAIL: recovered'})")

    # Test 6: Tissue-specific simulations
    print("\n--- Test 6: Tissue-specific simulations ---")
    for tissue in ["default", "brain", "muscle", "cardiac"]:
        r6 = simulate(tissue_type=tissue, sim_years=30)
        print(f"  {tissue:8s}: final_het={r6['heteroplasmy'][-1]:.4f}  "
              f"final_ATP={r6['states'][-1, 2]:.4f}")

    # Test 7: Stochastic mode (single trajectory)
    print("\n--- Test 7: Stochastic mode ---")
    r7 = simulate(stochastic=True, noise_scale=0.01, rng_seed=42, sim_years=30)
    r7d = simulate(stochastic=False, sim_years=30)
    print(f"  Deterministic: final_het={r7d['heteroplasmy'][-1]:.4f}  "
          f"final_ATP={r7d['states'][-1, 2]:.4f}")
    print(f"  Stochastic:    final_het={r7['heteroplasmy'][-1]:.4f}  "
          f"final_ATP={r7['states'][-1, 2]:.4f}")

    # Test 8: Multi-trajectory stochastic
    print("\n--- Test 8: Multi-trajectory stochastic (10 runs) ---")
    r8 = simulate(stochastic=True, noise_scale=0.02, n_trajectories=10,
                  rng_seed=42, sim_years=30)
    final_hets = r8["heteroplasmy"][:, -1]
    final_atps = r8["states"][:, -1, 2]
    print(f"  Het: mean={np.mean(final_hets):.4f}  "
          f"std={np.std(final_hets):.4f}  "
          f"range=[{np.min(final_hets):.4f}, {np.max(final_hets):.4f}]")
    print(f"  ATP: mean={np.mean(final_atps):.4f}  "
          f"std={np.std(final_atps):.4f}  "
          f"range=[{np.min(final_atps):.4f}, {np.max(final_atps):.4f}]")

    # Test 9: Phased intervention schedule (time-varying)
    print("\n--- Test 9: Phased intervention schedule ---")
    no_treatment = dict(DEFAULT_INTERVENTION)
    full_cocktail = {
        "rapamycin_dose": 0.5,
        "nad_supplement": 0.75,
        "senolytic_dose": 0.5,
        "yamanaka_intensity": 0.0,
        "transplant_rate": 0.0,
        "exercise_level": 0.5,
    }
    # No treatment years 0-10, full treatment years 10-30
    schedule = phased_schedule([(0, no_treatment), (10, full_cocktail)])
    r9_phased = simulate(intervention=schedule, sim_years=30)
    r9_constant = simulate(intervention=full_cocktail, sim_years=30)
    r9_none = simulate(sim_years=30)
    print(f"  No treatment:     final_het={r9_none['heteroplasmy'][-1]:.4f}  "
          f"final_ATP={r9_none['states'][-1, 2]:.4f}")
    print(f"  Phased (0→10→30): final_het={r9_phased['heteroplasmy'][-1]:.4f}  "
          f"final_ATP={r9_phased['states'][-1, 2]:.4f}")
    print(f"  Constant cocktail: final_het={r9_constant['heteroplasmy'][-1]:.4f}  "
          f"final_ATP={r9_constant['states'][-1, 2]:.4f}")
    phased_differs = (abs(r9_phased['heteroplasmy'][-1]
                         - r9_constant['heteroplasmy'][-1]) > 0.001)
    print(f"  Phased differs from constant: "
          f"{'PASS' if phased_differs else 'FAIL'}")

    # Test 10: Cramer corrections — CD38 and transplant (C7, C8)
    print("\n--- Test 10: Cramer corrections (CD38 + transplant) ---")

    # 10a: NAD-only at low dose should have LESS effect than at high dose
    # (CD38 destroys more at low dose)
    i_nad_low = dict(DEFAULT_INTERVENTION)
    i_nad_low["nad_supplement"] = 0.25
    i_nad_high = dict(DEFAULT_INTERVENTION)
    i_nad_high["nad_supplement"] = 1.0
    r_nad_low = simulate(intervention=i_nad_low, sim_years=30)
    r_nad_high = simulate(intervention=i_nad_high, sim_years=30)
    # Benefit ratio: high dose should give >2x the het reduction of low dose
    # (due to CD38 nonlinearity, not just linear dose scaling)
    het_base = r_none['heteroplasmy'][-1]
    benefit_low = het_base - r_nad_low['heteroplasmy'][-1]
    benefit_high = het_base - r_nad_high['heteroplasmy'][-1]
    ratio = benefit_high / max(benefit_low, 1e-6)
    print(f"  [10a] NAD low=0.25: het_benefit={benefit_low:.4f}  "
          f"NAD high=1.0: het_benefit={benefit_high:.4f}  "
          f"ratio={ratio:.1f}x ({'PASS: >2x' if ratio > 2 else 'MARGINAL'})")

    # 10b: Transplant should be highly effective at reducing heteroplasmy
    i_transplant = dict(DEFAULT_INTERVENTION)
    i_transplant["transplant_rate"] = 1.0
    r_transplant = simulate(intervention=i_transplant, sim_years=30)
    transplant_benefit = het_base - r_transplant['heteroplasmy'][-1]
    print(f"  [10b] Transplant=1.0: final_het={r_transplant['heteroplasmy'][-1]:.4f}  "
          f"het_benefit={transplant_benefit:.4f}  "
          f"({'PASS: strong' if transplant_benefit > 0.20 else 'FAIL: weak'})")

    # 10c: Transplant should outperform NAD supplementation for rejuvenation
    nad_benefit = benefit_high  # NAD at max dose
    print(f"  [10c] Transplant benefit={transplant_benefit:.4f} vs "
          f"NAD benefit={nad_benefit:.4f}  "
          f"({'PASS: transplant > NAD' if transplant_benefit > nad_benefit else 'FAIL: NAD > transplant'})")

    # 10d: Transplant should help even near-cliff patients
    i_rescue = dict(DEFAULT_INTERVENTION)
    i_rescue["transplant_rate"] = 1.0
    i_rescue["rapamycin_dose"] = 0.5
    r_rescue = simulate(intervention=i_rescue, patient=near_cliff_patient, sim_years=30)
    print(f"  [10d] Near-cliff + transplant: final_het={r_rescue['heteroplasmy'][-1]:.4f}  "
          f"final_ATP={r_rescue['states'][-1, 2]:.4f}  "
          f"(vs untreated: het={result3['heteroplasmy'][-1]:.4f}  "
          f"ATP={result3['states'][-1, 2]:.4f})")

    print("\n" + "=" * 70)
    print("All tests completed.")
