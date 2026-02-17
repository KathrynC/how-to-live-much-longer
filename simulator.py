"""Numpy-only RK4 ODE integrator for mitochondrial aging dynamics.

Simulates 8 state variables over a configurable time horizon using a
4th-order Runge-Kutta method. Models the core biology from Cramer (2025):
ROS-damage vicious cycle, heteroplasmy cliff, age-dependent deletion
doubling, and six intervention mechanisms.

C11 split (Cramer email 2026-02-17): N_damaged split into N_deletion
(exponential growth, drives cliff) and N_point (linear growth, no
replication advantage). Cliff factor now based on deletion heteroplasmy
only. ROS coupling weakened for point mutations.

Reference:
    Cramer, J.G. (2025). *How to Live Much Longer: The Mitochondrial
    DNA Connection*. ISBN 979-8-9928220-0-4.

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

State variables:
    0. N_healthy          — healthy mtDNA copies (normalized, 1.0 = full)
    1. N_deletion         — deletion-mutated mtDNA (exponential growth, drives cliff)
    2. ATP                — ATP production rate (MU/day)
    3. ROS                — reactive oxygen species level
    4. NAD                — NAD+ availability
    5. Senescent_fraction — fraction of senescent cells
    6. Membrane_potential — mitochondrial ΔΨ (normalized)
    7. N_point            — point-mutated mtDNA (linear growth, C11)

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

# Default tissue modifiers (no modification)
_DEFAULT_TISSUE_MODS = {"ros_sensitivity": 1.0, "biogenesis_rate": 1.0}


# ── Time-varying intervention schedules ─────────────────────────────────────

class InterventionSchedule:
    """Allow interventions to change over the simulation horizon.

    Holds a sorted list of (start_year, intervention_dict) phases.
    At any time t, returns the intervention dict of the most recent phase.

    Usage:
        schedule = InterventionSchedule([
            (0, no_treatment),
            (10, full_cocktail),
        ])
        schedule.at(5)   # → no_treatment
        schedule.at(15)  # → full_cocktail
    """

    def __init__(self, phases: list[tuple[float, dict[str, float]]]) -> None:
        self.phases = sorted(phases, key=lambda x: x[0])

    def at(self, t: float) -> dict[str, float]:
        """Return the intervention dict active at time t."""
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

    Args:
        on_intervention: Intervention dict during "on" phases.
        off_intervention: Intervention dict during "off" phases.
        period: Cycle length in years.
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
    """Resolve intervention at time t (handles both dict and schedule)."""
    if isinstance(intervention, InterventionSchedule):
        return intervention.at(t)
    return intervention


def _heteroplasmy_fraction(n_healthy: float, n_damaged: float) -> float:
    """Compute heteroplasmy as fraction of damaged copies.

    DEPRECATED (C11): Use _total_heteroplasmy() or _deletion_heteroplasmy()
    instead. Kept for backward compatibility with existing code that uses
    the 7-variable state vector (index 1 = N_damaged).
    """
    total = n_healthy + n_damaged
    if total < 1e-12:
        return 1.0
    return n_damaged / total


def _total_heteroplasmy(n_healthy: float, n_deletion: float, n_point: float) -> float:
    """Total heteroplasmy: (N_del + N_pt) / total. For reporting."""
    total = n_healthy + n_deletion + n_point
    if total < 1e-12:
        return 1.0
    return (n_deletion + n_point) / total


def _deletion_heteroplasmy(n_healthy: float, n_deletion: float, n_point: float) -> float:
    """Deletion heteroplasmy: N_del / total. Drives the cliff factor."""
    total = n_healthy + n_deletion + n_point
    if total < 1e-12:
        return 1.0
    return n_deletion / total


def _cliff_factor(heteroplasmy: float) -> float:
    """Sigmoid cliff function: ATP efficiency drops steeply at the threshold.

    Returns a value in (0, 1) where 1.0 = fully healthy and ~0 = collapsed.
    Uses a logistic sigmoid centered at HETEROPLASMY_CLIFF.
    """
    return 1.0 / (1.0 + np.exp(CLIFF_STEEPNESS * (heteroplasmy - HETEROPLASMY_CLIFF)))


def _deletion_rate(age: float, genetic_vulnerability: float,
                   atp_norm: float = 1.0, mitophagy_rate: float = BASELINE_MITOPHAGY_RATE) -> float:
    """ATP- and mitophagy-dependent mtDNA deletion rate.

    Cramer Appendix 2, p.155, Fig. 23 (data from Va23: Vandiver et al.,
    Aging Cell 22(6), 2023): DT = 11.81 yr before age 65, 3.06 yr after.
    Also Ch. II.H, p.15: "deletion damage builds exponentially."

    Corrected 2026-02-15 (C9): AGE_TRANSITION=65 per Cramer email.
    Corrected 2026-02-15 (C10): Per Cramer email, the transition age is NOT
    a fixed number — it is coupled to ATP energy level and mitophagy efficiency.
    When ATP is high and mitophagy is effective, the cell maintains "young"
    repair capacity longer (transition shifts later). When ATP drops or
    mitophagy fails, the transition happens earlier. Implementation: the
    effective transition age is AGE_TRANSITION ± shift based on cellular health,
    with a smooth sigmoid blend (width 5 years) instead of a hard cutoff.
    """
    # Health-dependent shift: good ATP + high mitophagy → later transition.
    # Calibrated so a NATURALLY AGING person (no intervention) has shift≈0
    # at age 65, reproducing the Va23 empirical data. Iterative calibration:
    # natural aging yields ATP_norm ≈ 0.77 at age 65 with baseline mitophagy,
    # so we normalize by that reference (not by perfect health = 1.0).
    # This ensures the AVERAGE transition age ≈ 65 per Cramer's requirement
    # (residual shift < 0.05 years = 18 days).
    NATURAL_HEALTH_REF = 0.77  # calibrated: ATP_norm at age 65, no intervention
    health_factor = atp_norm * (mitophagy_rate / BASELINE_MITOPHAGY_RATE)
    shift = 10.0 * (health_factor / NATURAL_HEALTH_REF - 1.0)
    shift = max(-15.0, min(shift, 10.0))  # cap the shift range
    effective_transition = AGE_TRANSITION + shift

    # Smooth sigmoid blend instead of hard cutoff (width ~5 years)
    blend = 1.0 / (1.0 + np.exp(-(age - effective_transition) / 2.5))
    doubling_time = DOUBLING_TIME_YOUNG * (1.0 - blend) + DOUBLING_TIME_OLD * blend

    # Rate = ln(2) / doubling_time, scaled by genetic vulnerability
    return (np.log(2) / doubling_time) * genetic_vulnerability


def derivatives(
    state: npt.NDArray[np.float64],
    t: float,
    intervention: dict[str, float],
    patient: dict[str, float],
    tissue_mods: dict[str, float] | None = None,
) -> npt.NDArray[np.float64]:
    """Compute time derivatives of all 8 state variables.

    C11 split (Cramer email 2026-02-17, Appendix 2 pp.152-155):
      - N_damaged split into N_deletion (index 1) and N_point (index 7)
      - Deletions: exponential growth with replication advantage (drives cliff)
      - Point mutations: linear growth, no replication advantage
      - ROS coupling weakened: ~33% of old coefficient, feeds only point mutations
      - Cliff factor based on deletion heteroplasmy only

    Previous fixes preserved: C1-C4, C7-C8, C10, M1-M5.

    Args:
        state: np.array of shape (8,) — current state.
        t: Current time in years from simulation start.
        intervention: Dict with 6 intervention parameter values.
        patient: Dict with 6 patient parameter values.
        tissue_mods: Optional tissue-specific modifiers.

    Returns:
        np.array of shape (8,) — derivatives (dstate/dt).
    """
    if tissue_mods is None:
        tissue_mods = _DEFAULT_TISSUE_MODS
    n_h, n_del, atp, ros, nad, sen, psi, n_pt = state

    # Prevent negative values in derivative computation
    n_h = max(n_h, 1e-6)
    n_del = max(n_del, 1e-6)
    n_pt = max(n_pt, 0.0)
    atp = max(atp, 0.0)
    ros = max(ros, 0.0)
    nad = max(nad, 0.0)
    sen = max(sen, 0.0)
    psi = max(psi, 0.0)

    # Unpack intervention
    rapa = intervention["rapamycin_dose"]
    nad_supp = intervention["nad_supplement"]
    seno = intervention["senolytic_dose"]
    yama = intervention["yamanaka_intensity"]
    transplant = intervention["transplant_rate"]
    exercise = intervention["exercise_level"]

    # CD38 survival factor (C7)
    cd38_survival = CD38_BASE_SURVIVAL + CD38_SUPPRESSION_GAIN * nad_supp

    # Unpack patient
    age = patient["baseline_age"] + t
    gen_vuln = patient["genetic_vulnerability"]
    met_demand = patient["metabolic_demand"]
    inflammation = patient["inflammation_level"]

    # ── Derived quantities ────────────────────────────────────────────────
    total = max(n_h + n_del + n_pt, 1e-12)    # zero-division guard
    het_del = n_del / total                    # deletion heteroplasmy (drives cliff)
    het_total = (n_del + n_pt) / total         # total heteroplasmy (for ROS equation)
    cliff = _cliff_factor(het_del)             # C11: cliff from DELETION het only
    atp_norm = min(atp / BASELINE_ATP, 1.0)
    energy_available = max(atp_norm, 0.05)

    # ── Copy number regulation (C2, now across 3 pools) ──────────────────
    copy_number_pressure = max(1.0 - total, -0.5)

    # ── Age- and health-dependent deletion rate (C10) ────────────────────
    _current_mitophagy_rate = (BASELINE_MITOPHAGY_RATE
                               + rapa * 0.08
                               + nad_supp * 0.03 * cd38_survival)
    del_rate = _deletion_rate(age, gen_vuln, atp_norm=energy_available,
                              mitophagy_rate=_current_mitophagy_rate)

    # ── 1. dN_healthy/dt ─────────────────────────────────────────────────
    base_replication_rate = 0.1
    replication_h = (base_replication_rate * n_h * nad
                     * energy_available * max(copy_number_pressure, 0.0))

    # C11: ROS-induced damage creates POINT mutations only (~33% of old rate).
    ros_point_damage = (ROS_POINT_COEFF * ros * gen_vuln * n_h
                        * tissue_mods["ros_sensitivity"])

    # Transplant (C8)
    transplant_headroom = max(TRANSPLANT_HEADROOM - total, 0.0)
    transplant_add = (transplant * TRANSPLANT_ADDITION_RATE
                      * min(transplant_headroom, 1.0))
    transplant_displace = (transplant * TRANSPLANT_DISPLACEMENT_RATE
                           * n_del * energy_available)

    # Yamanaka repair (M1: gated by ATP)
    repair_deletion = yama * 0.05 * n_del * energy_available
    repair_point = yama * 0.02 * n_pt * energy_available

    # Exercise biogenesis (M5)
    exercise_biogenesis = (exercise * 0.03 * energy_available
                           * max(copy_number_pressure, 0.0)
                           * tissue_mods["biogenesis_rate"])

    # Apoptosis (C1: cliff feedback)
    apoptosis_h = 0.02 * max(1.0 - energy_available, 0.0) * n_h * (1.0 - cliff)

    dn_h = (replication_h - ros_point_damage + transplant_add
            + repair_deletion + repair_point + exercise_biogenesis - apoptosis_h)

    # ── 2. dN_deletion/dt (C11: REVISED from dN_damaged) ────────────────
    replication_del = (base_replication_rate * DELETION_REPLICATION_ADVANTAGE
                       * n_del * nad * energy_available
                       * max(copy_number_pressure, 0.0))

    # De novo deletions from Pol gamma slippage (NOT from ROS)
    age_deletions = del_rate * 0.05 * n_h * energy_available

    # Mitophagy: selective for deletions (low delta-psi -> PINK1, C3)
    mitophagy_del = _current_mitophagy_rate * n_del

    # Apoptosis
    apoptosis_del = (0.02 * max(1.0 - energy_available, 0.0)
                     * n_del * (1.0 - cliff))

    dn_del = (replication_del + age_deletions
              - mitophagy_del - repair_deletion - apoptosis_del
              - transplant_displace)

    # ── 3. dN_point/dt (C11: NEW) ───────────────────────────────────────
    # Point mutations replicate at SAME rate as healthy (no advantage)
    replication_pt = (base_replication_rate * n_pt * nad
                      * energy_available * max(copy_number_pressure, 0.0))

    # New point mutations from Pol gamma errors during healthy replication
    point_from_replication = POINT_ERROR_RATE * replication_h

    # Mitophagy: LOW selectivity for point mutations
    mitophagy_pt = (_current_mitophagy_rate * POINT_MITOPHAGY_SELECTIVITY
                    * n_pt)

    # Apoptosis
    apoptosis_pt = (0.02 * max(1.0 - energy_available, 0.0)
                    * n_pt * (1.0 - cliff))

    dn_pt = (replication_pt + point_from_replication + ros_point_damage
             - mitophagy_pt - repair_point - apoptosis_pt)

    # ── 4. dATP/dt (unchanged logic, uses DELETION cliff) ───────────────
    atp_target = (BASELINE_ATP * cliff
                  * (0.6 + 0.4 * min(nad, 1.0))
                  * (1.0 - 0.15 * sen))
    yama_cost = yama * (0.15 + 0.2 * yama)
    exercise_cost = exercise * 0.03
    atp_target = max(atp_target - yama_cost - exercise_cost, 0.0)
    datp = 1.0 * (atp_target - atp)

    # ── 5. dROS/dt (C11: uses total het, both damage types produce ROS) ─
    ros_baseline = BASELINE_ROS * met_demand
    ros_from_damage = (ROS_PER_DAMAGED * het_total * het_total
                       * (1.0 + inflammation))
    defense_factor = 1.0 + 0.4 * min(nad, 1.0)
    defense_factor += exercise * 0.2
    exercise_ros = exercise * 0.03
    ros_eq = (ros_baseline + ros_from_damage + exercise_ros) / defense_factor
    dros = 1.0 * (ros_eq - ros)

    # ── 6. dNAD/dt (unchanged) ──────────────────────────────────────────
    age_factor = max(1.0 - NAD_DECLINE_RATE * max(age - 30, 0), 0.2)
    nad_boost = nad_supp * 0.25 * cd38_survival
    nad_target = BASELINE_NAD * age_factor + nad_boost
    nad_target = min(nad_target, 1.2)
    ros_drain = 0.03 * ros
    yama_drain = yama * 0.03
    dnad = 0.3 * (nad_target - nad) - ros_drain - yama_drain

    # ── 7. dSenescent/dt (unchanged) ───────────────────────────────────
    energy_stress = max(1.0 - energy_available, 0.0)
    new_sen = (SENESCENCE_RATE * (1.0 + 2.0 * ros + energy_stress)
               * (1.0 + 0.01 * max(age - 40, 0)))
    clearance = seno * 0.2 * sen
    immune_clear = 0.01 * sen * max(1.0 - 0.01 * max(age - 50, 0), 0.1)
    if sen >= 1.0:
        new_sen = 0.0
    dsen = new_sen - clearance - immune_clear

    # ── 8. dMembrane_potential/dt (unchanged) ──────────────────────────
    psi_eq = cliff * min(nad, 1.0) * (1.0 - 0.3 * sen)
    psi_eq = min(psi_eq, BASELINE_MEMBRANE_POTENTIAL)
    dpsi = 0.5 * (psi_eq - psi)

    return np.array([dn_h, dn_del, datp, dros, dnad, dsen, dpsi, dn_pt])


def _rk4_step(
    state: npt.NDArray[np.float64],
    t: float,
    dt: float,
    intervention: dict[str, float],
    patient: dict[str, float],
    tissue_mods: dict[str, float] | None = None,
) -> npt.NDArray[np.float64]:
    """Single 4th-order Runge-Kutta step."""
    k1 = derivatives(state, t, intervention, patient, tissue_mods)
    k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt, intervention, patient, tissue_mods)
    k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt, intervention, patient, tissue_mods)
    k4 = derivatives(state + dt * k3, t + dt, intervention, patient, tissue_mods)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def initial_state(patient: dict[str, float]) -> npt.NDArray[np.float64]:
    """Compute initial state vector from patient parameters.

    Total copy number N_h + N_del + N_pt = 1.0 (normalized). The split between
    deletion and point mutations is age-dependent: older patients have a higher
    fraction of deletions (exponential growth catches up over decades).

    State vector (8D):
        [0] N_healthy, [1] N_deletion, [2] ATP, [3] ROS, [4] NAD,
        [5] Senescent_fraction, [6] Membrane_potential, [7] N_point

    Args:
        patient: Dict with patient parameter values.

    Returns:
        np.array of shape (8,) — initial state.
    """
    het0 = patient["baseline_heteroplasmy"]
    nad0 = patient["baseline_nad_level"]
    met_demand = patient["metabolic_demand"]
    age = patient["baseline_age"]

    # Age-dependent deletion fraction (C11: Cramer Appendix 2)
    age_frac = min(max(age - 20.0, 0.0) / 70.0, 1.0)
    deletion_frac = (DELETION_FRACTION_YOUNG
                     + (DELETION_FRACTION_OLD - DELETION_FRACTION_YOUNG) * age_frac)

    n_h0 = 1.0 - het0
    n_del0 = het0 * deletion_frac
    n_pt0 = het0 * (1.0 - deletion_frac)

    # Deletion heteroplasmy for cliff factor
    het_del0 = _deletion_heteroplasmy(n_h0, n_del0, n_pt0)
    cliff0 = _cliff_factor(het_del0)

    # Senescence
    sen0 = BASELINE_SENESCENT + 0.005 * max(age - 40, 0)
    sen0 = min(sen0, 0.5)

    # ATP: uses deletion cliff
    atp0 = (BASELINE_ATP * cliff0
            * (0.6 + 0.4 * min(nad0, 1.0))
            * (1.0 - 0.15 * sen0))

    # ROS: uses total het
    ros0 = (BASELINE_ROS * met_demand
            + ROS_PER_DAMAGED * het0 * het0) / (1.0 + 0.4 * min(nad0, 1.0))

    # Membrane potential
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
) -> dict:
    """Run the full mitochondrial aging simulation.

    Args:
        intervention: Dict of 6 intervention params (defaults to no treatment).
        patient: Dict of 6 patient params (defaults to typical 70yo).
        sim_years: Override simulation horizon (default: constants.SIM_YEARS).
        dt: Override timestep (default: constants.DT).
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
            "time": np.array of time points (years from start)
            "states": np.array of shape (n_steps+1, 8) — full trajectory
                (or (n_trajectories, n_steps+1, 8) if stochastic with
                n_trajectories > 1)
            "heteroplasmy": np.array — total heteroplasmy at each step
                (or (n_trajectories, n_steps+1) if stochastic)
            "deletion_heteroplasmy": np.array — deletion-only heteroplasmy
                (drives cliff factor)
            "intervention": the intervention dict used
            "patient": the patient dict used
            "tissue_type": tissue type used (or None)
    """
    if intervention is None:
        intervention = dict(DEFAULT_INTERVENTION)
    if patient is None:
        patient = dict(DEFAULT_PATIENT)
    if sim_years is None:
        sim_years = SIM_YEARS
    if dt is None:
        dt = DT

    # Apply tissue profile
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

    if stochastic and n_trajectories > 1:
        return _simulate_stochastic(
            intervention, patient, n_steps, dt, tissue_mods,
            noise_scale, n_trajectories, rng_seed)

    state = initial_state(patient)

    # Pre-allocate output arrays
    time_arr = np.zeros(n_steps + 1)
    states = np.zeros((n_steps + 1, N_STATES))
    het_arr = np.zeros(n_steps + 1)
    del_het_arr = np.zeros(n_steps + 1)

    # Record initial conditions
    states[0] = state
    het_arr[0] = _total_heteroplasmy(state[0], state[1], state[7])
    del_het_arr[0] = _deletion_heteroplasmy(state[0], state[1], state[7])

    if stochastic:
        rng = np.random.default_rng(rng_seed)

    for i in range(n_steps):
        t = i * dt
        current_intervention = _resolve_intervention(intervention, t)
        if stochastic:
            # Euler-Maruyama: deterministic step + noise
            deriv = derivatives(state, t, current_intervention, patient, tissue_mods)
            # Additive noise on ROS (index 3) and damage rate (index 1)
            dW = rng.normal(0, np.sqrt(dt), N_STATES)
            noise = np.zeros(N_STATES)
            noise[1] = noise_scale * state[1] * dW[1]  # deletion noise
            noise[3] = noise_scale * state[3] * dW[3]  # ROS noise
            noise[7] = noise_scale * state[7] * dW[7]  # point mutation noise
            state = state + dt * deriv + noise
        else:
            state = _rk4_step(state, t, dt, current_intervention, patient, tissue_mods)

        # Enforce non-negativity (biological constraint)
        state = np.maximum(state, 0.0)
        # Cap senescent fraction at 1.0
        state[5] = min(state[5], 1.0)

        time_arr[i + 1] = (i + 1) * dt
        states[i + 1] = state
        het_arr[i + 1] = _total_heteroplasmy(state[0], state[1], state[7])
        del_het_arr[i + 1] = _deletion_heteroplasmy(state[0], state[1], state[7])

    return {
        "time": time_arr,
        "states": states,
        "heteroplasmy": het_arr,
        "deletion_heteroplasmy": del_het_arr,
        "intervention": intervention,
        "patient": patient,
        "tissue_type": tissue_type,
    }


def _simulate_stochastic(intervention, patient, n_steps, dt, tissue_mods,
                          noise_scale, n_trajectories, rng_seed):
    """Run multiple stochastic trajectories for confidence intervals.

    Uses Euler-Maruyama integration with multiplicative noise on ROS
    and damage accumulation.

    Returns dict with arrays of shape (n_trajectories, n_steps+1, ...).
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
            current_intervention = _resolve_intervention(intervention, t)
            deriv = derivatives(state, t, current_intervention, patient, tissue_mods)
            dW = rng.normal(0, np.sqrt(dt), N_STATES)
            noise = np.zeros(N_STATES)
            noise[1] = noise_scale * state[1] * dW[1]
            noise[3] = noise_scale * state[3] * dW[3]
            noise[7] = noise_scale * state[7] * dW[7]
            state = state + dt * deriv + noise
            state = np.maximum(state, 0.0)
            state[5] = min(state[5], 1.0)

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
