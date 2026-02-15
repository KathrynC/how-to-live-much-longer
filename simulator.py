"""Numpy-only RK4 ODE integrator for mitochondrial aging dynamics.

Simulates 7 state variables over a configurable time horizon using a
4th-order Runge-Kutta method. Models the core biology from Cramer (2025):
ROS-damage vicious cycle, heteroplasmy cliff, age-dependent deletion
doubling, and six intervention mechanisms.

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
    1. N_damaged          — damaged mtDNA copies (normalized)
    2. ATP                — ATP production rate (MU/day)
    3. ROS                — reactive oxygen species level
    4. NAD                — NAD+ availability
    5. Senescent_fraction — fraction of senescent cells
    6. Membrane_potential — mitochondrial ΔΨ (normalized)

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
    BASELINE_MITOPHAGY_RATE, DAMAGED_REPLICATION_ADVANTAGE,
    CD38_BASE_SURVIVAL, CD38_SUPPRESSION_GAIN,
    TRANSPLANT_ADDITION_RATE, TRANSPLANT_DISPLACEMENT_RATE, TRANSPLANT_HEADROOM,
    DEFAULT_INTERVENTION, DEFAULT_PATIENT,
    TISSUE_PROFILES,
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
    """Compute heteroplasmy as fraction of damaged copies."""
    total = n_healthy + n_damaged
    if total < 1e-12:
        return 1.0
    return n_damaged / total


def _cliff_factor(heteroplasmy: float) -> float:
    """Sigmoid cliff function: ATP efficiency drops steeply at the threshold.

    Returns a value in (0, 1) where 1.0 = fully healthy and ~0 = collapsed.
    Uses a logistic sigmoid centered at HETEROPLASMY_CLIFF.
    """
    return 1.0 / (1.0 + np.exp(CLIFF_STEEPNESS * (heteroplasmy - HETEROPLASMY_CLIFF)))


def _deletion_rate(age: float, genetic_vulnerability: float) -> float:
    """Age-dependent mtDNA deletion rate.

    Cramer Appendix 2, p.155, Fig. 23 (data from Va23: Vandiver et al.,
    Aging Cell 22(6), 2023): DT = 11.81 yr before age 65, 3.06 yr after.
    Also Ch. II.H, p.15: "deletion damage builds exponentially."
    Corrected 2026-02-15: AGE_TRANSITION=65 per Cramer email.
    """
    if age < AGE_TRANSITION:
        doubling_time = DOUBLING_TIME_YOUNG
    else:
        doubling_time = DOUBLING_TIME_OLD
    # Rate = ln(2) / doubling_time, scaled by genetic vulnerability
    return (np.log(2) / doubling_time) * genetic_vulnerability


def derivatives(
    state: npt.NDArray[np.float64],
    t: float,
    intervention: dict[str, float],
    patient: dict[str, float],
    tissue_mods: dict[str, float] | None = None,
) -> npt.NDArray[np.float64]:
    """Compute time derivatives of all 7 state variables.

    Fixes applied 2026-02-15 per falsifier report:
      C1: Cliff now feeds back into replication and apoptosis
      C2: Total copy number (N_h + N_d) regulated toward 1.0
      C3: NAD supplementation selectively benefits healthy mitochondria
      C4: Damaged replication advantage creates bistability past cliff
      M1: Yamanaka repair gated by available ATP
      M2: Removed spurious 0.5 factor on damaged replication
      M3: Strengthened ROS-damage vicious cycle coupling
      M5: Exercise upregulates mitochondrial biogenesis (actual hormesis)

    Corrections applied 2026-02-15 per Cramer email:
      C7: CD38 degrades NMN/NR — NAD+ boost is CD38-gated and reduced
      C8: Transplant is primary rejuvenation — doubled rate + displacement

    Args:
        state: np.array of shape (7,) — current state.
        t: Current time in years from simulation start.
        intervention: Dict with 6 intervention parameter values.
        patient: Dict with 6 patient parameter values.
        tissue_mods: Optional dict with tissue-specific modifiers:
            "ros_sensitivity" (float): ROS damage multiplier (default 1.0)
            "biogenesis_rate" (float): exercise biogenesis multiplier (default 1.0)

    Returns:
        np.array of shape (7,) — derivatives (dstate/dt).
    """
    if tissue_mods is None:
        tissue_mods = _DEFAULT_TISSUE_MODS
    n_h, n_d, atp, ros, nad, sen, psi = state

    # Prevent negative values in derivative computation
    n_h = max(n_h, 1e-6)
    n_d = max(n_d, 1e-6)
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

    # CD38 survival factor (Cramer Ch. VI.A.3 p.73, email 2026-02-15):
    # CD38 enzyme destroys NMN/NR precursors. At low supplementation,
    # only ~40% survives. High doses imply CD38 suppression via apigenin,
    # raising survival toward 100%.
    cd38_survival = CD38_BASE_SURVIVAL + CD38_SUPPRESSION_GAIN * nad_supp

    # Unpack patient
    age = patient["baseline_age"] + t
    gen_vuln = patient["genetic_vulnerability"]
    met_demand = patient["metabolic_demand"]
    inflammation = patient["inflammation_level"]

    # ── Derived quantities ────────────────────────────────────────────────
    total = n_h + n_d
    het = n_d / total                          # heteroplasmy fraction
    cliff = _cliff_factor(het)                 # sigmoid efficiency factor
    atp_norm = min(atp / BASELINE_ATP, 1.0)    # normalized ATP (0-1)
    energy_available = max(atp_norm, 0.05)     # floor to prevent total shutdown

    # ── Copy number regulation ────────────────────────────────────────────
    # Total mtDNA copy number is homeostatically regulated. When total > 1,
    # replication slows; when total < 1, replication speeds up. This prevents
    # unbounded growth (fixes C2/M4).
    copy_number_pressure = max(1.0 - total, -0.5)  # caps downward pressure

    # ── Age-dependent deletion rate ───────────────────────────────────────
    del_rate = _deletion_rate(age, gen_vuln)

    # ── 1. dN_healthy/dt ──────────────────────────────────────────────────
    # Replication: gated by ATP (fix C1), NAD, and copy number pressure.
    # Healthy copies replicate at the base rate.
    base_replication_rate = 0.1
    replication_h = (base_replication_rate * n_h * nad
                     * energy_available * max(copy_number_pressure, 0.0))

    # ROS-induced damage: converts healthy → damaged (the vicious cycle
    # entry point). Cramer Ch. II.H p.14, Appendix 2 pp.152-153.
    # Coupling strength increased (fix M3).
    # Tissue-specific ROS sensitivity (brain=1.5x, muscle=0.8x, etc.)
    damage_rate = 0.15 * ros * gen_vuln * n_h * tissue_mods["ros_sensitivity"]

    # Transplant: adds healthy copies AND displaces damaged ones.
    # Cramer Ch. VIII.G pp.104-107: high-volume mitochondrial transplant
    # via bioreactor-grown stem cells → mitlet encapsulation.
    # This is the ONLY method for actual rejuvenation — reversing
    # accumulated mtDNA damage at scale (Cramer email, 2026-02-15).
    transplant_headroom = max(TRANSPLANT_HEADROOM - total, 0.0)
    transplant_add = transplant * TRANSPLANT_ADDITION_RATE * min(transplant_headroom, 1.0)

    # Competitive displacement: transplanted healthy mitochondria with
    # intact ETC and high ΔΨ outcompete damaged copies for cellular resources.
    transplant_displace = transplant * TRANSPLANT_DISPLACEMENT_RATE * n_d * energy_available

    # Yamanaka repair: converts damaged → healthy. GATED BY ATP (fix M1).
    # Cramer Ch. VII.B pp.92-95: epigenetic reprogramming (OSKMLN).
    # Ch. VIII.A Table 3 p.100: costs ~3-5 MU (Ci24, Fo18).
    # At ATP=0, no repair occurs regardless of intensity.
    repair_rate = yama * 0.06 * n_d * energy_available

    # Exercise hormesis: upregulates mitochondrial biogenesis (fix M5).
    # Moderate exercise increases healthy copy replication via PGC-1α.
    # Tissue-specific biogenesis rate (muscle=1.5x, brain=0.3x, etc.)
    exercise_biogenesis = (exercise * 0.03 * energy_available
                           * max(copy_number_pressure, 0.0)
                           * tissue_mods["biogenesis_rate"])

    # Apoptosis of cells with severe damage (cliff feedback, fix C1):
    # When ATP is very low, cells die, removing both healthy and damaged copies.
    apoptosis_h = 0.02 * max(1.0 - energy_available, 0.0) * n_h * (1.0 - cliff)

    dn_h = (replication_h - damage_rate + transplant_add + repair_rate
            + exercise_biogenesis - apoptosis_h)

    # ── 2. dN_damaged/dt ──────────────────────────────────────────────────
    # Damaged copies replicate FASTER ("replicative advantage").
    # Cramer Appendix 2 pp.154-155: deleted mtDNA (>3kbp) replicates
    # "at least 21% faster" (Va23). Code uses conservative 1.05 (5%).
    # No spurious 0.5 factor (fix M2). Also gated by ATP and copy number.
    replication_d = (base_replication_rate * DAMAGED_REPLICATION_ADVANTAGE
                     * n_d * nad * energy_available
                     * max(copy_number_pressure, 0.0))

    # New damage from ROS (arrives from healthy pool)
    new_damage = damage_rate

    # Age-dependent de novo deletions.
    # Cramer Appendix 2 pp.152-155: deletions arise from Pol γ replication
    # errors, double-strand breaks, and replication slippage.
    # Proportional to healthy pool (deletions occur during replication).
    # Removed the 0.01 suppression factor (fix m6).
    age_deletions = del_rate * 0.05 * n_h * energy_available

    # Mitophagy: selective clearance of damaged copies.
    # Cramer Ch. VI.B p.75: PINK1/Parkin pathway — low ΔΨ → PINK1
    # accumulates → Parkin signals removal.
    # Enhanced by rapamycin (Ch. VI.A.1 pp.71-72: mTOR inhibition).
    # NAD+ supplementation improves mitochondrial quality control —
    # selectively enhances mitophagy of damaged copies rather than
    # boosting damaged replication (fix C3).
    mitophagy_rate = (BASELINE_MITOPHAGY_RATE
                      + rapa * 0.08                    # rapamycin: ~4x boost at max dose
                      + nad_supp * 0.03 * cd38_survival)  # NAD: CD38-gated quality control
    mitophagy = mitophagy_rate * n_d

    # Apoptosis removes damaged copies too
    apoptosis_d = 0.02 * max(1.0 - energy_available, 0.0) * n_d * (1.0 - cliff)

    dn_d = (replication_d + new_damage + age_deletions
            - mitophagy - repair_rate - apoptosis_d - transplant_displace)

    # ── 3. dATP/dt ────────────────────────────────────────────────────────
    # ATP represents the cell's energy production capacity (normalized to
    # 1.0 = healthy baseline). The cliff factor IS the primary driver:
    # when het crosses the threshold, production capacity collapses.
    # This then feeds back into replication shutdown (C1 fix).
    #
    # ATP = cliff × health_modifiers - intervention_costs
    # A healthy person (cliff=1, NAD=1, sen=0) has ATP ≈ 1.0
    # Past the cliff, ATP crashes toward 0.
    atp_target = (BASELINE_ATP * cliff
                  * (0.6 + 0.4 * min(nad, 1.0))    # NAD modulates ±40%
                  * (1.0 - 0.15 * sen))              # senescence burden

    # Yamanaka energy cost (rescaled: 0.15-0.35 MU at max; fix m5)
    yama_cost = yama * (0.15 + 0.2 * yama)

    # Exercise: modest acute energy cost
    exercise_cost = exercise * 0.03

    atp_target = max(atp_target - yama_cost - exercise_cost, 0.0)
    # Relaxation time constant ~1 year
    datp = 1.0 * (atp_target - atp)

    # ── 4. dROS/dt ────────────────────────────────────────────────────────
    # ROS production: baseline + damage-dependent (the vicious cycle).
    # Coupling to heteroplasmy is stronger (fix M3): damaged mitochondria
    # have electron transport chain defects that leak electrons → superoxide.
    ros_baseline = BASELINE_ROS * met_demand
    ros_from_damage = ROS_PER_DAMAGED * het * het * (1.0 + inflammation)
    # ^^ quadratic in het: damage accelerates ROS production nonlinearly

    # Antioxidant defense: NAD-dependent (via sirtuins → SOD2/catalase)
    defense_factor = 1.0 + 0.4 * min(nad, 1.0)
    # Exercise upregulates antioxidant defenses (hormesis: mild ROS → adaptation)
    defense_factor += exercise * 0.2

    # Exercise acute ROS burst
    exercise_ros = exercise * 0.03

    # ROS equilibrium
    ros_eq = (ros_baseline + ros_from_damage + exercise_ros) / defense_factor
    # Relaxation
    dros = 1.0 * (ros_eq - ros)

    # ── 5. dNAD/dt ────────────────────────────────────────────────────────
    # NAD+ declines with age. Cramer Ch. VI.A.3 pp.72-73: NMN/NR
    # supplementation. Ca16 = Camacho-Pereira et al. 2016 (Ch. VI refs p.87).
    # NAD does NOT boost damaged replication (fix C3) — it's consumed
    # by quality control processes (sirtuins, PARPs) that preferentially
    # benefit healthy mitochondria.
    #
    # CD38 degradation (fix C7, Cramer p.73, email 2026-02-15):
    # CD38 enzyme destroys NMN/NR before absorption. Low-dose NMN/NR
    # supplementation is largely futile. High doses imply combined
    # NMN/NR + CD38 suppression (apigenin), improving delivery.
    # Coefficient reduced from 0.35 → 0.25 and gated by cd38_survival.
    age_factor = max(1.0 - NAD_DECLINE_RATE * max(age - 30, 0), 0.2)
    nad_boost = nad_supp * 0.25 * cd38_survival
    nad_target = BASELINE_NAD * age_factor + nad_boost
    nad_target = min(nad_target, 1.2)
    # ROS consumes NAD (via PARP activation from DNA damage)
    ros_drain = 0.03 * ros
    # Yamanaka consumes NAD
    yama_drain = yama * 0.03
    # Relaxation toward target
    dnad = 0.3 * (nad_target - nad) - ros_drain - yama_drain

    # ── 6. dSenescent/dt ─────────────────────────────────────────────────
    # Cramer Ch. VII.A pp.89-90: senescent cells cease dividing, emit SASP.
    # Ch. VIII.F p.103: use ~2x energy, apoptosis costs ~0.5-0.7 MU (Table 3).
    # Senescence triggered by ROS, low ATP, and age.
    # Low energy accelerates senescence (cliff feedback, fix C1).
    energy_stress = max(1.0 - energy_available, 0.0)
    new_sen = (SENESCENCE_RATE * (1.0 + 2.0 * ros + energy_stress)
               * (1.0 + 0.01 * max(age - 40, 0)))
    # Senolytic clearance (Cramer Ch. VII.A.2 p.91: D+Q+F protocol)
    clearance = seno * 0.2 * sen
    # Natural immune clearance (declines with age)
    immune_clear = 0.01 * sen * max(1.0 - 0.01 * max(age - 50, 0), 0.1)
    # Cap
    if sen >= 1.0:
        new_sen = 0.0
    dsen = new_sen - clearance - immune_clear

    # ── 7. dMembrane_potential/dt ────────────────────────────────────────
    # Cramer Ch. IV pp.46-47: proton gradient across inner membrane.
    # Ch. VI.B p.75: low ΔΨ triggers PINK1-mediated mitophagy.
    # ΔΨ is maintained by the electron transport chain. Collapses with
    # the cliff (healthy ETC function required).
    psi_eq = cliff * min(nad, 1.0) * (1.0 - 0.3 * sen)
    psi_eq = min(psi_eq, BASELINE_MEMBRANE_POTENTIAL)
    dpsi = 0.5 * (psi_eq - psi)

    return np.array([dn_h, dn_d, datp, dros, dnad, dsen, dpsi])


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

    Total copy number N_h + N_d = 1.0 (normalized). Initial ATP computed
    from the same formula used in the dynamics (cliff-based production
    minus demand) to avoid transient jumps at t=0.

    Args:
        patient: Dict with patient parameter values.

    Returns:
        np.array of shape (7,) — initial state.
    """
    het0 = patient["baseline_heteroplasmy"]
    nad0 = patient["baseline_nad_level"]
    met_demand = patient["metabolic_demand"]

    n_h0 = 1.0 - het0      # healthy fraction (total = 1.0)
    n_d0 = het0             # damaged fraction
    cliff0 = _cliff_factor(het0)

    # Senescence: accumulates with age (compute before ATP, which depends on it)
    sen0 = BASELINE_SENESCENT + 0.005 * max(patient["baseline_age"] - 40, 0)
    sen0 = min(sen0, 0.5)

    # ATP: consistent with dynamics equilibrium formula
    atp0 = (BASELINE_ATP * cliff0
            * (0.6 + 0.4 * min(nad0, 1.0))
            * (1.0 - 0.15 * sen0))

    # ROS: quadratic in het (matching dynamics)
    ros0 = (BASELINE_ROS * met_demand + ROS_PER_DAMAGED * het0 * het0) / (1.0 + 0.4 * min(nad0, 1.0))

    # Membrane potential
    psi0 = cliff0 * min(nad0, 1.0) * (1.0 - 0.3 * sen0)
    psi0 = min(psi0, BASELINE_MEMBRANE_POTENTIAL)

    return np.array([n_h0, n_d0, atp0, ros0, nad0, min(sen0, 1.0), psi0])


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
            "states": np.array of shape (n_steps+1, 7) — full trajectory
                (or (n_trajectories, n_steps+1, 7) if stochastic with
                n_trajectories > 1)
            "heteroplasmy": np.array — heteroplasmy fraction at each step
                (or (n_trajectories, n_steps+1) if stochastic)
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

    # Record initial conditions
    states[0] = state
    het_arr[0] = _heteroplasmy_fraction(state[0], state[1])

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
            noise[1] = noise_scale * state[1] * dW[1]  # damage noise
            noise[3] = noise_scale * state[3] * dW[3]  # ROS noise
            state = state + dt * deriv + noise
        else:
            state = _rk4_step(state, t, dt, current_intervention, patient, tissue_mods)

        # Enforce non-negativity (biological constraint)
        state = np.maximum(state, 0.0)
        # Cap senescent fraction at 1.0
        state[5] = min(state[5], 1.0)

        time_arr[i + 1] = (i + 1) * dt
        states[i + 1] = state
        het_arr[i + 1] = _heteroplasmy_fraction(state[0], state[1])

    return {
        "time": time_arr,
        "states": states,
        "heteroplasmy": het_arr,
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

    for traj in range(n_trajectories):
        state = state0.copy()
        all_states[traj, 0] = state
        all_het[traj, 0] = _heteroplasmy_fraction(state[0], state[1])

        for i in range(n_steps):
            t = i * dt
            current_intervention = _resolve_intervention(intervention, t)
            deriv = derivatives(state, t, current_intervention, patient, tissue_mods)
            dW = rng.normal(0, np.sqrt(dt), N_STATES)
            noise = np.zeros(N_STATES)
            noise[1] = noise_scale * state[1] * dW[1]
            noise[3] = noise_scale * state[3] * dW[3]
            state = state + dt * deriv + noise
            state = np.maximum(state, 0.0)
            state[5] = min(state[5], 1.0)

            all_states[traj, i + 1] = state
            all_het[traj, i + 1] = _heteroplasmy_fraction(state[0], state[1])

    return {
        "time": time_arr,
        "states": all_states,
        "heteroplasmy": all_het,
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
    for j, name in enumerate(["N_healthy", "N_damaged", "ATP", "ROS",
                               "NAD", "Senescent", "ΔΨ"]):
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
    for j, name in enumerate(["N_healthy", "N_damaged", "ATP", "ROS",
                               "NAD", "Senescent", "ΔΨ"]):
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
              f"N_total={r['states'][-1, 0] + r['states'][-1, 1]:.3f}")

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
          f"N_total={r5d['states'][-1, 0] + r5d['states'][-1, 1]:.3f}")

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
