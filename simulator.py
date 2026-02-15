"""Numpy-only RK4 ODE integrator for mitochondrial aging dynamics.

Simulates 7 state variables over a configurable time horizon using a
4th-order Runge-Kutta method. Models the core biology from Cramer (2025):
ROS-damage vicious cycle, heteroplasmy cliff, age-dependent deletion
doubling, and six intervention mechanisms.

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

import numpy as np

from constants import (
    SIM_YEARS, DT, N_STEPS, N_STATES,
    BASELINE_MTDNA_COPIES, HETEROPLASMY_CLIFF, CLIFF_STEEPNESS,
    DOUBLING_TIME_YOUNG, DOUBLING_TIME_OLD, AGE_TRANSITION,
    BASELINE_ATP, BASELINE_ROS, ROS_PER_DAMAGED,
    BASELINE_NAD, NAD_DECLINE_RATE,
    BASELINE_MEMBRANE_POTENTIAL, BASELINE_SENESCENT, SENESCENCE_RATE,
    YAMANAKA_ENERGY_COST_MIN, YAMANAKA_ENERGY_COST_MAX,
    BASELINE_MITOPHAGY_RATE, DAMAGED_REPLICATION_ADVANTAGE,
    DEFAULT_INTERVENTION, DEFAULT_PATIENT,
)


def _heteroplasmy_fraction(n_healthy, n_damaged):
    """Compute heteroplasmy as fraction of damaged copies."""
    total = n_healthy + n_damaged
    if total < 1e-12:
        return 1.0
    return n_damaged / total


def _cliff_factor(heteroplasmy):
    """Sigmoid cliff function: ATP efficiency drops steeply at the threshold.

    Returns a value in (0, 1) where 1.0 = fully healthy and ~0 = collapsed.
    Uses a logistic sigmoid centered at HETEROPLASMY_CLIFF.
    """
    return 1.0 / (1.0 + np.exp(CLIFF_STEEPNESS * (heteroplasmy - HETEROPLASMY_CLIFF)))


def _deletion_rate(age, genetic_vulnerability):
    """Age-dependent mtDNA deletion rate.

    Before AGE_TRANSITION: slow doubling (11.8 yr).
    After: fast doubling (3.06 yr). Cramer Ch. 4.
    """
    if age < AGE_TRANSITION:
        doubling_time = DOUBLING_TIME_YOUNG
    else:
        doubling_time = DOUBLING_TIME_OLD
    # Rate = ln(2) / doubling_time, scaled by genetic vulnerability
    return (np.log(2) / doubling_time) * genetic_vulnerability


def derivatives(state, t, intervention, patient):
    """Compute time derivatives of all 7 state variables.

    Args:
        state: np.array of shape (7,) — current state.
        t: Current time in years from simulation start.
        intervention: Dict with 6 intervention parameter values.
        patient: Dict with 6 patient parameter values.

    Returns:
        np.array of shape (7,) — derivatives (dstate/dt).
    """
    n_h, n_d, atp, ros, nad, sen, psi = state

    # Prevent negative values in derivative computation
    n_h = max(n_h, 0.0)
    n_d = max(n_d, 0.0)
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

    # Unpack patient
    age = patient["baseline_age"] + t
    gen_vuln = patient["genetic_vulnerability"]
    met_demand = patient["metabolic_demand"]
    inflammation = patient["inflammation_level"]

    # ── Heteroplasmy and cliff ────────────────────────────────────────────
    het = _heteroplasmy_fraction(n_h, n_d)
    cliff = _cliff_factor(het)

    # ── 1. dN_healthy/dt ──────────────────────────────────────────────────
    # Natural replication (homeostatic, targets ~1.0)
    replication_h = 0.1 * (1.0 - n_h) * nad
    # ROS-induced damage converts healthy → damaged
    damage_rate = 0.05 * ros * gen_vuln * n_h
    # Mitophagy doesn't remove healthy copies much
    # Transplant adds healthy copies
    transplant_add = transplant * 0.2  # up to 0.2 copies/year at max dose
    # Yamanaka can repair some damaged back to healthy
    repair_rate = yama * 0.08 * n_d
    dn_h = replication_h - damage_rate + transplant_add + repair_rate

    # ── 2. dN_damaged/dt ──────────────────────────────────────────────────
    # Damaged copies replicate faster (shorter, replication advantage)
    replication_d = 0.1 * DAMAGED_REPLICATION_ADVANTAGE * n_d * nad * 0.5
    # New damage from ROS
    new_damage = damage_rate
    # Age-dependent deletion accumulation
    del_rate = _deletion_rate(age, gen_vuln)
    age_deletions = del_rate * 0.01  # small baseline accumulation
    # Mitophagy clears damaged copies (enhanced by rapamycin)
    mitophagy = (BASELINE_MITOPHAGY_RATE + rapa * 0.15) * n_d
    # Yamanaka repair removes from damaged pool
    dn_d = replication_d + new_damage + age_deletions - mitophagy - repair_rate

    # ── 3. dATP/dt ────────────────────────────────────────────────────────
    # ATP modeled as relaxation toward an equilibrium set by mitochondrial
    # health. The equilibrium is the production capacity; ATP tracks it with
    # a relaxation time constant. This avoids the instability of
    # production-minus-consumption accounting.
    #
    # Production capacity uses a softer formula: arithmetic mean of health
    # indicators rather than multiplicative coupling.
    health_score = (0.4 * cliff + 0.25 * min(nad, 1.0)
                    + 0.2 * min(psi, 1.0) + 0.15 * n_h)
    production_capacity = BASELINE_ATP * health_score * (1.0 - 0.1 * sen)
    # Yamanaka energy cost (significant!)
    yama_cost = yama * (YAMANAKA_ENERGY_COST_MIN +
                        (YAMANAKA_ENERGY_COST_MAX - YAMANAKA_ENERGY_COST_MIN) * yama)
    # Exercise: hormetic boost if below cliff, but costs some energy
    exercise_boost = exercise * 0.1 * cliff
    exercise_cost = exercise * 0.05
    # Net ATP equilibrium target
    atp_target = production_capacity + exercise_boost - yama_cost - exercise_cost
    atp_target = max(atp_target, 0.0)
    # Relaxation toward target (time constant ~2 years)
    datp = 0.5 * (atp_target - atp)

    # ── 4. dROS/dt ────────────────────────────────────────────────────────
    # ROS equilibrium determined by damage state and defenses.
    # Baseline ROS production
    ros_baseline = BASELINE_ROS * met_demand
    # Damaged mitochondria produce excess ROS (vicious cycle!)
    ros_from_damage = ROS_PER_DAMAGED * het * (1.0 + inflammation)
    # Antioxidant defense scales with NAD availability
    defense_factor = 1.0 + 0.3 * min(nad, 1.5)
    # Exercise: mild hormetic ROS burst
    exercise_ros = exercise * 0.05
    # ROS equilibrium target
    ros_eq = (ros_baseline + ros_from_damage + exercise_ros) / defense_factor
    ros_eq = max(ros_eq, 0.0)
    # Relaxation toward equilibrium
    dros = 0.8 * (ros_eq - ros)

    # ── 5. dNAD/dt ────────────────────────────────────────────────────────
    # NAD+ equilibrium target: base level minus age decline plus supplement
    age_factor = max(1.0 - NAD_DECLINE_RATE * max(age - 30, 0), 0.2)
    nad_target = BASELINE_NAD * age_factor + nad_supp * 0.4
    nad_target = min(nad_target, 1.2)  # cap: supplements can't exceed ~120%
    # Consumption by repair processes and ROS
    repair_drain = yama * 0.05 + 0.01 * ros
    # Relaxation toward target
    dnad = 0.3 * (nad_target - nad) - repair_drain

    # ── 6. dSenescent/dt ─────────────────────────────────────────────────
    # New senescence (driven by ROS and age)
    new_sen = SENESCENCE_RATE * (1.0 + ros) * (1.0 + 0.01 * max(age - 40, 0))
    # Senolytic clearance
    clearance = seno * 0.2 * sen
    # Natural immune clearance (declines with age)
    immune_clear = 0.01 * sen * max(1.0 - 0.01 * max(age - 50, 0), 0.1)
    # Cap at 1.0
    if sen >= 1.0:
        new_sen = 0.0
    dsen = new_sen - clearance - immune_clear

    # ── 7. dMembrane_potential/dt ────────────────────────────────────────
    # ΔΨ depends on healthy mitochondrial function, capped at 1.0
    psi_eq = cliff * min(nad, 1.0) * (1.0 - 0.3 * sen)
    psi_eq = min(psi_eq, BASELINE_MEMBRANE_POTENTIAL)
    # Relaxation toward equilibrium
    dpsi = 0.5 * (psi_eq - psi)

    return np.array([dn_h, dn_d, datp, dros, dnad, dsen, dpsi])


def _rk4_step(state, t, dt, intervention, patient):
    """Single 4th-order Runge-Kutta step."""
    k1 = derivatives(state, t, intervention, patient)
    k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt, intervention, patient)
    k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt, intervention, patient)
    k4 = derivatives(state + dt * k3, t + dt, intervention, patient)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def initial_state(patient):
    """Compute initial state vector from patient parameters.

    Args:
        patient: Dict with patient parameter values.

    Returns:
        np.array of shape (7,) — initial state.
    """
    het0 = patient["baseline_heteroplasmy"]
    nad0 = patient["baseline_nad_level"]

    n_h0 = 1.0 - het0      # healthy fraction
    n_d0 = het0             # damaged fraction
    cliff0 = _cliff_factor(het0)
    atp0 = BASELINE_ATP * n_h0 * cliff0 * nad0
    ros0 = BASELINE_ROS + ROS_PER_DAMAGED * het0
    sen0 = BASELINE_SENESCENT + 0.01 * max(patient["baseline_age"] - 40, 0)
    sen0 = min(sen0, 0.5)
    psi0 = cliff0 * nad0

    return np.array([n_h0, n_d0, atp0, ros0, nad0, min(sen0, 1.0), psi0])


def simulate(intervention=None, patient=None, sim_years=None, dt=None):
    """Run the full mitochondrial aging simulation.

    Args:
        intervention: Dict of 6 intervention params (defaults to no treatment).
        patient: Dict of 6 patient params (defaults to typical 70yo).
        sim_years: Override simulation horizon (default: constants.SIM_YEARS).
        dt: Override timestep (default: constants.DT).

    Returns:
        Dict with:
            "time": np.array of time points (years from start)
            "states": np.array of shape (n_steps+1, 7) — full trajectory
            "heteroplasmy": np.array — heteroplasmy fraction at each step
            "intervention": the intervention dict used
            "patient": the patient dict used
    """
    if intervention is None:
        intervention = dict(DEFAULT_INTERVENTION)
    if patient is None:
        patient = dict(DEFAULT_PATIENT)
    if sim_years is None:
        sim_years = SIM_YEARS
    if dt is None:
        dt = DT

    n_steps = int(sim_years / dt)
    state = initial_state(patient)

    # Pre-allocate output arrays
    time_arr = np.zeros(n_steps + 1)
    states = np.zeros((n_steps + 1, N_STATES))
    het_arr = np.zeros(n_steps + 1)

    # Record initial conditions
    states[0] = state
    het_arr[0] = _heteroplasmy_fraction(state[0], state[1])

    for i in range(n_steps):
        t = i * dt
        state = _rk4_step(state, t, dt, intervention, patient)
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
    print("\n--- Test 4: Cliff verification (sweep 0→0.95) ---")
    for het_start in [0.1, 0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]:
        p = dict(DEFAULT_PATIENT)
        p["baseline_heteroplasmy"] = het_start
        r = simulate(patient=p, sim_years=5, dt=0.01)
        print(f"  het={het_start:.2f} → final_ATP={r['states'][-1, 2]:.4f}  "
              f"final_het={r['heteroplasmy'][-1]:.4f}")

    print("\n" + "=" * 70)
    print("All tests completed.")
