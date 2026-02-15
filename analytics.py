"""4-pillar health analytics for mitochondrial aging simulations.

Mirrors the Beer framework from the parent Evolutionary-Robotics project,
adapted for cellular energetics. Four pillars:

    1. Energy    — ATP trajectory, reserves, crisis timing
    2. Damage    — heteroplasmy trajectory, cliff proximity, acceleration
    3. Dynamics  — ROS oscillations, membrane stability, correlations
    4. Intervention — energy cost, heteroplasmy benefit, benefit-cost ratio

Usage:
    from simulator import simulate
    from analytics import compute_all
    result = simulate(intervention, patient)
    analytics = compute_all(result)
"""

import json
import numpy as np

from constants import (
    HETEROPLASMY_CLIFF, DT,
    YAMANAKA_ENERGY_COST_MIN, YAMANAKA_ENERGY_COST_MAX,
)

EPS = 1e-12


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types.

    Converts numpy integers, floats, booleans, and arrays to their
    Python equivalents. All floats are rounded to 6 decimal places.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return round(float(obj), 6)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return self._convert_array(obj)
        return super().default(obj)

    def _convert_array(self, arr):
        flat = arr.tolist()
        return self._round_nested(flat)

    def _round_nested(self, obj):
        if isinstance(obj, float):
            return round(obj, 6)
        if isinstance(obj, list):
            return [self._round_nested(x) for x in obj]
        return obj


# ── Pillar 1: Energy ─────────────────────────────────────────────────────────

def compute_energy(result):
    """Compute energy pillar metrics from simulation result.

    Analyzes the ATP trajectory to characterize the cell's energy state
    over the simulation horizon.

    Args:
        result: Dict from simulate() with "time", "states", etc.

    Returns:
        Dict with energy metrics.
    """
    time = result["time"]
    atp = result["states"][:, 2]
    n_points = len(atp)

    # Basic trajectory stats
    atp_initial = float(atp[0])
    atp_final = float(atp[-1])
    atp_min = float(np.min(atp))
    atp_max = float(np.max(atp))
    atp_mean = float(np.mean(atp))
    atp_std = float(np.std(atp))
    atp_cv = float(atp_std / atp_mean) if atp_mean > EPS else 0.0

    # Reserve ratio: how much headroom above minimum
    reserve_ratio = float((atp_mean - atp_min) / atp_mean) if atp_mean > EPS else 0.0

    # Linear slope of ATP over time (overall trend)
    if n_points > 1:
        coeffs = np.polyfit(time, atp, 1)
        atp_slope = float(coeffs[0])  # MU/day per year
    else:
        atp_slope = 0.0

    # Time to energy crisis: first time ATP drops below 50% of initial
    crisis_threshold = 0.5 * atp_initial
    crisis_indices = np.where(atp < crisis_threshold)[0]
    if len(crisis_indices) > 0:
        time_to_crisis = float(time[crisis_indices[0]])
    else:
        time_to_crisis = float("inf")

    # Terminal ATP decline rate (last 20% of simulation)
    tail_start = int(0.8 * n_points)
    if tail_start < n_points - 1:
        tail_coeffs = np.polyfit(time[tail_start:], atp[tail_start:], 1)
        terminal_slope = float(tail_coeffs[0])
    else:
        terminal_slope = 0.0

    return {
        "atp_initial": atp_initial,
        "atp_final": atp_final,
        "atp_min": atp_min,
        "atp_max": atp_max,
        "atp_mean": atp_mean,
        "atp_cv": atp_cv,
        "reserve_ratio": reserve_ratio,
        "atp_slope": atp_slope,
        "terminal_slope": terminal_slope,
        "time_to_crisis_years": time_to_crisis,
    }


# ── Pillar 2: Damage ─────────────────────────────────────────────────────────

def compute_damage(result):
    """Compute damage pillar metrics from simulation result.

    Analyzes heteroplasmy trajectory and proximity to the cliff.

    Args:
        result: Dict from simulate().

    Returns:
        Dict with damage metrics.
    """
    time = result["time"]
    het = result["heteroplasmy"]
    n_points = len(het)

    het_initial = float(het[0])
    het_final = float(het[-1])
    het_max = float(np.max(het))
    het_min = float(np.min(het))

    # Distance from cliff at start and end
    cliff_distance_initial = float(HETEROPLASMY_CLIFF - het_initial)
    cliff_distance_final = float(HETEROPLASMY_CLIFF - het_final)

    # Time to reach cliff threshold
    cliff_indices = np.where(het >= HETEROPLASMY_CLIFF)[0]
    if len(cliff_indices) > 0:
        time_to_cliff = float(time[cliff_indices[0]])
    else:
        time_to_cliff = float("inf")

    # Heteroplasmy slope (overall)
    if n_points > 1:
        coeffs = np.polyfit(time, het, 1)
        het_slope = float(coeffs[0])
    else:
        het_slope = 0.0

    # Acceleration: second derivative (fit quadratic)
    if n_points > 2:
        coeffs2 = np.polyfit(time, het, 2)
        het_acceleration = float(2.0 * coeffs2[0])  # 2a from ax^2 + bx + c
    else:
        het_acceleration = 0.0

    # Delta heteroplasmy
    delta_het = float(het_final - het_initial)

    # Proportion of time spent above cliff
    frac_above_cliff = float(np.mean(het >= HETEROPLASMY_CLIFF))

    return {
        "het_initial": het_initial,
        "het_final": het_final,
        "het_max": het_max,
        "delta_het": delta_het,
        "het_slope": het_slope,
        "het_acceleration": het_acceleration,
        "cliff_distance_initial": cliff_distance_initial,
        "cliff_distance_final": cliff_distance_final,
        "time_to_cliff_years": time_to_cliff,
        "frac_above_cliff": frac_above_cliff,
    }


# ── Pillar 3: Dynamics ──────────────────────────────────────────────────────

def _fft_peak(signal, dt):
    """Find dominant frequency and amplitude via FFT.

    Args:
        signal: 1D numpy array.
        dt: Timestep.

    Returns:
        (dominant_freq, dominant_amplitude) tuple.
    """
    n = len(signal)
    if n < 4:
        return 0.0, 0.0

    detrended = signal - np.mean(signal)
    fft_vals = np.fft.rfft(detrended)
    freqs = np.fft.rfftfreq(n, d=dt)
    magnitudes = np.abs(fft_vals)

    # Skip DC component (index 0)
    if len(magnitudes) < 2:
        return 0.0, 0.0
    peak_idx = np.argmax(magnitudes[1:]) + 1
    return float(freqs[peak_idx]), float(magnitudes[peak_idx] / n)


def compute_dynamics(result):
    """Compute dynamics pillar metrics from simulation result.

    Analyzes oscillatory behavior, stability, and cross-variable
    correlations in the simulation.

    Args:
        result: Dict from simulate().

    Returns:
        Dict with dynamics metrics.
    """
    time = result["time"]
    states = result["states"]
    het = result["heteroplasmy"]
    dt = time[1] - time[0] if len(time) > 1 else DT

    ros = states[:, 3]
    nad = states[:, 4]
    psi = states[:, 6]  # membrane potential

    # ROS frequency analysis (look for oscillatory damage cycles)
    ros_freq, ros_amplitude = _fft_peak(ros, dt)

    # Membrane potential coefficient of variation (stability measure)
    psi_mean = float(np.mean(psi))
    psi_cv = float(np.std(psi) / psi_mean) if psi_mean > EPS else 0.0

    # NAD slope (is NAD declining, stable, or recovering?)
    if len(time) > 1:
        nad_coeffs = np.polyfit(time, nad, 1)
        nad_slope = float(nad_coeffs[0])
    else:
        nad_slope = 0.0

    # ROS-heteroplasmy correlation (the vicious cycle metric)
    if len(ros) > 1 and np.std(ros) > EPS and np.std(het) > EPS:
        ros_het_corr = float(np.corrcoef(ros, het)[0, 1])
    else:
        ros_het_corr = 0.0

    # ROS-ATP anti-correlation (damage → energy loss)
    atp = states[:, 2]
    if len(ros) > 1 and np.std(ros) > EPS and np.std(atp) > EPS:
        ros_atp_corr = float(np.corrcoef(ros, atp)[0, 1])
    else:
        ros_atp_corr = 0.0

    # Membrane potential trend
    if len(time) > 1:
        psi_coeffs = np.polyfit(time, psi, 1)
        psi_slope = float(psi_coeffs[0])
    else:
        psi_slope = 0.0

    # Senescence rate (late acceleration indicator)
    sen = states[:, 5]
    sen_final = float(sen[-1])
    if len(time) > 1:
        sen_coeffs = np.polyfit(time, sen, 1)
        sen_slope = float(sen_coeffs[0])
    else:
        sen_slope = 0.0

    return {
        "ros_dominant_freq": ros_freq,
        "ros_amplitude": ros_amplitude,
        "membrane_potential_cv": psi_cv,
        "membrane_potential_slope": psi_slope,
        "nad_slope": nad_slope,
        "ros_het_correlation": ros_het_corr,
        "ros_atp_correlation": ros_atp_corr,
        "senescent_final": sen_final,
        "senescent_slope": sen_slope,
    }


# ── Pillar 4: Intervention ──────────────────────────────────────────────────

def compute_intervention(result, baseline_result=None):
    """Compute intervention pillar metrics.

    Evaluates the cost-effectiveness of the intervention by comparing
    against a no-treatment baseline.

    Args:
        result: Dict from simulate() with intervention applied.
        baseline_result: Dict from simulate() with no intervention.
            If None, a baseline simulation is run internally.

    Returns:
        Dict with intervention cost-benefit metrics.
    """
    from simulator import simulate
    from constants import DEFAULT_INTERVENTION

    intervention = result["intervention"]
    patient = result["patient"]

    if baseline_result is None:
        baseline_result = simulate(
            intervention=DEFAULT_INTERVENTION,
            patient=patient,
        )

    # Terminal values
    atp_final = float(result["states"][-1, 2])
    atp_baseline = float(baseline_result["states"][-1, 2])
    het_final = float(result["heteroplasmy"][-1])
    het_baseline = float(baseline_result["heteroplasmy"][-1])

    # Benefits
    atp_benefit = atp_final - atp_baseline
    het_benefit = het_baseline - het_final  # positive = less damage

    # Mean ATP over simulation
    atp_mean = float(np.mean(result["states"][:, 2]))
    atp_mean_baseline = float(np.mean(baseline_result["states"][:, 2]))
    atp_mean_benefit = atp_mean - atp_mean_baseline

    # Energy cost estimate (Yamanaka is the big one)
    yama = intervention.get("yamanaka_intensity", 0.0)
    yamanaka_cost = yama * (YAMANAKA_ENERGY_COST_MIN +
                            (YAMANAKA_ENERGY_COST_MAX - YAMANAKA_ENERGY_COST_MIN) * yama)
    # Other interventions have metabolic costs too
    exercise_cost = intervention.get("exercise_level", 0.0) * 0.05
    total_energy_cost = yamanaka_cost + exercise_cost

    # Benefit-cost ratio
    if total_energy_cost > EPS:
        benefit_cost_ratio = float(atp_benefit / total_energy_cost)
    else:
        benefit_cost_ratio = float("inf") if atp_benefit > 0 else 0.0

    # Intervention intensity (sum of all doses, crude measure)
    total_dose = sum(intervention.get(k, 0.0) for k in intervention)

    # Time-to-crisis comparison
    crisis_thresh = 0.5 * result["states"][0, 2]
    crisis_idx = np.where(result["states"][:, 2] < crisis_thresh)[0]
    crisis_time = float(result["time"][crisis_idx[0]]) if len(crisis_idx) > 0 else float("inf")

    crisis_idx_base = np.where(baseline_result["states"][:, 2] < crisis_thresh)[0]
    crisis_time_base = (float(baseline_result["time"][crisis_idx_base[0]])
                        if len(crisis_idx_base) > 0 else float("inf"))

    # Crisis delay: how much longer until crisis with vs without intervention
    if np.isinf(crisis_time) and np.isinf(crisis_time_base):
        crisis_delay = 0.0  # neither reaches crisis
    elif np.isinf(crisis_time):
        crisis_delay = 999.0  # intervention prevents crisis entirely
    else:
        crisis_delay = crisis_time - crisis_time_base

    return {
        "atp_benefit_terminal": float(atp_benefit),
        "atp_benefit_mean": float(atp_mean_benefit),
        "het_benefit_terminal": float(het_benefit),
        "energy_cost_per_year": float(total_energy_cost),
        "benefit_cost_ratio": float(benefit_cost_ratio) if not np.isinf(benefit_cost_ratio) else 999.0,
        "total_dose": float(total_dose),
        "crisis_delay_years": float(crisis_delay) if not np.isinf(crisis_delay) else 999.0,
    }


# ── Combined analytics ──────────────────────────────────────────────────────

def compute_all(result, baseline_result=None):
    """Compute all 4 pillars and return as a single analytics dict.

    Args:
        result: Dict from simulate().
        baseline_result: Optional no-treatment baseline for intervention pillar.

    Returns:
        Dict with keys "energy", "damage", "dynamics", "intervention".
    """
    return {
        "energy": compute_energy(result),
        "damage": compute_damage(result),
        "dynamics": compute_dynamics(result),
        "intervention": compute_intervention(result, baseline_result),
    }


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from simulator import simulate
    from constants import DEFAULT_INTERVENTION

    print("=" * 70)
    print("4-Pillar Analytics — Standalone Test")
    print("=" * 70)

    # Baseline (no treatment)
    baseline = simulate()
    # With treatment
    cocktail = {
        "rapamycin_dose": 0.5,
        "nad_supplement": 0.75,
        "senolytic_dose": 0.5,
        "yamanaka_intensity": 0.0,
        "transplant_rate": 0.0,
        "exercise_level": 0.5,
    }
    treated = simulate(intervention=cocktail)

    analytics = compute_all(treated, baseline)

    print("\nPillar 1 — Energy:")
    for k, v in analytics["energy"].items():
        print(f"  {k:30s} = {v}")

    print("\nPillar 2 — Damage:")
    for k, v in analytics["damage"].items():
        print(f"  {k:30s} = {v}")

    print("\nPillar 3 — Dynamics:")
    for k, v in analytics["dynamics"].items():
        print(f"  {k:30s} = {v}")

    print("\nPillar 4 — Intervention:")
    for k, v in analytics["intervention"].items():
        print(f"  {k:30s} = {v}")

    # Test JSON serialization
    print("\nJSON serialization test:")
    json_str = json.dumps(analytics, cls=NumpyEncoder, indent=2)
    print(f"  JSON length: {len(json_str)} chars")
    print(f"  First 200 chars: {json_str[:200]}...")

    print("\n" + "=" * 70)
    print("Analytics tests completed.")
