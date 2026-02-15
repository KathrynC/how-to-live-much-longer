"""Heteroplasmy threshold mapping and cliff characterization.

Maps the nonlinear relationship between heteroplasmy and ATP collapse
(the "cliff" from Cramer 2025). Provides:
    - 1D sweeps of baseline heteroplasmy vs terminal ATP
    - Bisection search for precise cliff edge location
    - Intervention-dependent cliff shift analysis
    - 2D heatmaps (heteroplasmy × age, heteroplasmy × rapamycin)
    - Cliff feature extraction (threshold, sharpness, width, asymmetry)

Usage:
    python cliff_mapping.py
"""

import numpy as np
from simulator import simulate
from constants import (
    DEFAULT_INTERVENTION, DEFAULT_PATIENT,
    HETEROPLASMY_CLIFF, SIM_YEARS,
)


# ── 1D sweep ─────────────────────────────────────────────────────────────────

def sweep_heteroplasmy(n_points=50, sim_years=None, intervention=None,
                       patient_base=None):
    """Sweep baseline heteroplasmy from 0 to 0.95, record terminal ATP.

    Args:
        n_points: Number of heteroplasmy values to test.
        sim_years: Simulation horizon (default: constants.SIM_YEARS).
        intervention: Intervention dict (default: no treatment).
        patient_base: Base patient dict (heteroplasmy will be overridden).

    Returns:
        Dict with "het_values" and "terminal_atp" arrays.
    """
    if intervention is None:
        intervention = dict(DEFAULT_INTERVENTION)
    if patient_base is None:
        patient_base = dict(DEFAULT_PATIENT)

    het_values = np.linspace(0.0, 0.95, n_points)
    terminal_atp = np.zeros(n_points)

    for i, het in enumerate(het_values):
        patient = dict(patient_base)
        patient["baseline_heteroplasmy"] = float(het)
        result = simulate(intervention=intervention, patient=patient,
                          sim_years=sim_years)
        terminal_atp[i] = result["states"][-1, 2]

    return {
        "het_values": het_values,
        "terminal_atp": terminal_atp,
    }


# ── Bisection search for cliff edge ─────────────────────────────────────────

def find_cliff_edge(threshold_atp_frac=0.5, tol=0.001, max_iter=50,
                    intervention=None, patient_base=None, sim_years=None):
    """Find the heteroplasmy value where ATP drops below a threshold.

    Uses bisection search to locate the cliff edge to `tol` precision.

    Args:
        threshold_atp_frac: Fraction of max ATP that defines "crisis"
            (default 0.5 = ATP drops to 50% of maximum).
        tol: Precision of the heteroplasmy value (default 0.001 = 3 decimals).
        max_iter: Maximum bisection iterations.
        intervention: Intervention dict.
        patient_base: Base patient dict.
        sim_years: Simulation horizon.

    Returns:
        Dict with cliff edge heteroplasmy, ATP at edge, and search history.
    """
    if intervention is None:
        intervention = dict(DEFAULT_INTERVENTION)
    if patient_base is None:
        patient_base = dict(DEFAULT_PATIENT)

    # First, find reference ATP at low heteroplasmy
    patient_ref = dict(patient_base)
    patient_ref["baseline_heteroplasmy"] = 0.05
    ref_result = simulate(intervention=intervention, patient=patient_ref,
                          sim_years=sim_years)
    ref_atp = ref_result["states"][-1, 2]
    crisis_atp = threshold_atp_frac * ref_atp

    lo, hi = 0.0, 0.95
    history = []

    for iteration in range(max_iter):
        mid = (lo + hi) / 2.0
        patient = dict(patient_base)
        patient["baseline_heteroplasmy"] = float(mid)
        result = simulate(intervention=intervention, patient=patient,
                          sim_years=sim_years)
        atp = result["states"][-1, 2]

        history.append({"het": float(mid), "atp": float(atp),
                        "iteration": iteration})

        if atp > crisis_atp:
            lo = mid  # cliff is higher
        else:
            hi = mid  # cliff is lower

        if hi - lo < tol:
            break

    edge_het = (lo + hi) / 2.0
    # Get final ATP at the edge
    patient_edge = dict(patient_base)
    patient_edge["baseline_heteroplasmy"] = float(edge_het)
    edge_result = simulate(intervention=intervention, patient=patient_edge,
                           sim_years=sim_years)
    edge_atp = edge_result["states"][-1, 2]

    return {
        "cliff_edge_het": float(edge_het),
        "cliff_edge_atp": float(edge_atp),
        "reference_atp": float(ref_atp),
        "crisis_threshold_atp": float(crisis_atp),
        "iterations": len(history),
        "precision": float(hi - lo),
        "history": history,
    }


# ── Intervention-dependent cliff shift ───────────────────────────────────────

def cliff_shift_analysis(patient_base=None, sim_years=None):
    """Compare cliff edge location under different interventions.

    Tests: no intervention, rapamycin-only, NAD-only, full cocktail, Yamanaka.

    Returns:
        Dict mapping intervention name → cliff edge result.
    """
    if patient_base is None:
        patient_base = dict(DEFAULT_PATIENT)

    interventions = {
        "no_intervention": dict(DEFAULT_INTERVENTION),
        "rapamycin_only": {
            **DEFAULT_INTERVENTION,
            "rapamycin_dose": 0.5,
        },
        "nad_only": {
            **DEFAULT_INTERVENTION,
            "nad_supplement": 0.75,
        },
        "full_cocktail": {
            "rapamycin_dose": 0.5,
            "nad_supplement": 0.75,
            "senolytic_dose": 0.5,
            "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0,
            "exercise_level": 0.5,
        },
        "yamanaka": {
            **DEFAULT_INTERVENTION,
            "yamanaka_intensity": 0.25,
        },
    }

    results = {}
    for name, intervention in interventions.items():
        print(f"  Finding cliff edge for: {name}...")
        edge = find_cliff_edge(intervention=intervention,
                               patient_base=patient_base,
                               sim_years=sim_years)
        results[name] = edge
        print(f"    → cliff at het={edge['cliff_edge_het']:.3f}, "
              f"ATP={edge['cliff_edge_atp']:.4f}")

    return results


# ── 2D heatmaps ─────────────────────────────────────────────────────────────

def heatmap_het_age(n_het=20, n_age=15, intervention=None, sim_years=None):
    """2D heatmap: heteroplasmy × starting age → terminal ATP.

    Args:
        n_het: Number of heteroplasmy grid points.
        n_age: Number of age grid points.
        intervention: Intervention dict.
        sim_years: Simulation horizon.

    Returns:
        Dict with "het_axis", "age_axis", "atp_grid" (2D array).
    """
    if intervention is None:
        intervention = dict(DEFAULT_INTERVENTION)

    het_axis = np.linspace(0.05, 0.9, n_het)
    age_axis = np.linspace(30.0, 85.0, n_age)
    atp_grid = np.zeros((n_het, n_age))

    for i, het in enumerate(het_axis):
        for j, age in enumerate(age_axis):
            patient = dict(DEFAULT_PATIENT)
            patient["baseline_heteroplasmy"] = float(het)
            patient["baseline_age"] = float(age)
            # Adjust NAD for age
            patient["baseline_nad_level"] = max(0.2, 1.0 - 0.01 * max(age - 30, 0))
            result = simulate(intervention=intervention, patient=patient,
                              sim_years=sim_years)
            atp_grid[i, j] = result["states"][-1, 2]

    return {
        "het_axis": het_axis,
        "age_axis": age_axis,
        "atp_grid": atp_grid,
    }


def heatmap_het_rapamycin(n_het=20, n_rapa=15, patient_base=None,
                          sim_years=None):
    """2D heatmap: heteroplasmy × rapamycin dose → terminal ATP.

    Args:
        n_het: Number of heteroplasmy grid points.
        n_rapa: Number of rapamycin dose grid points.
        patient_base: Base patient dict.
        sim_years: Simulation horizon.

    Returns:
        Dict with "het_axis", "rapa_axis", "atp_grid" (2D array).
    """
    if patient_base is None:
        patient_base = dict(DEFAULT_PATIENT)

    het_axis = np.linspace(0.05, 0.9, n_het)
    rapa_axis = np.linspace(0.0, 1.0, n_rapa)
    atp_grid = np.zeros((n_het, n_rapa))

    for i, het in enumerate(het_axis):
        for j, rapa in enumerate(rapa_axis):
            patient = dict(patient_base)
            patient["baseline_heteroplasmy"] = float(het)
            intervention = dict(DEFAULT_INTERVENTION)
            intervention["rapamycin_dose"] = float(rapa)
            result = simulate(intervention=intervention, patient=patient,
                              sim_years=sim_years)
            atp_grid[i, j] = result["states"][-1, 2]

    return {
        "het_axis": het_axis,
        "rapa_axis": rapa_axis,
        "atp_grid": atp_grid,
    }


# ── Cliff feature extraction ────────────────────────────────────────────────

def extract_cliff_features(sweep_result):
    """Extract cliff features from a 1D heteroplasmy sweep.

    Args:
        sweep_result: Dict from sweep_heteroplasmy() with "het_values"
            and "terminal_atp".

    Returns:
        Dict with:
            threshold: heteroplasmy at maximum ATP decline rate
            sharpness: maximum |dATP/dHet| (steepness of cliff)
            width: heteroplasmy range over which 80% of ATP drop occurs
            asymmetry: ratio of pre-cliff slope to post-cliff slope
    """
    het = sweep_result["het_values"]
    atp = sweep_result["terminal_atp"]

    if len(het) < 3:
        return {"threshold": 0.0, "sharpness": 0.0, "width": 0.0,
                "asymmetry": 1.0}

    # Numerical gradient
    d_het = np.diff(het)
    d_atp = np.diff(atp)
    gradient = d_atp / (d_het + 1e-12)

    # Threshold: heteroplasmy at steepest decline
    steepest_idx = np.argmin(gradient)  # most negative gradient
    threshold = float(het[steepest_idx])
    sharpness = float(abs(gradient[steepest_idx]))

    # Width: range over which 80% of total ATP drop occurs
    atp_max = np.max(atp)
    atp_min = np.min(atp)
    total_drop = atp_max - atp_min
    if total_drop < 1e-12:
        return {"threshold": threshold, "sharpness": sharpness,
                "width": 0.0, "asymmetry": 1.0}

    # Find het values where ATP is at 90% and 10% of the drop
    target_high = atp_max - 0.1 * total_drop
    target_low = atp_max - 0.9 * total_drop

    het_high_idx = np.argmin(np.abs(atp - target_high))
    het_low_idx = np.argmin(np.abs(atp - target_low))
    width = float(abs(het[het_low_idx] - het[het_high_idx]))

    # Asymmetry: compare average slope before vs after threshold
    pre_cliff = gradient[:steepest_idx] if steepest_idx > 0 else gradient[:1]
    post_cliff = gradient[steepest_idx:] if steepest_idx < len(gradient) - 1 else gradient[-1:]
    pre_slope = float(abs(np.mean(pre_cliff)))
    post_slope = float(abs(np.mean(post_cliff)))
    asymmetry = float(pre_slope / post_slope) if post_slope > 1e-12 else 0.0

    return {
        "threshold": threshold,
        "sharpness": sharpness,
        "width": width,
        "asymmetry": asymmetry,
    }


# ── Standalone execution ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Heteroplasmy Cliff Mapping")
    print("=" * 70)

    # 1D sweep
    print("\n--- 1D Sweep: heteroplasmy 0→0.95, 50 points ---")
    sweep = sweep_heteroplasmy(n_points=50, sim_years=10)
    print(f"  Het range: {sweep['het_values'][0]:.2f} → {sweep['het_values'][-1]:.2f}")
    print(f"  ATP range: {sweep['terminal_atp'].max():.4f} → {sweep['terminal_atp'].min():.4f}")

    # Cliff features
    features = extract_cliff_features(sweep)
    print(f"\n--- Cliff Features ---")
    for k, v in features.items():
        print(f"  {k:15s} = {v:.4f}")

    # Bisection search
    print("\n--- Bisection Search for Cliff Edge ---")
    edge = find_cliff_edge(sim_years=10)
    print(f"  Cliff edge at het = {edge['cliff_edge_het']:.3f}")
    print(f"  ATP at edge = {edge['cliff_edge_atp']:.4f}")
    print(f"  Reference ATP = {edge['reference_atp']:.4f}")
    print(f"  Precision = {edge['precision']:.6f}")
    print(f"  Iterations = {edge['iterations']}")

    # Intervention-dependent cliff shift
    print("\n--- Cliff Shift Under Different Interventions ---")
    shifts = cliff_shift_analysis(sim_years=10)
    print("\n  Summary:")
    for name, result in shifts.items():
        print(f"    {name:25s} → cliff at het={result['cliff_edge_het']:.3f}")

    # 2D heatmaps (reduced resolution for speed)
    print("\n--- 2D Heatmap: heteroplasmy × age (10×8 grid) ---")
    hm_age = heatmap_het_age(n_het=10, n_age=8, sim_years=10)
    print(f"  ATP grid shape: {hm_age['atp_grid'].shape}")
    print(f"  ATP range: [{hm_age['atp_grid'].min():.4f}, {hm_age['atp_grid'].max():.4f}]")

    print("\n--- 2D Heatmap: heteroplasmy × rapamycin (10×8 grid) ---")
    hm_rapa = heatmap_het_rapamycin(n_het=10, n_rapa=8, sim_years=10)
    print(f"  ATP grid shape: {hm_rapa['atp_grid'].shape}")
    print(f"  ATP range: [{hm_rapa['atp_grid'].min():.4f}, {hm_rapa['atp_grid'].max():.4f}]")

    print("\n" + "=" * 70)
    print("Cliff mapping completed.")
