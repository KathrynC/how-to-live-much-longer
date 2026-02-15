#!/usr/bin/env python3
"""
sobol_sensitivity.py

Global sensitivity analysis of the mitochondrial aging ODE using
Saltelli sampling and Sobol indices. Identifies parameter interactions
(second-order effects) that one-at-a-time perturbation (perturbation_probing.py)
cannot capture.

Method:
  1. Generate N(2D+2) parameter samples using Saltelli's scheme
     (D=12 parameters, N=base samples)
  2. Run simulation for each sample
  3. Compute first-order (S1) and total-order (ST) Sobol indices
     for heteroplasmy and ATP outcomes

The Saltelli sampling and Sobol computation are implemented in pure
numpy (no scipy/SALib dependency), matching the project's constraint.

Reference:
    Saltelli, A. (2002). "Making best use of model evaluations to
    compute sensitivity indices." Computer Physics Communications,
    145(2), 280-297.

Scale: N=256 base → 256*(2*12+2) = 6656 sims, ~2-3 min
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from constants import (
    INTERVENTION_PARAMS, PATIENT_PARAMS,
    INTERVENTION_NAMES, PATIENT_NAMES,
)
from simulator import simulate
from analytics import NumpyEncoder


# ── Parameter bounds ────────────────────────────────────────────────────────

def _get_param_bounds():
    """Get lower/upper bounds for all 12 parameters."""
    bounds = []
    names = []
    for name in INTERVENTION_NAMES:
        spec = INTERVENTION_PARAMS[name]
        bounds.append(spec["range"])
        names.append(name)
    for name in PATIENT_NAMES:
        spec = PATIENT_PARAMS[name]
        bounds.append(spec["range"])
        names.append(name)
    return names, np.array(bounds)


# ── Saltelli sampling ──────────────────────────────────────────────────────

def saltelli_sample(n_base, d, rng):
    """Generate Saltelli's sampling matrices for Sobol analysis.

    Produces N(2D+2) samples by constructing matrices A, B, and
    D cross-matrices A_B^(i) and B_A^(i).

    Args:
        n_base: Number of base samples (N).
        d: Number of parameters (D).
        rng: numpy random generator.

    Returns:
        np.array of shape (N*(2D+2), D) — all parameter samples in [0,1].
    """
    # Two independent quasi-random matrices
    A = rng.random((n_base, d))
    B = rng.random((n_base, d))

    samples = [A, B]  # 2N samples

    # Cross-matrices for first-order indices
    for i in range(d):
        AB_i = B.copy()
        AB_i[:, i] = A[:, i]  # column i from A, rest from B
        samples.append(AB_i)

    # Cross-matrices for total-order indices
    for i in range(d):
        BA_i = A.copy()
        BA_i[:, i] = B[:, i]  # column i from B, rest from A
        samples.append(BA_i)

    return np.vstack(samples)


def rescale_samples(samples_01, bounds):
    """Rescale samples from [0,1] to actual parameter ranges."""
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return lo + samples_01 * (hi - lo)


# ── Sobol index computation ────────────────────────────────────────────────

def sobol_indices(y_A, y_B, y_AB, y_BA):
    """Compute first-order (S1) and total-order (ST) Sobol indices.

    Args:
        y_A: Model output for matrix A, shape (N,).
        y_B: Model output for matrix B, shape (N,).
        y_AB: Model output for AB cross-matrices, shape (D, N).
        y_BA: Model output for BA cross-matrices, shape (D, N).

    Returns:
        (S1, ST) — arrays of shape (D,).
    """
    n = len(y_A)
    f0 = np.mean(np.concatenate([y_A, y_B]))
    var_total = np.var(np.concatenate([y_A, y_B]))

    if var_total < 1e-12:
        d = y_AB.shape[0]
        return np.zeros(d), np.zeros(d)

    d = y_AB.shape[0]
    S1 = np.zeros(d)
    ST = np.zeros(d)

    for i in range(d):
        # First-order: Jansen (1999) estimator
        V_i = np.mean(y_B * (y_AB[i] - y_A))
        S1[i] = V_i / var_total

        # Total-order: Jansen (1999) estimator
        VT_i = 0.5 * np.mean((y_A - y_BA[i]) ** 2)
        ST[i] = VT_i / var_total

    return S1, ST


# ── Main ───────────────────────────────────────────────────────────────────

def run_sobol(n_base=256, sim_years=30, rng_seed=42):
    """Run Sobol sensitivity analysis.

    Args:
        n_base: Base sample count (total sims = N*(2D+2)).
        sim_years: Simulation horizon.
        rng_seed: Random seed for reproducibility.

    Returns:
        Dict with Sobol indices and metadata.
    """
    param_names, bounds = _get_param_bounds()
    d = len(param_names)
    rng = np.random.default_rng(rng_seed)

    # Generate samples
    n_total = n_base * (2 * d + 2)
    print(f"Sobol sensitivity analysis: {d} parameters, N={n_base}")
    print(f"Total simulations: {n_total}")

    samples_01 = saltelli_sample(n_base, d, rng)
    samples = rescale_samples(samples_01, bounds)

    # Run simulations
    print(f"Running {n_total} simulations...")
    t0 = time.time()

    het_final = np.zeros(n_total)
    atp_final = np.zeros(n_total)

    for idx in range(n_total):
        if idx % 500 == 0 and idx > 0:
            elapsed = time.time() - t0
            rate = idx / elapsed
            remaining = (n_total - idx) / rate
            print(f"  {idx}/{n_total} ({elapsed:.0f}s elapsed, "
                  f"~{remaining:.0f}s remaining)")

        row = samples[idx]
        intervention = {name: float(row[i])
                        for i, name in enumerate(INTERVENTION_NAMES)}
        patient = {name: float(row[len(INTERVENTION_NAMES) + i])
                   for i, name in enumerate(PATIENT_NAMES)}

        result = simulate(intervention=intervention, patient=patient,
                          sim_years=sim_years)
        het_final[idx] = result["heteroplasmy"][-1]
        atp_final[idx] = result["states"][-1, 2]

    elapsed = time.time() - t0
    print(f"Simulations complete: {elapsed:.1f}s "
          f"({n_total / elapsed:.0f} sims/sec)")

    # Extract sub-arrays for Sobol computation
    y_A_het = het_final[:n_base]
    y_B_het = het_final[n_base:2*n_base]
    y_AB_het = het_final[2*n_base:2*n_base + d*n_base].reshape(d, n_base)
    y_BA_het = het_final[2*n_base + d*n_base:].reshape(d, n_base)

    y_A_atp = atp_final[:n_base]
    y_B_atp = atp_final[n_base:2*n_base]
    y_AB_atp = atp_final[2*n_base:2*n_base + d*n_base].reshape(d, n_base)
    y_BA_atp = atp_final[2*n_base + d*n_base:].reshape(d, n_base)

    # Compute indices
    S1_het, ST_het = sobol_indices(y_A_het, y_B_het, y_AB_het, y_BA_het)
    S1_atp, ST_atp = sobol_indices(y_A_atp, y_B_atp, y_AB_atp, y_BA_atp)

    # Interaction indices: ST - S1 = higher-order contributions
    interaction_het = ST_het - S1_het
    interaction_atp = ST_atp - S1_atp

    # Print results
    print("\n" + "=" * 70)
    print("SOBOL SENSITIVITY INDICES")
    print("=" * 70)

    print("\n  Heteroplasmy (final):")
    print(f"  {'Parameter':30s}  {'S1':>8s}  {'ST':>8s}  {'Interact':>8s}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}")
    for i, name in enumerate(param_names):
        print(f"  {name:30s}  {S1_het[i]:8.4f}  {ST_het[i]:8.4f}  "
              f"{interaction_het[i]:8.4f}")

    print("\n  ATP (final):")
    print(f"  {'Parameter':30s}  {'S1':>8s}  {'ST':>8s}  {'Interact':>8s}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}")
    for i, name in enumerate(param_names):
        print(f"  {name:30s}  {S1_atp[i]:8.4f}  {ST_atp[i]:8.4f}  "
              f"{interaction_atp[i]:8.4f}")

    # Build result
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_base": n_base,
        "n_total_sims": n_total,
        "elapsed_sec": elapsed,
        "sim_years": sim_years,
        "parameter_names": param_names,
        "heteroplasmy": {
            "S1": {name: float(S1_het[i]) for i, name in enumerate(param_names)},
            "ST": {name: float(ST_het[i]) for i, name in enumerate(param_names)},
            "interaction": {name: float(interaction_het[i])
                            for i, name in enumerate(param_names)},
        },
        "atp": {
            "S1": {name: float(S1_atp[i]) for i, name in enumerate(param_names)},
            "ST": {name: float(ST_atp[i]) for i, name in enumerate(param_names)},
            "interaction": {name: float(interaction_atp[i])
                            for i, name in enumerate(param_names)},
        },
        "rankings": {
            "het_most_influential_S1": sorted(
                param_names, key=lambda n: S1_het[param_names.index(n)],
                reverse=True),
            "atp_most_influential_S1": sorted(
                param_names, key=lambda n: S1_atp[param_names.index(n)],
                reverse=True),
            "het_most_interactive": sorted(
                param_names, key=lambda n: interaction_het[param_names.index(n)],
                reverse=True),
            "atp_most_interactive": sorted(
                param_names, key=lambda n: interaction_atp[param_names.index(n)],
                reverse=True),
        },
    }

    return result


if __name__ == "__main__":
    n_base = 256
    if len(sys.argv) > 1:
        n_base = int(sys.argv[1])

    result = run_sobol(n_base=n_base)

    output_path = PROJECT / "artifacts" / "sobol_sensitivity.json"
    with open(output_path, "w") as f:
        json.dump(result, f, cls=NumpyEncoder, indent=2)
    print(f"\nResults saved to {output_path}")
