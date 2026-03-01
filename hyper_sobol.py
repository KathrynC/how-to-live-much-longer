#!/usr/bin/env python3
"""
Hyper-Sobol sensitivity analysis of rule confidence parameters.

Extends Sobol global sensitivity analysis to the 45 rule confidence parameters
in the mitochondrial semantic CA. Uses Saltelli sampling and computes first-order
(S1) and total-order (ST) Sobol indices for CA outcome metrics.

Identifies which rules (and combinations) most influence cellular fate, and
links hypergraph motifs (variable groups) to interaction strengths.

Phase 2 of the hypergraph analysis pipeline.
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter
import itertools

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from rule_confidence_simulator import RuleConfidenceSimulator
from analytics import NumpyEncoder
from sobol_sensitivity import saltelli_sample, sobol_indices

# ── Variable motifs extraction ────────────────────────────────────────────────

def extract_variable_motifs(rules_path="final_tuned_rules.json"):
    """Extract variable pairs/triplets/quadruplets that co-occur in rules."""
    with open(rules_path, "r") as f:
        rules = json.load(f)
    
    state_vars = {
        "N_healthy", "N_deletion", "ATP", "ROS",
        "NAD", "Senescent_fraction", "Membrane_potential", "N_point"
    }
    
    pair_counts = Counter()
    triplet_counts = Counter()
    quadruplet_counts = Counter()
    
    for rule in rules:
        vars_in_rule = set()
        for var_name in rule.get("inputs", {}):
            if var_name in state_vars:
                vars_in_rule.add(var_name)
        for var_name in rule.get("outputs", {}):
            if var_name in state_vars:
                vars_in_rule.add(var_name)
        
        if len(vars_in_rule) >= 2:
            for pair in itertools.combinations(sorted(vars_in_rule), 2):
                pair_counts[pair] += 1
        if len(vars_in_rule) >= 3:
            for triplet in itertools.combinations(sorted(vars_in_rule), 3):
                triplet_counts[triplet] += 1
        if len(vars_in_rule) >= 4:
            for quad in itertools.combinations(sorted(vars_in_rule), 4):
                quadruplet_counts[quad] += 1
    
    # Keep motifs that appear in at least 2 rules (shared structure)
    motifs = {
        "pairs": [{"variables": list(pair), "count": cnt}
                  for pair, cnt in pair_counts.items() if cnt >= 2],
        "triplets": [{"variables": list(trip), "count": cnt}
                     for trip, cnt in triplet_counts.items() if cnt >= 2],
        "quadruplets": [{"variables": list(quad), "count": cnt}
                        for quad, cnt in quadruplet_counts.items() if cnt >= 2],
    }
    return motifs

def map_rules_to_motifs(rules_path="final_tuned_rules.json"):
    """Return dict: motif_key -> list of rule names that contain that motif."""
    with open(rules_path, "r") as f:
        rules = json.load(f)
    
    state_vars = {
        "N_healthy", "N_deletion", "ATP", "ROS",
        "NAD", "Senescent_fraction", "Membrane_potential", "N_point"
    }
    
    motif_to_rules = defaultdict(list)
    
    for rule in rules:
        vars_in_rule = set()
        for var_name in rule.get("inputs", {}):
            if var_name in state_vars:
                vars_in_rule.add(var_name)
        for var_name in rule.get("outputs", {}):
            if var_name in state_vars:
                vars_in_rule.add(var_name)
        
        rule_name = rule["name"]
        # Generate all motifs of size 2-4
        for k in (2, 3, 4):
            if len(vars_in_rule) >= k:
                for combo in itertools.combinations(sorted(vars_in_rule), k):
                    motif_key = f"{k}:{','.join(combo)}"
                    motif_to_rules[motif_key].append(rule_name)
    
    return dict(motif_to_rules)

def compute_one_at_a_time_sensitivity(sim, output_keys):
    """Compute local sensitivity: effect of changing each rule confidence from 0 to 1.
    
    Returns dict mapping rule_name -> dict of output_key -> difference (confidence=1 - confidence=0).
    """
    # Baseline: all confidences at default (as defined in rule file)
    baseline = sim.run({})
    
    sensitivities = {}
    for rule_name in sim.rule_names:
        # Set this rule confidence to 1, others default
        params = {rule_name: 1.0}
        result = sim.run(params)
        diff = {key: result[key] - baseline[key] for key in output_keys}
        sensitivities[rule_name] = diff
    return sensitivities, baseline

# ── Saltelli sampling (reused from sobol_sensitivity.py) ─────────────────────
# Imported from sobol_sensitivity.py

# ── Sobol index computation ──────────────────────────────────────────────────
# Imported from sobol_sensitivity.py

# ── Main analysis ────────────────────────────────────────────────────────────

def run_hyper_sobol(n_base=64, sim_years=30, rng_seed=42):
    """Run Sobol sensitivity analysis on rule confidence parameters.
    
    Args:
        n_base: Base sample count (total sims = N*(2D+2)).
        sim_years: Simulation horizon (CA runs for 30 years, quarterly steps).
        rng_seed: Random seed for reproducibility.
    
    Returns:
        Dict with Sobol indices, motif analysis, and metadata.
    """
    # Initialize simulator and get rule names
    sim = RuleConfidenceSimulator()
    param_names = sim.rule_names
    d = len(param_names)
    param_bounds = np.array([[0.0, 1.0]] * d)  # all confidences [0,1]
    
    rng = np.random.default_rng(rng_seed)
    
    # Generate samples
    n_total = n_base * (2 * d + 2)
    print(f"Hyper-Sobol sensitivity analysis: {d} rule confidence parameters, N={n_base}")
    print(f"Total simulations: {n_total}")
    
    samples_01 = saltelli_sample(n_base, d, rng)
    # samples already in [0,1], no rescaling needed
    
    # Select output metrics to analyze
    # We'll compute Sobol indices for a few key outputs
    output_keys = [
        "final_bin_N_deletion",    # bin index of deletion state
        "final_heteroplasmy",      # continuous deletion heteroplasmy
        "final_atp",               # continuous ATP exemplar
        "final_bin_ATP",           # bin index of ATP state
        "final_bin_Senescent_fraction",  # senescence severity
    ]
    
    # Compute one-at-a-time local sensitivity
    print("Computing one-at-a-time sensitivity (45 rules)...")
    local_sensitivities, baseline = compute_one_at_a_time_sensitivity(sim, output_keys)
    
    # Prepare result arrays for Saltelli samples
    results = {key: np.zeros(n_total) for key in output_keys}
    
    # Run simulations
    print(f"Running {n_total} simulations...")
    t0 = time.time()
    
    for idx in range(n_total):
        if idx % 200 == 0 and idx > 0:
            elapsed = time.time() - t0
            rate = idx / elapsed
            remaining = (n_total - idx) / rate
            print(f"  {idx}/{n_total} ({elapsed:.0f}s elapsed, "
                  f"~{remaining:.0f}s remaining)")
        
        # Build parameter dict mapping rule names to confidence values
        row = samples_01[idx]
        params = {name: float(row[i]) for i, name in enumerate(param_names)}
        
        # Run CA with these rule confidences
        output = sim.run(params)
        
        for key in output_keys:
            results[key][idx] = output[key]
    
    elapsed = time.time() - t0
    print(f"Simulations complete: {elapsed:.1f}s "
          f"({n_total / elapsed:.0f} sims/sec)")
    
    # Extract sub-arrays for Sobol computation (same for all outputs)
    n = n_base
    sobol_results = {}
    for key in output_keys:
        y = results[key]
        y_A = y[:n]
        y_B = y[n:2*n]
        y_AB = y[2*n:2*n + d*n].reshape(d, n)
        y_BA = y[2*n + d*n:].reshape(d, n)
        
        S1, ST = sobol_indices(y_A, y_B, y_AB, y_BA)
        # Ensure indices are within [0,1] and ST >= S1
        S1 = np.clip(S1, 0.0, 1.0)
        ST = np.clip(ST, 0.0, 1.0)
        ST = np.maximum(ST, S1)
        interaction = ST - S1
        
        sobol_results[key] = {
            "S1": {name: float(S1[i]) for i, name in enumerate(param_names)},
            "ST": {name: float(ST[i]) for i, name in enumerate(param_names)},
            "interaction": {name: float(interaction[i]) for i, name in enumerate(param_names)},
        }
    
    # Compute rankings per output
    rankings = {}
    for key in output_keys:
        S1_dict = sobol_results[key]["S1"]
        ST_dict = sobol_results[key]["ST"]
        inter_dict = sobol_results[key]["interaction"]
        
        rankings[key] = {
            "top_S1": sorted(param_names, key=lambda n: S1_dict[n], reverse=True)[:10],
            "top_ST": sorted(param_names, key=lambda n: ST_dict[n], reverse=True)[:10],
            "top_interaction": sorted(param_names, key=lambda n: inter_dict[n], reverse=True)[:10],
        }
    
    # Extract motifs and map to rule interactions
    motifs = extract_variable_motifs()
    motif_to_rules = map_rules_to_motifs()
    
    # Compute average interaction strength per motif
    motif_interactions = {}
    for motif_key, rule_list in motif_to_rules.items():
        # For each output metric, compute average interaction of rules in this motif
        motif_interactions[motif_key] = {}
        for key in output_keys:
            inter_dict = sobol_results[key]["interaction"]
            # Filter to rules that appear in motif
            inter_vals = [inter_dict[r] for r in rule_list if r in inter_dict]
            if inter_vals:
                motif_interactions[motif_key][key] = {
                    "mean": float(np.mean(inter_vals)),
                    "std": float(np.std(inter_vals)),
                    "n_rules": len(inter_vals),
                }
            else:
                motif_interactions[motif_key][key] = None
    
    # Build final result dict
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_base": n_base,
        "n_total_sims": n_total,
        "elapsed_sec": elapsed,
        "sim_years": sim_years,
        "parameter_names": param_names,
        "sobol": sobol_results,
        "rankings": rankings,
        "motifs": motifs,
        "motif_to_rules": motif_to_rules,
        "motif_interactions": motif_interactions,
        "local_sensitivities": local_sensitivities,
        "baseline": baseline,
    }
    
    return result

def print_hyper_sobol_report(result):
    """Print human-readable summary of hyper-Sobol results."""
    print("\n" + "=" * 70)
    print("HYPER-SOBOL SENSITIVITY ANALYSIS OF RULE CONFIDENCES")
    print("=" * 70)
    
    param_names = result["parameter_names"]
    
    for output_key in result["sobol"].keys():
        print(f"\n  {output_key}:")
        print(f"  {'Rule':40s}  {'S1':>8s}  {'ST':>8s}  {'Interact':>8s}")
        print(f"  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*8}")
        
        sobol = result["sobol"][output_key]
        S1_dict = sobol["S1"]
        ST_dict = sobol["ST"]
        inter_dict = sobol["interaction"]
        
        # Show top 15 rules by ST
        top_rules = sorted(param_names, key=lambda n: ST_dict[n], reverse=True)[:15]
        for name in top_rules:
            print(f"  {name:40s}  {S1_dict[name]:8.4f}  {ST_dict[name]:8.4f}  "
                  f"{inter_dict[name]:8.4f}")
    
    # Motif interaction summary
    print("\n" + "=" * 70)
    print("MOTIF INTERACTION SUMMARY")
    print("=" * 70)
    
    motif_to_rules = result["motif_to_rules"]
    motif_interactions = result["motif_interactions"]
    
    # Sort motifs by average interaction across outputs
    motif_list = []
    for motif_key in motif_to_rules.keys():
        interactions = motif_interactions.get(motif_key, {})
        # Compute overall score (mean of mean interactions across outputs)
        scores = []
        for key, val in interactions.items():
            if val is not None:
                scores.append(val["mean"])
        if scores:
            overall = np.mean(scores)
            motif_list.append((motif_key, overall))
    
    if motif_list:
        motif_list.sort(key=lambda x: x[1], reverse=True)
        print("\nTop motifs by interaction strength:")
        for motif_key, score in motif_list[:15]:
            parts = motif_key.split(":")
            size = parts[0]
            vars_str = parts[1]
            print(f"  {size}-var motif [{vars_str}]: {score:.4f}")
    
    # Print top interacting rules overall
    print("\n" + "=" * 70)
    print("TOP INTERACTING RULES (highest ST - S1)")
    print("=" * 70)
    
    # Aggregate interaction across outputs
    rule_interaction_agg = defaultdict(float)
    rule_count = defaultdict(int)
    for output_key, sobol in result["sobol"].items():
        inter_dict = sobol["interaction"]
        for rule, val in inter_dict.items():
            rule_interaction_agg[rule] += val
            rule_count[rule] += 1
    
    rule_avg_interaction = {rule: rule_interaction_agg[rule] / rule_count[rule]
                            for rule in rule_interaction_agg}
    
    top_interacting = sorted(rule_avg_interaction.items(), 
                             key=lambda x: x[1], reverse=True)[:15]
    print("\nRule (average interaction across outputs):")
    for rule, inter in top_interacting:
        print(f"  {rule:40s}  {inter:.4f}")
    
    # Local sensitivity summary
    print("\n" + "=" * 70)
    print("LOCAL SENSITIVITY (rule confidence 0 → 1)")
    print("=" * 70)
    
    local_sensitivities = result.get("local_sensitivities", {})
    if local_sensitivities:
        # Compute overall magnitude per rule (average absolute difference across outputs)
        rule_magnitude = {}
        for rule, diffs in local_sensitivities.items():
            # Use absolute differences
            abs_vals = [abs(diffs[key]) for key in diffs]
            rule_magnitude[rule] = np.mean(abs_vals)
        
        top_local = sorted(rule_magnitude.items(), key=lambda x: x[1], reverse=True)[:15]
        print("\nRule (average absolute change across outputs):")
        for rule, mag in top_local:
            print(f"  {rule:40s}  {mag:.4f}")
    else:
        print("\nLocal sensitivities not available.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hyper-Sobol sensitivity analysis of CA rule confidences")
    parser.add_argument("--n-base", type=int, default=64,
                        help="Base sample count (default 64, total sims = N*(2*45+2))")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default="artifacts/hyper_sobol_report.json",
                        help="Output JSON path")
    
    args = parser.parse_args()
    
    result = run_hyper_sobol(n_base=args.n_base, rng_seed=args.seed)
    
    # Save results
    output_path = PROJECT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, cls=NumpyEncoder, indent=2)
    print(f"\nResults saved to {output_path}")
    
    # Print report
    print_hyper_sobol_report(result)