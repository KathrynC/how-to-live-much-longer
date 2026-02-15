#!/usr/bin/env python3
"""Generate 100 sample patients with biologically plausible correlations.

Creates a population that reflects real-world aging biology:
  - Older patients have more mtDNA damage (higher heteroplasmy)
  - Older patients have lower NAD+ (age-dependent decline)
  - Older patients have more inflammation (inflammaging)
  - Genetic vulnerability and metabolic demand are more independent
  - Near-cliff patients are rare (most people don't accumulate that much damage)

Then evaluates population quality: distributions, correlations, coverage,
clinical plausibility, and outcome diversity under simulation.
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from constants import (
    PATIENT_PARAMS, PATIENT_NAMES, DEFAULT_PATIENT,
    DEFAULT_INTERVENTION, snap_param,
)
from simulator import simulate
from analytics import compute_all, NumpyEncoder


# ── Generation ──────────────────────────────────────────────────────────────

def generate_patients(n: int = 100, seed: int = 42) -> list[dict]:
    """Generate n patients with biologically plausible correlations.

    Biology-informed correlation structure:
      - age drives heteroplasmy, NAD decline, and inflammation
      - genetic vulnerability adds scatter to heteroplasmy
      - metabolic demand is sampled independently (tissue type)
    """
    rng = np.random.default_rng(seed)
    patients = []

    for i in range(n):
        # 1. Sample age uniformly across the adult lifespan
        age = rng.uniform(20.0, 90.0)

        # 2. Heteroplasmy: age-dependent with scatter
        #    Young (~20): het ~0.02-0.10
        #    Middle (~50): het ~0.15-0.35
        #    Old (~80): het ~0.30-0.65
        #    Rare outliers can be higher (genetic disease, etc.)
        age_frac = (age - 20.0) / 70.0  # 0 at 20, 1 at 90
        het_mean = 0.05 + 0.45 * age_frac**1.3  # nonlinear: accelerating with age
        het_std = 0.08 + 0.07 * age_frac  # more variance in older patients
        het = rng.normal(het_mean, het_std)
        het = np.clip(het, 0.02, 0.90)

        # 3. NAD+: declines with age (Camacho-Pereira 2016)
        #    Young: ~0.85-1.0; Old: ~0.3-0.6
        nad_mean = 0.95 - 0.50 * age_frac
        nad = rng.normal(nad_mean, 0.08)
        nad = np.clip(nad, 0.2, 1.0)

        # 4. Genetic vulnerability: mostly normal (1.0), some outliers
        #    Haplogroup-dependent; most people are average
        gv = rng.lognormal(mean=0.0, sigma=0.25)  # centered at ~1.0
        gv = np.clip(gv, 0.5, 2.0)

        # 5. Metabolic demand: tissue-dependent, independent of age
        #    Most tissues ~1.0; brain/cardiac patients have higher demand
        md = rng.choice([0.5, 0.75, 1.0, 1.0, 1.0, 1.25, 1.5, 2.0],
                        p=[0.05, 0.10, 0.35, 0.15, 0.10, 0.10, 0.10, 0.05])

        # 6. Inflammation: age-driven (inflammaging) + independent component
        infl_mean = 0.05 + 0.40 * age_frac  # 0.05 at 20, 0.45 at 90
        infl = rng.normal(infl_mean, 0.10)
        infl = np.clip(infl, 0.0, 0.95)

        # Snap to grid
        patient = {
            "baseline_age": snap_param("baseline_age", age),
            "baseline_heteroplasmy": snap_param("baseline_heteroplasmy", het),
            "baseline_nad_level": snap_param("baseline_nad_level", nad),
            "genetic_vulnerability": snap_param("genetic_vulnerability", gv),
            "metabolic_demand": snap_param("metabolic_demand", md),
            "inflammation_level": snap_param("inflammation_level", infl),
        }
        patient["_raw"] = {
            "age": float(age), "het": float(het), "nad": float(nad),
            "gv": float(gv), "md": float(md), "infl": float(infl),
        }
        patient["_id"] = i
        patients.append(patient)

    return patients


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_population(patients: list[dict]) -> dict:
    """Evaluate the quality of a generated patient population.

    Returns a dict with distribution stats, correlations, coverage,
    clinical plausibility checks, and simulation outcome diversity.
    """
    n = len(patients)
    print(f"\n{'='*70}")
    print(f"PATIENT POPULATION EVALUATION — {n} patients")
    print(f"{'='*70}")

    # Extract arrays (snapped values)
    params = {}
    for name in PATIENT_NAMES:
        params[name] = np.array([p[name] for p in patients])

    # Also extract raw (pre-snap) values for correlation analysis
    raw = {}
    for key in ["age", "het", "nad", "gv", "md", "infl"]:
        raw[key] = np.array([p["_raw"][key] for p in patients])

    # ── 1. Distribution statistics ──────────────────────────────────────
    print(f"\n--- Distribution Statistics (snapped grid values) ---")
    print(f"{'Parameter':30s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'Median':>8s}")
    print("-" * 78)
    for name in PATIENT_NAMES:
        v = params[name]
        print(f"{name:30s} {np.mean(v):8.2f} {np.std(v):8.2f} "
              f"{np.min(v):8.2f} {np.max(v):8.2f} {np.median(v):8.2f}")

    # ── 2. Grid coverage ────────────────────────────────────────────────
    print(f"\n--- Grid Coverage ---")
    coverage_report = {}
    for name in PATIENT_NAMES:
        grid = PATIENT_PARAMS[name]["grid"]
        used = set(params[name])
        used_grid = used & set(grid)
        coverage = len(used_grid) / len(grid)
        coverage_report[name] = {
            "grid_size": len(grid),
            "grid_points_used": len(used_grid),
            "coverage": coverage,
            "unused": sorted(set(grid) - used_grid),
        }
        unused_str = ", ".join(f"{v}" for v in coverage_report[name]["unused"])
        print(f"  {name:30s}: {len(used_grid)}/{len(grid)} grid points "
              f"({100*coverage:.0f}%)"
              f"{f'  unused: [{unused_str}]' if unused_str else ''}")

    # Value frequency distribution
    print(f"\n--- Value Frequencies (top grid points per param) ---")
    for name in PATIENT_NAMES:
        vals, counts = np.unique(params[name], return_counts=True)
        top3 = sorted(zip(counts, vals), reverse=True)[:3]
        freqs = ", ".join(f"{v}({c})" for c, v in top3)
        print(f"  {name:30s}: {freqs}")

    # ── 3. Correlations (raw pre-snap values) ───────────────────────────
    print(f"\n--- Correlations (raw values, pre-snap) ---")
    raw_keys = ["age", "het", "nad", "gv", "md", "infl"]
    raw_labels = ["Age", "Het", "NAD", "GenVuln", "MetDem", "Inflam"]
    raw_matrix = np.column_stack([raw[k] for k in raw_keys])
    corr = np.corrcoef(raw_matrix, rowvar=False)

    # Print correlation matrix
    print(f"{'':10s}", end="")
    for lbl in raw_labels:
        print(f"{lbl:>8s}", end="")
    print()
    for i, lbl in enumerate(raw_labels):
        print(f"{lbl:10s}", end="")
        for j in range(len(raw_labels)):
            r = corr[i, j]
            marker = " *" if abs(r) > 0.3 and i != j else "  "
            print(f"{r:6.2f}{marker}", end="")
        print()
    print("  (* marks |r| > 0.3)")

    # Expected correlations check
    print(f"\n--- Biological Plausibility of Correlations ---")
    expected = [
        ("age", "het", "+", "Older → more mtDNA damage"),
        ("age", "nad", "-", "Older → lower NAD+ (Ca16)"),
        ("age", "infl", "+", "Older → more inflammaging"),
        ("het", "nad", "-", "More damage → worse NAD state"),
        ("gv", "md", "~0", "Independent (haplogroup vs tissue)"),
    ]
    for k1, k2, expected_sign, reason in expected:
        i1, i2 = raw_keys.index(k1), raw_keys.index(k2)
        r = corr[i1, i2]
        if expected_sign == "+":
            ok = r > 0.2
        elif expected_sign == "-":
            ok = r < -0.2
        else:
            ok = abs(r) < 0.3
        status = "OK" if ok else "CONCERN"
        print(f"  {k1}-{k2}: r={r:+.2f} (expected {expected_sign}) "
              f"[{reason}] → {status}")

    # ── 4. Clinical plausibility ────────────────────────────────────────
    print(f"\n--- Clinical Plausibility Checks ---")
    issues = []

    # Young patients shouldn't have very high heteroplasmy
    young_high_het = sum(1 for p in patients
                        if p["baseline_age"] <= 30 and p["baseline_heteroplasmy"] >= 0.5)
    if young_high_het > 2:
        issues.append(f"  WARNING: {young_high_het} patients age ≤30 with het ≥0.5")
    else:
        print(f"  Young (≤30) with high het (≥0.5): {young_high_het}/{sum(1 for p in patients if p['baseline_age'] <= 30)} — OK")

    # Old patients should rarely have perfect NAD
    old_perfect_nad = sum(1 for p in patients
                         if p["baseline_age"] >= 70 and p["baseline_nad_level"] >= 1.0)
    if old_perfect_nad > 3:
        issues.append(f"  WARNING: {old_perfect_nad} patients age ≥70 with NAD=1.0")
    else:
        print(f"  Old (≥70) with NAD=1.0: {old_perfect_nad}/{sum(1 for p in patients if p['baseline_age'] >= 70)} — OK")

    # Near-cliff patients should be minority
    near_cliff = sum(1 for p in patients if p["baseline_heteroplasmy"] >= 0.6)
    print(f"  Near-cliff (het ≥0.6): {near_cliff}/{n} ({100*near_cliff/n:.0f}%)"
          f" — {'OK' if near_cliff < n * 0.20 else 'HIGH'}")

    # Very young + very inflamed is unlikely
    young_inflamed = sum(1 for p in patients
                        if p["baseline_age"] <= 30 and p["inflammation_level"] >= 0.5)
    print(f"  Young (≤30) + high inflammation (≥0.5): {young_inflamed} — "
          f"{'OK' if young_inflamed <= 2 else 'CONCERN'}")

    # Age-decade distribution
    print(f"\n  Age decade distribution:")
    for decade_start in range(20, 100, 10):
        count = sum(1 for p in patients
                    if decade_start <= p["baseline_age"] < decade_start + 10)
        bar = "#" * count
        print(f"    {decade_start:2d}-{decade_start+9:2d}: {count:3d} {bar}")

    for issue in issues:
        print(issue)

    # ── 5. Simulation outcome diversity ─────────────────────────────────
    print(f"\n--- Simulation Outcome Diversity (no treatment) ---")
    atp_finals = []
    het_finals = []
    crisis_times = []

    for i, p in enumerate(patients):
        patient_dict = {k: p[k] for k in PATIENT_NAMES}
        try:
            result = simulate(intervention=DEFAULT_INTERVENTION, patient=patient_dict)
            baseline = simulate(intervention=DEFAULT_INTERVENTION, patient=patient_dict)
            analytics = compute_all(result, baseline)
            atp_finals.append(analytics["energy"]["atp_final"])
            het_finals.append(analytics["damage"]["het_final"])
            crisis_times.append(analytics["energy"]["time_to_crisis_years"])
        except Exception as e:
            print(f"  Patient {i} sim error: {e}")

    atp_finals = np.array(atp_finals)
    het_finals = np.array(het_finals)
    crisis_times = np.array(crisis_times)

    print(f"  Simulated {len(atp_finals)}/{n} patients successfully")
    print(f"\n  Final ATP:  mean={np.mean(atp_finals):.3f}, "
          f"std={np.std(atp_finals):.3f}, "
          f"range=[{np.min(atp_finals):.3f}, {np.max(atp_finals):.3f}]")
    print(f"  Final Het:  mean={np.mean(het_finals):.3f}, "
          f"std={np.std(het_finals):.3f}, "
          f"range=[{np.min(het_finals):.3f}, {np.max(het_finals):.3f}]")

    finite_crisis = crisis_times[np.isfinite(crisis_times)]
    inf_crisis = np.sum(np.isinf(crisis_times))
    print(f"  Crisis: {len(finite_crisis)} reach crisis, "
          f"{inf_crisis} never reach crisis")
    if len(finite_crisis) > 0:
        print(f"  Crisis time: mean={np.mean(finite_crisis):.1f}yr, "
              f"range=[{np.min(finite_crisis):.1f}, {np.max(finite_crisis):.1f}]")

    # Outcome categories
    collapsed = sum(1 for a in atp_finals if a < 0.2)
    declining = sum(1 for a in atp_finals if 0.2 <= a < 0.5)
    stable = sum(1 for a in atp_finals if 0.5 <= a < 0.8)
    healthy = sum(1 for a in atp_finals if a >= 0.8)

    print(f"\n  Outcome categories (by final ATP):")
    print(f"    Collapsed (<0.2):  {collapsed:3d} ({100*collapsed/n:.0f}%)")
    print(f"    Declining (0.2-0.5): {declining:3d} ({100*declining/n:.0f}%)")
    print(f"    Stable    (0.5-0.8): {stable:3d} ({100*stable/n:.0f}%)")
    print(f"    Healthy   (≥0.8):  {healthy:3d} ({100*healthy/n:.0f}%)")

    cliff_crossed = sum(1 for h in het_finals if h >= 0.7)
    print(f"  Cliff crossed (het ≥ 0.7): {cliff_crossed}/{n} ({100*cliff_crossed/n:.0f}%)")

    # ── 6. Overall quality score ────────────────────────────────────────
    print(f"\n--- Overall Population Quality ---")
    scores = {}

    # Coverage score: fraction of grid points used across all params
    total_grid = sum(len(PATIENT_PARAMS[n]["grid"]) for n in PATIENT_NAMES)
    total_used = sum(coverage_report[n]["grid_points_used"] for n in PATIENT_NAMES)
    scores["grid_coverage"] = total_used / total_grid

    # Correlation plausibility: count correct signs
    correct = 0
    for k1, k2, expected_sign, _ in expected:
        i1, i2 = raw_keys.index(k1), raw_keys.index(k2)
        r = corr[i1, i2]
        if expected_sign == "+" and r > 0.2:
            correct += 1
        elif expected_sign == "-" and r < -0.2:
            correct += 1
        elif expected_sign == "~0" and abs(r) < 0.3:
            correct += 1
    scores["correlation_plausibility"] = correct / len(expected)

    # Outcome diversity: entropy-like measure
    cat_fracs = np.array([collapsed, declining, stable, healthy]) / n
    cat_fracs = cat_fracs[cat_fracs > 0]
    outcome_entropy = -np.sum(cat_fracs * np.log2(cat_fracs + 1e-12))
    scores["outcome_diversity"] = min(outcome_entropy / 2.0, 1.0)  # max entropy for 4 categories = 2.0

    # Clinical plausibility: fewer issues = better
    n_issues = len(issues)
    scores["clinical_plausibility"] = max(0.0, 1.0 - 0.2 * n_issues)

    overall = np.mean(list(scores.values()))
    for k, v in scores.items():
        print(f"  {k:30s}: {v:.2f}")
    print(f"  {'OVERALL':30s}: {overall:.2f}")

    print(f"\n{'='*70}")

    return {
        "n_patients": n,
        "distribution": {
            name: {"mean": float(np.mean(params[name])),
                   "std": float(np.std(params[name])),
                   "min": float(np.min(params[name])),
                   "max": float(np.max(params[name]))}
            for name in PATIENT_NAMES
        },
        "correlations": {f"{raw_keys[i]}-{raw_keys[j]}": float(corr[i, j])
                        for i in range(len(raw_keys))
                        for j in range(i+1, len(raw_keys))},
        "coverage": coverage_report,
        "outcomes": {
            "atp_final_mean": float(np.mean(atp_finals)),
            "atp_final_std": float(np.std(atp_finals)),
            "het_final_mean": float(np.mean(het_finals)),
            "collapsed": collapsed, "declining": declining,
            "stable": stable, "healthy": healthy,
            "cliff_crossed": cliff_crossed,
        },
        "quality_scores": scores,
        "overall_score": overall,
    }


# ── Edge-case generation ────────────────────────────────────────────────────

def _edge_patient(pid: int, label: str, category: str, **kwargs) -> dict:
    """Helper to build an edge-case patient dict from explicit values."""
    p = dict(DEFAULT_PATIENT)  # start from default
    p.update(kwargs)
    # Snap all values
    snapped = {k: snap_param(k, p[k]) for k in PATIENT_NAMES}
    snapped["_raw"] = {
        "age": p["baseline_age"], "het": p["baseline_heteroplasmy"],
        "nad": p["baseline_nad_level"], "gv": p["genetic_vulnerability"],
        "md": p["metabolic_demand"], "infl": p["inflammation_level"],
    }
    snapped["_id"] = pid
    snapped["_label"] = label
    snapped["_category"] = category
    return snapped


def generate_edge_patients() -> list[dict]:
    """Generate ~100 edge-case patients for robustness testing.

    Organized into categories:
      1. Single-parameter extremes: one param at min or max, rest default
      2. Corner patients: all params at extremes simultaneously
      3. Cliff boundary: heteroplasmy near the 0.70 cliff threshold
      4. Contradictory combos: biologically unlikely but must not crash
      5. Maximum stress: worst-case parameter combinations
      6. Tissue × vulnerability crosses: metabolic demand vs genetic fragility
      7. Age-boundary sweeps: youngest/oldest with various damage levels
      8. Near-zero and near-saturation: values at or near hard limits
    """
    patients = []
    pid = 0

    def add(label, category, **kw):
        nonlocal pid
        patients.append(_edge_patient(pid, label, category, **kw))
        pid += 1

    # ── Category 1: Single-parameter extremes (12 patients) ─────────
    cat = "single_extreme"
    extremes = [
        ("baseline_age", 20.0, 90.0),
        ("baseline_heteroplasmy", 0.02, 0.95),
        ("baseline_nad_level", 0.2, 1.0),
        ("genetic_vulnerability", 0.5, 2.0),
        ("metabolic_demand", 0.5, 2.0),
        ("inflammation_level", 0.0, 1.0),
    ]
    for name, lo, hi in extremes:
        add(f"{name}_min", cat, **{name: lo})
        add(f"{name}_max", cat, **{name: hi})

    # ── Category 2: Corner patients (10 patients) ───────────────────
    cat = "corner"
    # All minimums (healthiest possible)
    add("all_min", cat,
        baseline_age=20, baseline_heteroplasmy=0.02, baseline_nad_level=1.0,
        genetic_vulnerability=0.5, metabolic_demand=0.5, inflammation_level=0.0)
    # All maximums (sickest possible)
    add("all_max", cat,
        baseline_age=90, baseline_heteroplasmy=0.95, baseline_nad_level=0.2,
        genetic_vulnerability=2.0, metabolic_demand=2.0, inflammation_level=1.0)
    # Worst patient params, best biology
    add("old_but_resilient", cat,
        baseline_age=90, baseline_heteroplasmy=0.05, baseline_nad_level=1.0,
        genetic_vulnerability=0.5, metabolic_demand=0.5, inflammation_level=0.0)
    # Young but genetically doomed
    add("young_but_doomed", cat,
        baseline_age=20, baseline_heteroplasmy=0.90, baseline_nad_level=0.2,
        genetic_vulnerability=2.0, metabolic_demand=2.0, inflammation_level=1.0)
    # Checkerboard: alternating min/max
    add("checkerboard_A", cat,
        baseline_age=20, baseline_heteroplasmy=0.90, baseline_nad_level=1.0,
        genetic_vulnerability=0.5, metabolic_demand=2.0, inflammation_level=0.0)
    add("checkerboard_B", cat,
        baseline_age=90, baseline_heteroplasmy=0.05, baseline_nad_level=0.2,
        genetic_vulnerability=2.0, metabolic_demand=0.5, inflammation_level=1.0)
    # High demand + low everything else
    add("high_demand_only", cat,
        baseline_age=50, baseline_heteroplasmy=0.10, baseline_nad_level=0.8,
        genetic_vulnerability=1.0, metabolic_demand=2.0, inflammation_level=0.1)
    # High vulnerability + low everything else
    add("high_vuln_only", cat,
        baseline_age=50, baseline_heteroplasmy=0.10, baseline_nad_level=0.8,
        genetic_vulnerability=2.0, metabolic_demand=1.0, inflammation_level=0.1)
    # High inflammation + low everything else
    add("high_infl_only", cat,
        baseline_age=50, baseline_heteroplasmy=0.10, baseline_nad_level=0.8,
        genetic_vulnerability=1.0, metabolic_demand=1.0, inflammation_level=1.0)
    # Minimum NAD + otherwise healthy
    add("nad_depleted_only", cat,
        baseline_age=50, baseline_heteroplasmy=0.10, baseline_nad_level=0.2,
        genetic_vulnerability=1.0, metabolic_demand=1.0, inflammation_level=0.1)

    # ── Category 3: Cliff boundary (14 patients) ───────────────────
    cat = "cliff_boundary"
    # Sweep heteroplasmy across the cliff with a moderate patient
    for het in [0.55, 0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80, 0.85, 0.90]:
        add(f"cliff_het_{het:.2f}", cat,
            baseline_age=70, baseline_heteroplasmy=het, baseline_nad_level=0.5,
            genetic_vulnerability=1.0, metabolic_demand=1.0, inflammation_level=0.3)
    # At cliff with young vs old
    add("cliff_young", cat,
        baseline_age=20, baseline_heteroplasmy=0.70, baseline_nad_level=0.8,
        genetic_vulnerability=1.0, metabolic_demand=1.0, inflammation_level=0.1)
    add("cliff_old", cat,
        baseline_age=90, baseline_heteroplasmy=0.70, baseline_nad_level=0.3,
        genetic_vulnerability=1.0, metabolic_demand=1.0, inflammation_level=0.5)
    # At cliff with max vulnerability
    add("cliff_max_vuln", cat,
        baseline_age=65, baseline_heteroplasmy=0.70, baseline_nad_level=0.5,
        genetic_vulnerability=2.0, metabolic_demand=1.5, inflammation_level=0.5)
    # At cliff with min vulnerability (centenarian genetics)
    add("cliff_resilient", cat,
        baseline_age=65, baseline_heteroplasmy=0.70, baseline_nad_level=0.8,
        genetic_vulnerability=0.5, metabolic_demand=0.5, inflammation_level=0.1)

    # ── Category 4: Contradictory / biologically unlikely (12 patients) ──
    cat = "contradictory"
    # Young + extremely high heteroplasmy (MELAS-like)
    add("young_melas", cat,
        baseline_age=20, baseline_heteroplasmy=0.80, baseline_nad_level=0.6,
        genetic_vulnerability=2.0, metabolic_demand=1.5, inflammation_level=0.3)
    # Old + pristine mitochondria (centenarian superager)
    add("old_superager", cat,
        baseline_age=90, baseline_heteroplasmy=0.05, baseline_nad_level=0.8,
        genetic_vulnerability=0.5, metabolic_demand=1.0, inflammation_level=0.1)
    # Old + perfect NAD (aggressive supplementation baseline)
    add("old_perfect_nad", cat,
        baseline_age=90, baseline_heteroplasmy=0.40, baseline_nad_level=1.0,
        genetic_vulnerability=1.0, metabolic_demand=1.0, inflammation_level=0.3)
    # Young + max inflammation (autoimmune)
    add("young_autoimmune", cat,
        baseline_age=25, baseline_heteroplasmy=0.10, baseline_nad_level=0.8,
        genetic_vulnerability=1.0, metabolic_demand=1.0, inflammation_level=1.0)
    # Low demand + high vulnerability (paradox: easy tissue but fragile DNA)
    add("low_demand_fragile", cat,
        baseline_age=60, baseline_heteroplasmy=0.40, baseline_nad_level=0.5,
        genetic_vulnerability=2.0, metabolic_demand=0.5, inflammation_level=0.3)
    # High demand + low vulnerability (robust brain)
    add("robust_brain", cat,
        baseline_age=60, baseline_heteroplasmy=0.20, baseline_nad_level=0.7,
        genetic_vulnerability=0.5, metabolic_demand=2.0, inflammation_level=0.2)
    # Zero heteroplasmy (theoretically perfect)
    add("zero_het", cat,
        baseline_age=40, baseline_heteroplasmy=0.0, baseline_nad_level=1.0,
        genetic_vulnerability=1.0, metabolic_demand=1.0, inflammation_level=0.0)
    # Maximum heteroplasmy (should be fully collapsed)
    add("max_het", cat,
        baseline_age=80, baseline_heteroplasmy=0.95, baseline_nad_level=0.2,
        genetic_vulnerability=1.5, metabolic_demand=1.0, inflammation_level=0.7)
    # High NAD + high inflammation (competing signals)
    add("nad_vs_infl", cat,
        baseline_age=60, baseline_heteroplasmy=0.30, baseline_nad_level=1.0,
        genetic_vulnerability=1.0, metabolic_demand=1.0, inflammation_level=1.0)
    # Low NAD + zero inflammation
    add("low_nad_no_infl", cat,
        baseline_age=60, baseline_heteroplasmy=0.30, baseline_nad_level=0.2,
        genetic_vulnerability=1.0, metabolic_demand=1.0, inflammation_level=0.0)
    # Everything at midpoint (the "boring" patient — tests default behavior)
    add("exact_midpoint", cat,
        baseline_age=55, baseline_heteroplasmy=0.47, baseline_nad_level=0.6,
        genetic_vulnerability=1.25, metabolic_demand=1.25, inflammation_level=0.5)
    # Identical to DEFAULT_PATIENT (sanity check)
    add("default_clone", cat, **DEFAULT_PATIENT)

    # ── Category 5: Maximum stress (10 patients) ──────────────────
    cat = "max_stress"
    # Triple threat: old + damaged + inflamed
    add("triple_threat", cat,
        baseline_age=85, baseline_heteroplasmy=0.80, baseline_nad_level=0.2,
        genetic_vulnerability=1.5, metabolic_demand=1.5, inflammation_level=0.9)
    # Brain on fire: high demand + high ROS-generating conditions
    add("brain_on_fire", cat,
        baseline_age=75, baseline_heteroplasmy=0.60, baseline_nad_level=0.3,
        genetic_vulnerability=1.5, metabolic_demand=2.0, inflammation_level=0.8)
    # Cardiac crisis: high demand + near cliff
    add("cardiac_crisis", cat,
        baseline_age=70, baseline_heteroplasmy=0.65, baseline_nad_level=0.4,
        genetic_vulnerability=1.25, metabolic_demand=1.8, inflammation_level=0.6)
    # MELAS child: young + genetically devastated
    add("melas_child", cat,
        baseline_age=20, baseline_heteroplasmy=0.70, baseline_nad_level=0.5,
        genetic_vulnerability=2.0, metabolic_demand=1.5, inflammation_level=0.5)
    # Sarcopenic elder: old + muscle wasting
    add("sarcopenic_elder", cat,
        baseline_age=85, baseline_heteroplasmy=0.55, baseline_nad_level=0.3,
        genetic_vulnerability=1.25, metabolic_demand=1.5, inflammation_level=0.7)
    # Post-cliff collapse: already past the cliff
    add("post_cliff_collapse", cat,
        baseline_age=80, baseline_heteroplasmy=0.85, baseline_nad_level=0.2,
        genetic_vulnerability=1.5, metabolic_demand=1.5, inflammation_level=0.8)
    # Extreme vulnerability, moderate everything else
    add("glass_cannon", cat,
        baseline_age=60, baseline_heteroplasmy=0.40, baseline_nad_level=0.5,
        genetic_vulnerability=2.0, metabolic_demand=2.0, inflammation_level=0.5)
    # NAD desert + high demand
    add("nad_desert_high_demand", cat,
        baseline_age=70, baseline_heteroplasmy=0.45, baseline_nad_level=0.2,
        genetic_vulnerability=1.0, metabolic_demand=2.0, inflammation_level=0.4)
    # Full inflammaging cascade
    add("inflammaging_cascade", cat,
        baseline_age=80, baseline_heteroplasmy=0.50, baseline_nad_level=0.3,
        genetic_vulnerability=1.0, metabolic_demand=1.0, inflammation_level=1.0)
    # Youngest possible + everything else maxed
    add("young_max_stress", cat,
        baseline_age=20, baseline_heteroplasmy=0.90, baseline_nad_level=0.2,
        genetic_vulnerability=2.0, metabolic_demand=2.0, inflammation_level=1.0)

    # ── Category 6: Tissue × vulnerability crosses (8 patients) ──
    cat = "tissue_vuln_cross"
    for md, md_label in [(0.5, "low_demand"), (2.0, "high_demand")]:
        for gv, gv_label in [(0.5, "resilient"), (2.0, "fragile")]:
            for age, age_label in [(30, "young"), (80, "old")]:
                add(f"{md_label}_{gv_label}_{age_label}", cat,
                    baseline_age=age, baseline_heteroplasmy=0.30,
                    baseline_nad_level=0.6, genetic_vulnerability=gv,
                    metabolic_demand=md, inflammation_level=0.25)

    # ── Category 7: Age sweep at cliff (8 patients) ────────────────
    cat = "age_at_cliff"
    for age in [20, 30, 40, 50, 60, 70, 80, 90]:
        add(f"age_{age}_at_cliff", cat,
            baseline_age=age, baseline_heteroplasmy=0.70,
            baseline_nad_level=0.5, genetic_vulnerability=1.0,
            metabolic_demand=1.0, inflammation_level=0.3)

    # ── Category 8: Near-zero / near-saturation (8 patients) ──────
    cat = "near_limits"
    # Epsilon above zero heteroplasmy
    add("het_epsilon", cat,
        baseline_age=40, baseline_heteroplasmy=0.01, baseline_nad_level=0.8,
        genetic_vulnerability=1.0, metabolic_demand=1.0, inflammation_level=0.1)
    # Just below max heteroplasmy
    add("het_near_max", cat,
        baseline_age=80, baseline_heteroplasmy=0.94, baseline_nad_level=0.2,
        genetic_vulnerability=1.0, metabolic_demand=1.0, inflammation_level=0.5)
    # NAD barely above minimum
    add("nad_barely_alive", cat,
        baseline_age=70, baseline_heteroplasmy=0.40, baseline_nad_level=0.21,
        genetic_vulnerability=1.0, metabolic_demand=1.0, inflammation_level=0.3)
    # Inflammation at exact 0.0 (no inflammaging at all)
    add("zero_inflammation", cat,
        baseline_age=70, baseline_heteroplasmy=0.30, baseline_nad_level=0.6,
        genetic_vulnerability=1.0, metabolic_demand=1.0, inflammation_level=0.0)
    # Inflammation at exact 1.0 (maximal)
    add("max_inflammation", cat,
        baseline_age=70, baseline_heteroplasmy=0.30, baseline_nad_level=0.6,
        genetic_vulnerability=1.0, metabolic_demand=1.0, inflammation_level=1.0)
    # Vulnerability at exact boundaries
    add("min_vulnerability", cat,
        baseline_age=60, baseline_heteroplasmy=0.30, baseline_nad_level=0.6,
        genetic_vulnerability=0.5, metabolic_demand=1.0, inflammation_level=0.25)
    add("max_vulnerability", cat,
        baseline_age=60, baseline_heteroplasmy=0.30, baseline_nad_level=0.6,
        genetic_vulnerability=2.0, metabolic_demand=1.0, inflammation_level=0.25)
    # Exact cliff threshold
    add("exact_cliff", cat,
        baseline_age=65, baseline_heteroplasmy=0.70, baseline_nad_level=0.5,
        genetic_vulnerability=1.0, metabolic_demand=1.0, inflammation_level=0.3)

    return patients


def evaluate_edge_population(patients: list[dict]) -> dict:
    """Evaluate edge-case population: focus on simulator robustness.

    Unlike evaluate_population (which checks biological plausibility),
    this checks whether the simulator handles all edge cases without
    errors, NaN/Inf values, or unexpected state variable violations.
    """
    n = len(patients)
    print(f"\n{'='*70}")
    print(f"EDGE-CASE POPULATION EVALUATION — {n} patients")
    print(f"{'='*70}")

    # ── Category summary ────────────────────────────────────────────
    categories = {}
    for p in patients:
        cat = p.get("_category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    print(f"\n--- Categories ---")
    for cat, count in sorted(categories.items()):
        print(f"  {cat:30s}: {count:3d} patients")
    print(f"  {'TOTAL':30s}: {n:3d}")

    # ── Simulate all patients ───────────────────────────────────────
    print(f"\n--- Simulation Robustness ---")
    results = []
    errors = []
    for p in patients:
        patient_dict = {k: p[k] for k in PATIENT_NAMES}
        label = p.get("_label", f"patient_{p['_id']}")
        cat = p.get("_category", "unknown")
        try:
            result = simulate(intervention=DEFAULT_INTERVENTION, patient=patient_dict)
            baseline = simulate(intervention=DEFAULT_INTERVENTION, patient=patient_dict)
            analytics = compute_all(result, baseline)

            # Check for NaN/Inf in state variables
            states = result["states"]
            het = result["heteroplasmy"]
            has_nan = bool(np.any(np.isnan(states)) or np.any(np.isnan(het)))
            has_inf = bool(np.any(np.isinf(states)) or np.any(np.isinf(het)))

            # Check for negative state variables (should never happen)
            has_negative = bool(np.any(states < -1e-10))

            # Check heteroplasmy stays in [0, 1]
            het_out_of_range = bool(np.any(het < -1e-10) or np.any(het > 1.0 + 1e-10))

            # Check ATP doesn't go wildly negative
            atp = states[:, 2]
            atp_negative = bool(np.any(atp < -0.01))

            issues_list = []
            if has_nan:
                issues_list.append("NaN in states")
            if has_inf:
                issues_list.append("Inf in states")
            if has_negative:
                issues_list.append("negative states")
            if het_out_of_range:
                issues_list.append(f"het out of [0,1]: [{np.min(het):.4f}, {np.max(het):.4f}]")
            if atp_negative:
                issues_list.append(f"ATP negative: min={np.min(atp):.4f}")

            results.append({
                "id": p["_id"],
                "label": label,
                "category": cat,
                "success": True,
                "issues": issues_list,
                "atp_final": float(analytics["energy"]["atp_final"]),
                "het_final": float(analytics["damage"]["het_final"]),
                "atp_min": float(analytics["energy"]["atp_min"]),
                "het_max": float(analytics["damage"]["het_max"]),
                "time_to_crisis": float(analytics["energy"]["time_to_crisis_years"]),
                "patient": patient_dict,
            })
        except Exception as e:
            errors.append({
                "id": p["_id"],
                "label": label,
                "category": cat,
                "error": str(e),
                "patient": patient_dict,
            })
            results.append({
                "id": p["_id"], "label": label, "category": cat,
                "success": False, "issues": [f"CRASH: {e}"],
                "atp_final": None, "het_final": None,
                "atp_min": None, "het_max": None,
                "time_to_crisis": None, "patient": patient_dict,
            })

    n_success = sum(1 for r in results if r["success"])
    n_issues = sum(1 for r in results if r["success"] and r["issues"])
    n_clean = sum(1 for r in results if r["success"] and not r["issues"])
    n_crash = len(errors)

    print(f"  Simulated: {n_success}/{n} succeeded, {n_crash} crashed")
    print(f"  Clean runs (no issues): {n_clean}/{n}")
    print(f"  Runs with issues: {n_issues}/{n}")

    if errors:
        print(f"\n  CRASHES:")
        for e in errors:
            print(f"    [{e['category']}] {e['label']}: {e['error']}")

    # Report issues
    issue_patients = [r for r in results if r["success"] and r["issues"]]
    if issue_patients:
        print(f"\n  ISSUES (non-crashing):")
        for r in issue_patients:
            print(f"    [{r['category']}] {r['label']}: {', '.join(r['issues'])}")

    # ── Outcome extremes ────────────────────────────────────────────
    print(f"\n--- Outcome Extremes ---")
    successful = [r for r in results if r["success"]]

    if successful:
        atp_finals = np.array([r["atp_final"] for r in successful])
        het_finals = np.array([r["het_final"] for r in successful])
        atp_mins = np.array([r["atp_min"] for r in successful])

        print(f"  Final ATP range: [{np.min(atp_finals):.4f}, {np.max(atp_finals):.4f}]")
        print(f"  Final het range: [{np.min(het_finals):.4f}, {np.max(het_finals):.4f}]")
        print(f"  Min ATP ever:    {np.min(atp_mins):.4f}")

        # Find the most extreme outcomes
        worst_atp = min(successful, key=lambda r: r["atp_final"])
        best_atp = max(successful, key=lambda r: r["atp_final"])
        worst_het = max(successful, key=lambda r: r["het_final"])
        best_het = min(successful, key=lambda r: r["het_final"])

        print(f"\n  Lowest final ATP:  {worst_atp['atp_final']:.4f} — {worst_atp['label']} [{worst_atp['category']}]")
        print(f"  Highest final ATP: {best_atp['atp_final']:.4f} — {best_atp['label']} [{best_atp['category']}]")
        print(f"  Highest final het: {worst_het['het_final']:.4f} — {worst_het['label']} [{worst_het['category']}]")
        print(f"  Lowest final het:  {best_het['het_final']:.4f} — {best_het['label']} [{best_het['category']}]")

        # Outcome categories
        collapsed = sum(1 for a in atp_finals if a < 0.2)
        declining = sum(1 for a in atp_finals if 0.2 <= a < 0.5)
        stable = sum(1 for a in atp_finals if 0.5 <= a < 0.8)
        healthy = sum(1 for a in atp_finals if a >= 0.8)
        cliff_crossed = sum(1 for h in het_finals if h >= 0.7)

        print(f"\n  Outcome categories:")
        print(f"    Collapsed (<0.2):    {collapsed:3d} ({100*collapsed/len(successful):.0f}%)")
        print(f"    Declining (0.2-0.5): {declining:3d} ({100*declining/len(successful):.0f}%)")
        print(f"    Stable    (0.5-0.8): {stable:3d} ({100*stable/len(successful):.0f}%)")
        print(f"    Healthy   (>=0.8):   {healthy:3d} ({100*healthy/len(successful):.0f}%)")
        print(f"    Cliff crossed:       {cliff_crossed:3d} ({100*cliff_crossed/len(successful):.0f}%)")

    # ── Per-category summary ────────────────────────────────────────
    print(f"\n--- Per-Category Outcomes ---")
    print(f"{'Category':30s} {'N':>4s} {'OK':>4s} {'Crash':>6s} "
          f"{'ATP_min':>8s} {'ATP_max':>8s} {'het_max':>8s}")
    print("-" * 78)
    for cat in sorted(categories.keys()):
        cat_results = [r for r in successful if r["category"] == cat]
        cat_crashes = sum(1 for r in results
                         if r["category"] == cat and not r["success"])
        if cat_results:
            cat_atp = [r["atp_final"] for r in cat_results]
            cat_het = [r["het_final"] for r in cat_results]
            print(f"{cat:30s} {len(cat_results):4d} {len(cat_results):4d} "
                  f"{cat_crashes:6d} "
                  f"{min(cat_atp):8.4f} {max(cat_atp):8.4f} "
                  f"{max(cat_het):8.4f}")
        else:
            print(f"{cat:30s}    0    0 {cat_crashes:6d}     —        —        —")

    # ── Robustness score ────────────────────────────────────────────
    print(f"\n--- Robustness Score ---")
    scores = {
        "crash_rate": 1.0 - n_crash / n,
        "issue_rate": 1.0 - n_issues / max(n_success, 1),
        "state_validity": n_clean / max(n_success, 1),
    }
    overall = np.mean(list(scores.values()))
    for k, v in scores.items():
        print(f"  {k:30s}: {v:.2f}")
    print(f"  {'OVERALL':30s}: {overall:.2f}")

    print(f"\n{'='*70}")

    return {
        "n_patients": n,
        "n_success": n_success,
        "n_crash": n_crash,
        "n_issues": n_issues,
        "n_clean": n_clean,
        "errors": errors,
        "results": results,
        "robustness_scores": scores,
        "overall_score": float(overall),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def _save_population(patients, evaluation, out_path, description, **extra_meta):
    """Save patients + evaluation to JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_patients = []
    for p in patients:
        clean = {k: p[k] for k in PATIENT_NAMES}
        clean["_id"] = p["_id"]
        if "_label" in p:
            clean["_label"] = p["_label"]
        if "_category" in p:
            clean["_category"] = p["_category"]
        clean_patients.append(clean)

    metadata = {"n_patients": len(patients), "description": description,
                "generator": "generate_patients.py", **extra_meta}
    with open(out_path, "w") as f:
        json.dump({"metadata": metadata, "patients": clean_patients,
                   "evaluation": evaluation}, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge", action="store_true",
                        help="Generate edge-case patients instead of normal population")
    parser.add_argument("--both", action="store_true",
                        help="Generate both normal and edge-case populations")
    args = parser.parse_args()

    if args.both or not args.edge:
        print("Generating normal population...")
        patients = generate_patients(100, seed=42)
        evaluation = evaluate_population(patients)
        _save_population(
            patients, evaluation,
            PROJECT / "artifacts" / "sample_patients_100.json",
            "100 sample patients with biologically plausible correlations",
            seed=42)

    if args.both or args.edge:
        print("\nGenerating edge-case population...")
        edge_patients = generate_edge_patients()
        edge_evaluation = evaluate_edge_population(edge_patients)
        _save_population(
            edge_patients, edge_evaluation,
            PROJECT / "artifacts" / "sample_patients_edge.json",
            "Edge-case patients for robustness testing")
