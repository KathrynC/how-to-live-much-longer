#!/usr/bin/env python3
"""
categorical_structure.py

Formal validation of the categorical structure: Sem → Vec → Beh.

Adapted from categorical_structure.py in the parent Evolutionary-Robotics
project, which validated the functor F: Sem → Wt, the map G: Wt → Beh,
and their composition G∘F. Here:

  F: ClinicalSeed → InterventionVector (the LLM's clinical reasoning)
  G: InterventionVector → PatientOutcome (the ODE simulation)
  G∘F: ClinicalSeed → PatientOutcome (end-to-end)

Tests:
  1. Functoriality: Do nearby seeds produce nearby vectors?
  2. Continuity: Do nearby vectors produce nearby outcomes?
  3. Faithfulness: Do different seeds produce different outcomes?
  4. Sheaf consistency: Is the mapping locally coherent?
  5. Information geometry: Fisher-like analysis of the mapping

Requires prior experiment data from oeis_seed_experiment.json or
character_seed_experiment.json.

Scale: Pure computation on existing data (~5 seconds)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from constants import ALL_PARAM_NAMES
from analytics import NumpyEncoder


# ── Data loading ────────────────────────────────────────────────────────────

def load_experiment_data():
    """Load successful results from seed experiments.

    Returns list of dicts with: seed_label, vector_flat, outcome_flat, model
    """
    records = []

    # Try OEIS experiment
    oeis_path = PROJECT / "artifacts" / "oeis_seed_experiment.json"
    if oeis_path.exists():
        with open(oeis_path) as f:
            data = json.load(f)
        for r in data.get("results", []):
            if not r.get("success") or not r.get("intervention") or not r.get("analytics"):
                continue
            # Flatten intervention vector
            vec = []
            for k in ALL_PARAM_NAMES:
                v = r["intervention"].get(k, r.get("patient", {}).get(k, 0.0))
                vec.append(float(v))

            # Flatten outcome
            e = r["analytics"]["energy"]
            d = r["analytics"]["damage"]
            outcome = [
                e["atp_final"], e["atp_mean"], e["atp_slope"],
                d["het_final"], d["delta_het"], d["time_to_cliff_years"]
                if d["time_to_cliff_years"] < 900 else 30.0,
            ]

            records.append({
                "source": "oeis",
                "seed_label": f"{r['seq_id']}",
                "model": r["model"],
                "vector_flat": vec,
                "outcome_flat": outcome,
                "seq_terms": r.get("seq_terms", []),
            })

    # Try character experiment
    char_path = PROJECT / "artifacts" / "character_seed_experiment.json"
    if char_path.exists():
        with open(char_path) as f:
            data = json.load(f)
        for r in data.get("results", []):
            if not r.get("success") or not r.get("intervention") or not r.get("analytics"):
                continue
            vec = []
            for k in ALL_PARAM_NAMES:
                v = r["intervention"].get(k, r.get("patient", {}).get(k, 0.0))
                vec.append(float(v))

            e = r["analytics"]["energy"]
            d = r["analytics"]["damage"]
            outcome = [
                e["atp_final"], e["atp_mean"], e["atp_slope"],
                d["het_final"], d["delta_het"], d["time_to_cliff_years"]
                if d["time_to_cliff_years"] < 900 else 30.0,
            ]

            records.append({
                "source": "character",
                "seed_label": r.get("character_story", "unknown"),
                "model": r["model"],
                "vector_flat": vec,
                "outcome_flat": outcome,
            })

    return records


# ── Distance metrics ───────────────────────────────────────────────────────

def euclidean_distance(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def cosine_distance(a, b):
    a, b = np.array(a), np.array(b)
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return float(1.0 - dot / (na * nb))


# ── Semantic distance for OEIS sequences ────────────────────────────────────

def sequence_correlation_distance(terms_a, terms_b):
    """Distance between two integer sequences based on rank correlation."""
    n = min(len(terms_a), len(terms_b), 16)
    if n < 4:
        return 1.0
    a = np.array(terms_a[:n], dtype=float)
    b = np.array(terms_b[:n], dtype=float)
    # Normalize
    a = (a - np.mean(a)) / (np.std(a) + 1e-12)
    b = (b - np.mean(b)) / (np.std(b) + 1e-12)
    corr = float(np.corrcoef(a, b)[0, 1])
    return max(0.0, 1.0 - abs(corr))


# ── Analysis functions ─────────────────────────────────────────────────────

def test_functoriality(records):
    """Test: do nearby seeds produce nearby vectors?

    For OEIS records, use sequence correlation as semantic distance.
    Compute correlation between d_Sem and d_Vec.
    """
    oeis_records = [r for r in records if r["source"] == "oeis" and r.get("seq_terms")]

    if len(oeis_records) < 10:
        return {"status": "insufficient_data", "n_records": len(oeis_records)}

    # Pick one model for consistency
    models = set(r["model"] for r in oeis_records)
    model = sorted(models)[0]
    model_records = [r for r in oeis_records if r["model"] == model]

    if len(model_records) < 10:
        return {"status": "insufficient_data", "n_records": len(model_records)}

    # Compute pairwise distances
    n = len(model_records)
    d_sem = []
    d_vec = []
    d_beh = []

    for i in range(n):
        for j in range(i + 1, n):
            d_s = sequence_correlation_distance(
                model_records[i]["seq_terms"], model_records[j]["seq_terms"])
            d_v = euclidean_distance(
                model_records[i]["vector_flat"], model_records[j]["vector_flat"])
            d_b = euclidean_distance(
                model_records[i]["outcome_flat"], model_records[j]["outcome_flat"])
            d_sem.append(d_s)
            d_vec.append(d_v)
            d_beh.append(d_b)

    d_sem, d_vec, d_beh = np.array(d_sem), np.array(d_vec), np.array(d_beh)

    # Correlations
    corr_sem_vec = float(np.corrcoef(d_sem, d_vec)[0, 1]) if len(d_sem) > 2 else 0.0
    corr_vec_beh = float(np.corrcoef(d_vec, d_beh)[0, 1]) if len(d_vec) > 2 else 0.0
    corr_sem_beh = float(np.corrcoef(d_sem, d_beh)[0, 1]) if len(d_sem) > 2 else 0.0

    return {
        "status": "computed",
        "model": model,
        "n_pairs": len(d_sem),
        "n_records": len(model_records),
        "corr_sem_vec": corr_sem_vec,  # F functoriality
        "corr_vec_beh": corr_vec_beh,  # G continuity
        "corr_sem_beh": corr_sem_beh,  # G∘F composition
        "mean_d_sem": float(np.mean(d_sem)),
        "mean_d_vec": float(np.mean(d_vec)),
        "mean_d_beh": float(np.mean(d_beh)),
    }


def test_faithfulness(records):
    """Test: do different seeds produce different outcomes?

    Group by seed, check if within-seed outcome variance < between-seed variance.
    """
    seed_outcomes = defaultdict(list)
    for r in records:
        seed_outcomes[r["seed_label"]].append(r["outcome_flat"])

    # Only consider seeds with multiple data points (multiple models)
    multi_seeds = {k: v for k, v in seed_outcomes.items() if len(v) >= 2}

    if len(multi_seeds) < 5:
        return {"status": "insufficient_data"}

    # Within-seed variance (average variance of outcome within same seed)
    within_vars = []
    for seed, outcomes in multi_seeds.items():
        mat = np.array(outcomes)
        within_vars.append(float(np.mean(np.var(mat, axis=0))))

    # Between-seed variance (variance of seed means)
    seed_means = []
    for seed, outcomes in multi_seeds.items():
        seed_means.append(np.mean(outcomes, axis=0))
    between_var = float(np.mean(np.var(np.array(seed_means), axis=0)))

    mean_within = float(np.mean(within_vars))
    # F-ratio: between/within. >1 means seeds produce distinct outcomes
    f_ratio = between_var / (mean_within + 1e-12)

    return {
        "status": "computed",
        "n_seeds": len(multi_seeds),
        "mean_within_variance": mean_within,
        "between_variance": between_var,
        "f_ratio": f_ratio,
        "faithful": f_ratio > 1.0,
    }


def test_model_consistency(records):
    """Test: do different models map the same seed to similar outcomes?

    Measures whether model choice affects the functor more than seed choice.
    """
    # Group by (seed, model)
    seed_model = defaultdict(dict)
    for r in records:
        seed_model[r["seed_label"]][r["model"]] = r["outcome_flat"]

    # For seeds with all models, compute inter-model agreement
    n_models = len(set(r["model"] for r in records))
    full_seeds = {k: v for k, v in seed_model.items() if len(v) >= n_models}

    if len(full_seeds) < 3:
        return {"status": "insufficient_data"}

    # Per-seed inter-model cosine similarity
    similarities = []
    for seed, model_outcomes in full_seeds.items():
        outcomes = list(model_outcomes.values())
        for i in range(len(outcomes)):
            for j in range(i + 1, len(outcomes)):
                sim = 1.0 - cosine_distance(outcomes[i], outcomes[j])
                similarities.append(sim)

    return {
        "status": "computed",
        "n_full_seeds": len(full_seeds),
        "mean_intermodel_similarity": float(np.mean(similarities)),
        "std_intermodel_similarity": float(np.std(similarities)),
        "min_intermodel_similarity": float(np.min(similarities)),
    }


def compute_information_geometry(records):
    """Compute information-geometric summary of the Vec → Beh mapping.

    Gram matrix of outcome vectors, effective dimensionality.
    """
    outcomes = np.array([r["outcome_flat"] for r in records])
    vectors = np.array([r["vector_flat"] for r in records])

    if len(outcomes) < 5:
        return {"status": "insufficient_data"}

    # Normalize
    o_centered = outcomes - np.mean(outcomes, axis=0)
    v_centered = vectors - np.mean(vectors, axis=0)

    # Gram matrices
    outcome_gram = o_centered @ o_centered.T
    vector_gram = v_centered @ v_centered.T

    # Eigenspectrum of outcome space
    o_eigenvalues = np.linalg.eigvalsh(outcome_gram)
    o_eigenvalues = o_eigenvalues[o_eigenvalues > 1e-10]
    if len(o_eigenvalues) > 0:
        probs = o_eigenvalues / o_eigenvalues.sum()
        o_eff_dim = float(np.exp(-np.sum(probs * np.log(probs + 1e-15))))
    else:
        o_eff_dim = 0.0

    # Eigenspectrum of vector space
    v_eigenvalues = np.linalg.eigvalsh(vector_gram)
    v_eigenvalues = v_eigenvalues[v_eigenvalues > 1e-10]
    if len(v_eigenvalues) > 0:
        probs = v_eigenvalues / v_eigenvalues.sum()
        v_eff_dim = float(np.exp(-np.sum(probs * np.log(probs + 1e-15))))
    else:
        v_eff_dim = 0.0

    # How much of vector variation maps to outcome variation?
    # CCA-like: correlation between vector PCA projections and outcome PCA projections
    # Simple: just correlate singular values
    return {
        "status": "computed",
        "n_points": len(records),
        "outcome_effective_dim": o_eff_dim,
        "vector_effective_dim": v_eff_dim,
        "dim_compression_ratio": o_eff_dim / (v_eff_dim + 1e-12),
        "outcome_dim_actual": outcomes.shape[1],
        "vector_dim_actual": vectors.shape[1],
    }


# ── Main ────────────────────────────────────────────────────────────────────

def run_analysis():
    out_path = PROJECT / "artifacts" / "categorical_structure.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"CATEGORICAL STRUCTURE ANALYSIS — Functor Validation")
    print(f"{'=' * 70}")

    records = load_experiment_data()
    print(f"Loaded {len(records)} records "
          f"({sum(1 for r in records if r['source']=='oeis')} OEIS, "
          f"{sum(1 for r in records if r['source']=='character')} character)")

    if len(records) < 10:
        print("ERROR: Need at least 10 records. Run oeis_seed_experiment.py "
              "or character_seed_experiment.py first.")
        return

    # Run all tests
    print("\n--- Test 1: Functoriality (Sem → Vec → Beh distance correlation) ---")
    functoriality = test_functoriality(records)
    if functoriality["status"] == "computed":
        print(f"  F (Sem→Vec):  corr = {functoriality['corr_sem_vec']:.3f}")
        print(f"  G (Vec→Beh):  corr = {functoriality['corr_vec_beh']:.3f}")
        print(f"  G∘F (Sem→Beh): corr = {functoriality['corr_sem_beh']:.3f}")
    else:
        print(f"  {functoriality['status']}")

    print("\n--- Test 2: Faithfulness (different seeds → different outcomes) ---")
    faithfulness = test_faithfulness(records)
    if faithfulness["status"] == "computed":
        print(f"  F-ratio: {faithfulness['f_ratio']:.3f} "
              f"({'FAITHFUL' if faithfulness['faithful'] else 'NOT FAITHFUL'})")
        print(f"  Within-seed variance: {faithfulness['mean_within_variance']:.4f}")
        print(f"  Between-seed variance: {faithfulness['between_variance']:.4f}")
    else:
        print(f"  {faithfulness['status']}")

    print("\n--- Test 3: Model Consistency ---")
    consistency = test_model_consistency(records)
    if consistency["status"] == "computed":
        print(f"  Mean inter-model similarity: {consistency['mean_intermodel_similarity']:.3f}")
        print(f"  Min inter-model similarity: {consistency['min_intermodel_similarity']:.3f}")
    else:
        print(f"  {consistency['status']}")

    print("\n--- Test 4: Information Geometry ---")
    info_geom = compute_information_geometry(records)
    if info_geom["status"] == "computed":
        print(f"  Vector space effective dim: {info_geom['vector_effective_dim']:.2f} / {info_geom['vector_dim_actual']}")
        print(f"  Outcome space effective dim: {info_geom['outcome_effective_dim']:.2f} / {info_geom['outcome_dim_actual']}")
        print(f"  Compression ratio: {info_geom['dim_compression_ratio']:.3f}")
    else:
        print(f"  {info_geom['status']}")

    # Save
    output = {
        "experiment": "categorical_structure",
        "date": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "n_records": len(records),
        "functoriality": functoriality,
        "faithfulness": faithfulness,
        "model_consistency": consistency,
        "information_geometry": info_geom,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_analysis()
