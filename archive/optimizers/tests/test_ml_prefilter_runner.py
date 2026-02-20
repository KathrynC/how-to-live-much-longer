"""Tests for ml_prefilter_runner additive ML workflow."""

from __future__ import annotations

import numpy as np

import ml_prefilter_runner as mpr
import surrogate_optimizer as so


def test_build_candidate_pool_includes_seed_variants():
    rng = np.random.default_rng(3)
    seed = {k: 0.5 for k in so.INTERVENTION_NAMES}
    pool = mpr.build_candidate_pool(
        pool_size=20,
        rng=rng,
        seed_protocols=[seed],
        perturb_per_seed=10,
        perturb_sigma=0.05,
    )
    # 20 random + 1 seed + 10 perturbations.
    assert len(pool) == 31
    for iv in pool:
        for k in so.INTERVENTION_NAMES:
            lo, hi = so.INTERVENTION_PARAMS[k]["range"]
            assert lo <= iv[k] <= hi


def test_run_prefilter_toy_objective_lift_positive():
    # Use deterministic monotonic objective so the surrogate can exploit it.
    patient = dict(so.DEFAULT_PATIENT)

    def toy_objective(iv):
        # Strongly depends on rapamycin + nad; easy to learn.
        fit = 2.0 * iv["rapamycin_dose"] + 1.5 * iv["nad_supplement"] - 0.1 * iv["yamanaka_intensity"]
        return {
            "fitness": float(fit),
            "final_atp": float(fit),
            "final_het": float(1.0 - fit),
            "atp_benefit": float(fit),
            "het_benefit": float(fit),
        }

    # Monkeypatch objective builder to avoid simulator cost.
    orig_make = mpr.make_objective
    orig_profiles = dict(mpr.PATIENT_PROFILES)
    try:
        mpr.make_objective = lambda patient, metric: toy_objective
        mpr.PATIENT_PROFILES = {"default": patient}
        out = mpr.run_prefilter(
            patient_name="default",
            metric="combined",
            train_samples=80,
            pool_size=300,
            top_k=20,
            seed=9,
            seed_protocols=[],
            perturb_per_seed=0,
            perturb_sigma=0.0,
            knn_k=9,
        )
    finally:
        mpr.make_objective = orig_make
        mpr.PATIENT_PROFILES = orig_profiles

    assert out["mean_true_lift_vs_random"] > 0.0
    assert len(out["top_ranked_true"]) == 20
