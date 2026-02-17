"""Tests for new additive search method modules."""

from __future__ import annotations

import bayesian_optimizer as bo
import nsga2_optimizer as nsga2
import robust_optimizer as ro
import active_learning_optimizer as alo
import map_elites_optimizer as me
import search_addons as sa


def test_non_dominated_sort_basic():
    rows = [
        {"a": 1.0, "b": 0.0},  # nondom
        {"a": 0.0, "b": 1.0},  # nondom
        {"a": 0.0, "b": 0.0},  # dominated by either extreme point
    ]
    fronts = sa.non_dominated_sort(rows, keys=["a", "b"])
    assert set(fronts[0]) == {0, 1}
    assert 2 in fronts[1]


def test_bayes_optimizer_small_run():
    out = bo.run_bo(
        patient_name="default",
        metric="combined",
        n_init=4,
        iterations=4,
        candidate_pool=30,
        kappa=1.0,
        seed=11,
    )
    assert out["n_evals"] == 8
    assert "best" in out and "fitness" in out["best"]


def test_nsga2_small_run():
    out = nsga2.run_nsga2(
        patient_name="default",
        metric="combined",
        pop_size=10,
        generations=3,
        sigma=0.07,
        seed=12,
    )
    assert out["n_evals"] == 40
    assert out["pareto_count"] >= 1


def test_robust_optimizer_small_run():
    out = ro.run_robust(
        patient_name="default",
        metric="combined",
        budget=8,
        ensemble_size=5,
        sigma=0.07,
        seed=13,
    )
    assert len(out["history"]) == 9
    assert "best" in out and "robust_fitness" in out["best"]


def test_active_learning_small_run():
    out = alo.run_active_learning(
        patient_name="default",
        metric="combined",
        n_init=5,
        rounds=3,
        propose_per_round=2,
        seed=14,
    )
    assert out["n_evals"] == 11
    assert len(out["round_summaries"]) == 3


def test_map_elites_small_run():
    out = me.run_map_elites(
        patient_name="default",
        metric="combined",
        budget=60,
        bins_atp=6,
        bins_het=6,
        sigma=0.08,
        seed=15,
    )
    assert out["n_elites"] >= 1
    assert 0.0 <= out["coverage"] <= 1.0
