#!/usr/bin/env python3
"""
llm_seeded_evolution.py

Test whether LLM-generated intervention vectors are better starting points
for optimization than random vectors.

Adapted from llm_seeded_evolution.py in the parent Evolutionary-Robotics
project, which tested whether LLM weight seeds were a "launchpad or trap"
for evolutionary optimization. Here we hill-climb from LLM-designed
protocols and compare to random starting points.

Experiments:
  - 20 LLM seeds + 20 random seeds × 100 hill-climb evaluations each
  - Single-parameter mutations (±1 grid step)
  - Fitness = ATP benefit over no-treatment baseline
  - Compare: final fitness, improvement trajectory, convergence speed

Scale: 40 seeds × 100 evaluations = 4000 sims
Estimated time: ~10 minutes (pure simulation after initial LLM queries)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from constants import (
    INTERVENTION_PARAMS,
    INTERVENTION_NAMES, ALL_PARAM_NAMES,
    DEFAULT_INTERVENTION,
)
from simulator import simulate
from analytics import NumpyEncoder


# ── Configuration ───────────────────────────────────────────────────────────

N_LLM_SEEDS = 20
N_RANDOM_SEEDS = 20
EVAL_BUDGET = 100  # hill-climb evaluations per seed

# Fixed patient for fair comparison
TEST_PATIENT = {
    "baseline_age": 60.0, "baseline_heteroplasmy": 0.40,
    "baseline_nad_level": 0.6, "genetic_vulnerability": 1.0,
    "metabolic_demand": 1.0, "inflammation_level": 0.3,
}


# ── Fitness function ───────────────────────────────────────────────────────

# Cache the baseline simulation
_baseline_cache = {}

def get_baseline():
    """Get cached baseline simulation for TEST_PATIENT."""
    key = str(TEST_PATIENT)
    if key not in _baseline_cache:
        _baseline_cache[key] = simulate(
            intervention=DEFAULT_INTERVENTION, patient=TEST_PATIENT)
    return _baseline_cache[key]


def fitness(intervention):
    """Compute fitness as ATP benefit over no-treatment baseline.

    Returns float. Higher = better.
    """
    result = simulate(intervention=intervention, patient=TEST_PATIENT)
    baseline = get_baseline()

    final_atp = float(result["states"][-1, 2])
    base_atp = float(baseline["states"][-1, 2])
    final_het = float(result["heteroplasmy"][-1])
    base_het = float(baseline["heteroplasmy"][-1])

    # Combined fitness: ATP benefit + heteroplasmy benefit
    atp_benefit = final_atp - base_atp
    het_benefit = base_het - final_het  # positive = less damage

    return atp_benefit + 0.5 * het_benefit


# ── Mutation operator ──────────────────────────────────────────────────────

def mutate(intervention, rng):
    """Mutate a single random intervention parameter by ±1 grid step.

    Returns new intervention dict.
    """
    mutated = dict(intervention)
    param = rng.choice(INTERVENTION_NAMES)
    grid = sorted(INTERVENTION_PARAMS[param]["grid"])
    current = mutated[param]

    # Find nearest grid index
    current_grid = min(grid, key=lambda g: abs(g - current))
    idx = grid.index(current_grid)

    # Pick direction
    if idx == 0:
        new_idx = 1
    elif idx == len(grid) - 1:
        new_idx = len(grid) - 2
    else:
        new_idx = idx + rng.choice([-1, 1])

    mutated[param] = grid[new_idx]
    return mutated


# ── Hill climber ───────────────────────────────────────────────────────────

def hill_climb(initial_intervention, budget, rng):
    """Simple (1+1) hill climber.

    Returns:
        (best_intervention, best_fitness, fitness_trajectory)
    """
    current = dict(initial_intervention)
    current_fit = fitness(current)
    trajectory = [current_fit]

    for step in range(budget):
        candidate = mutate(current, rng)
        candidate_fit = fitness(candidate)

        if candidate_fit > current_fit:
            current = candidate
            current_fit = candidate_fit

        trajectory.append(current_fit)

    return current, current_fit, trajectory


# ── Random intervention generation ─────────────────────────────────────────

def random_intervention(rng):
    """Generate a random intervention vector snapped to grid."""
    intervention = {}
    for name, spec in INTERVENTION_PARAMS.items():
        intervention[name] = float(rng.choice(spec["grid"]))
    return intervention


# ── Load LLM-generated vectors ─────────────────────────────────────────────

def load_llm_seeds(max_seeds=N_LLM_SEEDS):
    """Load top-performing LLM vectors from prior experiments."""
    seeds = []

    for filename in ["oeis_seed_experiment.json", "character_seed_experiment.json"]:
        path = PROJECT / "artifacts" / filename
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for r in data.get("results", []):
            if r.get("success") and r.get("intervention") and r.get("analytics"):
                benefit = r["analytics"]["intervention"]["atp_benefit_terminal"]
                label = r.get("seq_id") or r.get("character_story", "unknown")
                seeds.append({
                    "label": f"{label} ({r.get('model', '?')})",
                    "intervention": r["intervention"],
                    "initial_benefit": benefit,
                })

    # Sort by benefit and take top N
    seeds.sort(key=lambda s: s["initial_benefit"], reverse=True)
    return seeds[:max_seeds]


def generate_fresh_llm_seeds(n_seeds=N_LLM_SEEDS):
    """Generate LLM seeds from clinical scenarios if no prior data exists."""
    from constants import CLINICAL_SEEDS
    from llm_common import query_ollama as qo, split_vector, MODELS

    seeds = []
    model = MODELS[0]["name"]  # Use primary model

    prompt_template = (
        "Design a mitochondrial intervention protocol for a 60-year-old with "
        "40% heteroplasmy. Output JSON with keys: rapamycin_dose, nad_supplement, "
        "senolytic_dose, yamanaka_intensity, transplant_rate, exercise_level "
        "(all 0.0-1.0). Scenario: {scenario}\n"
        "Output ONLY the JSON object."
    )

    for i, seed in enumerate(CLINICAL_SEEDS[:n_seeds]):
        prompt = prompt_template.format(scenario=seed["description"])
        vector, _ = qo(model, prompt, temperature=0.8)
        if vector:
            intervention, _ = split_vector(vector)
            seeds.append({
                "label": f"clinical:{seed['id']}",
                "intervention": intervention,
            })
            print(f"  Generated LLM seed {i+1}/{n_seeds}: {seed['id']}")

    return seeds


# ── Main experiment ─────────────────────────────────────────────────────────

def run_experiment(seed=42):
    out_path = PROJECT / "artifacts" / "llm_seeded_evolution.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    start_time = time.time()

    print(f"{'=' * 70}")
    print(f"LLM-SEEDED EVOLUTION — Launchpad or Trap?")
    print(f"{'=' * 70}")
    print(f"LLM seeds: {N_LLM_SEEDS}, Random seeds: {N_RANDOM_SEEDS}")
    print(f"Evaluation budget per seed: {EVAL_BUDGET}")
    print(f"Total evaluations: ~{(N_LLM_SEEDS + N_RANDOM_SEEDS) * (EVAL_BUDGET + 1)}")
    print()

    # Load or generate LLM seeds
    llm_seeds = load_llm_seeds()
    if len(llm_seeds) < N_LLM_SEEDS:
        print(f"Only found {len(llm_seeds)} prior LLM vectors, generating fresh ones...")
        fresh = generate_fresh_llm_seeds(N_LLM_SEEDS - len(llm_seeds))
        llm_seeds.extend(fresh)

    if not llm_seeds:
        print("No LLM seeds available. Using random seeds with varied starting fitness.")
        # Generate "structured" random seeds (not truly random — use interesting points)
        for i in range(N_LLM_SEEDS):
            intv = random_intervention(rng)
            # Bias toward moderate doses to simulate "LLM-like" starting points
            for k in intv:
                if rng.random() < 0.5:
                    intv[k] = min(intv[k], 0.5)
            llm_seeds.append({"label": f"pseudo_llm_{i}", "intervention": intv})

    print(f"LLM seeds ready: {len(llm_seeds)}")

    # Generate random seeds
    random_seeds = []
    for i in range(N_RANDOM_SEEDS):
        random_seeds.append({
            "label": f"random_{i}",
            "intervention": random_intervention(rng),
        })

    # ── Run hill climbers ──────────────────────────────────────────────────
    llm_results = []
    random_results = []

    print(f"\n--- LLM-seeded hill climbing ---")
    for i, seed_info in enumerate(llm_seeds[:N_LLM_SEEDS]):
        initial_fit = fitness(seed_info["intervention"])
        best_intv, best_fit, trajectory = hill_climb(
            seed_info["intervention"], EVAL_BUDGET, rng)
        improvement = best_fit - initial_fit

        print(f"  [{i+1}/{N_LLM_SEEDS}] {seed_info['label'][:45]:45s}: "
              f"{initial_fit:+.3f} → {best_fit:+.3f} ({improvement:+.3f})")

        llm_results.append({
            "seed_type": "llm",
            "label": seed_info["label"],
            "initial_intervention": seed_info["intervention"],
            "initial_fitness": initial_fit,
            "final_intervention": best_intv,
            "final_fitness": best_fit,
            "improvement": improvement,
            "trajectory": trajectory,
        })

    print(f"\n--- Random-seeded hill climbing ---")
    for i, seed_info in enumerate(random_seeds):
        initial_fit = fitness(seed_info["intervention"])
        best_intv, best_fit, trajectory = hill_climb(
            seed_info["intervention"], EVAL_BUDGET, rng)
        improvement = best_fit - initial_fit

        print(f"  [{i+1}/{N_RANDOM_SEEDS}] {seed_info['label']:45s}: "
              f"{initial_fit:+.3f} → {best_fit:+.3f} ({improvement:+.3f})")

        random_results.append({
            "seed_type": "random",
            "label": seed_info["label"],
            "initial_intervention": seed_info["intervention"],
            "initial_fitness": initial_fit,
            "final_intervention": best_intv,
            "final_fitness": best_fit,
            "improvement": improvement,
            "trajectory": trajectory,
        })

    elapsed = time.time() - start_time

    # ── Analysis ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"LLM-SEEDED EVOLUTION COMPLETE — {elapsed:.1f}s")
    print(f"{'=' * 70}")

    llm_initial = [r["initial_fitness"] for r in llm_results]
    llm_final = [r["final_fitness"] for r in llm_results]
    llm_improvement = [r["improvement"] for r in llm_results]

    rnd_initial = [r["initial_fitness"] for r in random_results]
    rnd_final = [r["final_fitness"] for r in random_results]
    rnd_improvement = [r["improvement"] for r in random_results]

    print(f"\n{'':30s} {'LLM seeds':>15s} {'Random seeds':>15s}")
    print("-" * 62)
    print(f"{'Initial fitness (median)':30s} {np.median(llm_initial):>15.3f} {np.median(rnd_initial):>15.3f}")
    print(f"{'Final fitness (median)':30s} {np.median(llm_final):>15.3f} {np.median(rnd_final):>15.3f}")
    print(f"{'Improvement (median)':30s} {np.median(llm_improvement):>15.3f} {np.median(rnd_improvement):>15.3f}")
    print(f"{'Initial fitness (max)':30s} {max(llm_initial):>15.3f} {max(rnd_initial):>15.3f}")
    print(f"{'Final fitness (max)':30s} {max(llm_final):>15.3f} {max(rnd_final):>15.3f}")

    # Statistical comparison
    llm_wins_final = sum(1 for l, r in zip(sorted(llm_final, reverse=True),
                                            sorted(rnd_final, reverse=True))
                         if l > r)
    print(f"\nLLM > Random (final fitness): {llm_wins_final}/{min(len(llm_final), len(rnd_final))}")

    # Convergence speed: how many evaluations to reach 90% of final improvement
    def convergence_speed(trajectory):
        if len(trajectory) < 2:
            return 0
        total_improvement = trajectory[-1] - trajectory[0]
        if total_improvement <= 0:
            return len(trajectory)
        target = trajectory[0] + 0.9 * total_improvement
        for i, f in enumerate(trajectory):
            if f >= target:
                return i
        return len(trajectory)

    llm_speeds = [convergence_speed(r["trajectory"]) for r in llm_results]
    rnd_speeds = [convergence_speed(r["trajectory"]) for r in random_results]
    print(f"Convergence speed (median evals to 90%): "
          f"LLM={np.median(llm_speeds):.0f}, Random={np.median(rnd_speeds):.0f}")

    # Verdict
    if np.median(llm_final) > np.median(rnd_final) + 0.01:
        verdict = "LAUNCHPAD — LLM seeds reach higher fitness"
    elif np.median(llm_final) < np.median(rnd_final) - 0.01:
        verdict = "TRAP — LLM seeds get stuck in local optima"
    else:
        verdict = "NEUTRAL — LLM and random seeds converge similarly"
    print(f"\nVerdict: {verdict}")

    # Save
    output = {
        "experiment": "llm_seeded_evolution",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "eval_budget": EVAL_BUDGET,
        "n_llm_seeds": len(llm_results),
        "n_random_seeds": len(random_results),
        "verdict": verdict,
        "summary": {
            "llm_initial_median": float(np.median(llm_initial)),
            "llm_final_median": float(np.median(llm_final)),
            "random_initial_median": float(np.median(rnd_initial)),
            "random_final_median": float(np.median(rnd_final)),
            "llm_convergence_median": float(np.median(llm_speeds)),
            "random_convergence_median": float(np.median(rnd_speeds)),
        },
        "llm_results": llm_results,
        "random_results": random_results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


# ── Narrative evolution (Zimmerman-informed) ─────────────────────────────────
# Instead of blind hill-climbing, feed the LLM the trajectory of previous
# attempts and ask it to adjust. This creates a feedback loop where the
# LLM learns from simulation outcomes (Zimmerman's meaning-from-context).

NARRATIVE_PROMPT = """\
You are a mitochondrial medicine specialist iteratively refining a protocol.

Patient: 60 years old, 40% heteroplasmy, NAD=0.6, normal vulnerability.

Your PREVIOUS protocol was:
  Rapamycin: {rapa}, NAD+: {nad}, Senolytics: {seno}
  Yamanaka: {yama}, Transplant: {trans}, Exercise: {ex}

After 30-year simulation, the results were:
  Heteroplasmy: {het_init:.3f} -> {het_final:.3f} (change: {het_delta:+.3f})
  ATP: {atp_init:.3f} -> {atp_final:.3f} (change: {atp_delta:+.3f})
  {feedback}

This was iteration {iteration} of {max_iter}. Your best result so far:
  Best het change: {best_het:+.3f}, Best ATP change: {best_atp:+.3f}

Adjust your protocol to improve outcomes. Consider:
- If het increased too much, boost rapamycin or reduce Yamanaka
- If ATP dropped, add more NAD+ or reduce energy-costly interventions
- If both are good, try small tweaks to optimize further

Output ONLY a JSON object with the 6 intervention keys (0.0-1.0):
{{"rapamycin_dose":_, "nad_supplement":_, "senolytic_dose":_, \
"yamanaka_intensity":_, "transplant_rate":_, "exercise_level":_}}"""


def run_narrative_evolution(n_iterations=10, model=None):
    """Run narrative-feedback LLM optimization.

    Instead of blind mutations, the LLM sees previous results and
    adjusts its protocol accordingly.
    """
    from llm_common import query_ollama_raw, parse_json_response, MODELS

    if model is None:
        model = MODELS[0]["name"]

    out_path = PROJECT / "artifacts" / "narrative_evolution.json"

    print(f"{'=' * 70}")
    print(f"NARRATIVE EVOLUTION — LLM-in-the-Loop Optimization")
    print(f"  Model: {model}")
    print(f"  Iterations: {n_iterations}")
    print(f"{'=' * 70}")

    # Start with LLM's initial guess
    initial_prompt = (
        "Design a mitochondrial intervention protocol for a 60-year-old with "
        "40% heteroplasmy. Output ONLY a JSON object with keys: "
        "rapamycin_dose, nad_supplement, senolytic_dose, yamanaka_intensity, "
        "transplant_rate, exercise_level (all 0.0-1.0)."
    )
    resp = query_ollama_raw(model, initial_prompt, temperature=0.7)
    current = parse_json_response(resp)
    if current is None:
        current = dict(DEFAULT_INTERVENTION)
        current["rapamycin_dose"] = 0.5
        current["nad_supplement"] = 0.5

    # Ensure all keys present
    intervention = {k: float(current.get(k, DEFAULT_INTERVENTION[k]))
                    for k in INTERVENTION_NAMES}

    history = []
    best_fitness = -999
    best_het_change = 999
    best_atp_change = -999
    best_intervention = dict(intervention)

    for iteration in range(n_iterations):
        # Simulate
        result = simulate(intervention=intervention, patient=TEST_PATIENT)
        baseline = get_baseline()

        het_init = float(result["heteroplasmy"][0])
        het_final = float(result["heteroplasmy"][-1])
        het_delta = het_final - het_init
        atp_init = float(result["states"][0, 2])
        atp_final = float(result["states"][-1, 2])
        atp_delta = atp_final - atp_init

        fit = fitness(intervention)

        # Track best
        if fit > best_fitness:
            best_fitness = fit
            best_intervention = dict(intervention)
            best_het_change = het_delta
            best_atp_change = atp_delta

        # Generate feedback text
        if het_delta > 0.1:
            feedback = "WARNING: Heteroplasmy increased significantly. More mitophagy needed."
        elif het_delta < -0.05:
            feedback = "Good: heteroplasmy decreased."
        else:
            feedback = "Heteroplasmy roughly stable."

        if atp_delta < -0.2:
            feedback += " ATP dropped — energy cost may be too high."
        elif atp_delta > 0:
            feedback += " ATP improved."

        print(f"  [{iteration+1}/{n_iterations}] "
              f"het={het_delta:+.3f} ATP={atp_delta:+.3f} "
              f"fit={fit:+.3f}  "
              f"[rapa={intervention['rapamycin_dose']:.2f} "
              f"nad={intervention['nad_supplement']:.2f} "
              f"yama={intervention['yamanaka_intensity']:.2f}]")

        history.append({
            "iteration": iteration,
            "intervention": dict(intervention),
            "het_change": het_delta,
            "atp_change": atp_delta,
            "fitness": fit,
        })

        # Ask LLM to adjust
        if iteration < n_iterations - 1:
            prompt = NARRATIVE_PROMPT.format(
                rapa=intervention["rapamycin_dose"],
                nad=intervention["nad_supplement"],
                seno=intervention["senolytic_dose"],
                yama=intervention["yamanaka_intensity"],
                trans=intervention["transplant_rate"],
                ex=intervention["exercise_level"],
                het_init=het_init, het_final=het_final, het_delta=het_delta,
                atp_init=atp_init, atp_final=atp_final, atp_delta=atp_delta,
                feedback=feedback,
                iteration=iteration + 1, max_iter=n_iterations,
                best_het=best_het_change, best_atp=best_atp_change,
            )
            resp = query_ollama_raw(model, prompt, temperature=0.5)
            new_params = parse_json_response(resp)
            if new_params:
                intervention = {k: float(new_params.get(k, intervention[k]))
                                for k in INTERVENTION_NAMES}
                # Clamp to valid range
                for k in intervention:
                    intervention[k] = max(0.0, min(1.0, intervention[k]))

    # Summary
    print(f"\n{'=' * 70}")
    print(f"NARRATIVE EVOLUTION COMPLETE")
    print(f"  Best fitness: {best_fitness:+.3f}")
    print(f"  Best het change: {best_het_change:+.3f}")
    print(f"  Best ATP change: {best_atp_change:+.3f}")
    print(f"  Best protocol: {best_intervention}")
    print(f"{'=' * 70}")

    # Compare with hill-climbing
    rng = np.random.RandomState(42)
    hc_best, hc_fit, hc_traj = hill_climb(
        history[0]["intervention"], n_iterations, rng)
    print(f"\n  Comparison (same budget={n_iterations} evals):")
    print(f"    Narrative:     {best_fitness:+.3f}")
    print(f"    Hill-climbing: {hc_fit:+.3f}")
    if best_fitness > hc_fit + 0.01:
        print(f"    -> Narrative wins (+{best_fitness - hc_fit:.3f})")
    elif hc_fit > best_fitness + 0.01:
        print(f"    -> Hill-climbing wins (+{hc_fit - best_fitness:.3f})")
    else:
        print(f"    -> Roughly equal")

    output = {
        "experiment": "narrative_evolution",
        "model": model,
        "n_iterations": n_iterations,
        "best_fitness": best_fitness,
        "best_intervention": best_intervention,
        "history": history,
        "hill_climb_comparison": hc_fit,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    if "--narrative" in sys.argv:
        n_iter = 10
        if "--iterations" in sys.argv:
            idx = sys.argv.index("--iterations")
            if idx + 1 < len(sys.argv):
                n_iter = int(sys.argv[idx + 1])
        run_narrative_evolution(n_iterations=n_iter)
    else:
        run_experiment()
