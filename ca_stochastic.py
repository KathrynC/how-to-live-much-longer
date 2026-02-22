"""Stochastic variant of the mitochondrial semantic cellular automaton.

Provides a stochastic rule application engine and an ensemble runner for
Monte Carlo analysis of CA dynamics. While the deterministic CA in ca_rules.py
resolves conflicts by highest-confidence-wins, this module samples proportional
to confidence, enabling probabilistic outcome estimation.

Three public functions:

1. apply_rules_stochastic() -- stochastic rule application with confidence-
   weighted sampling. Low-confidence rules may not fire; multi-rule conflicts
   are resolved by confidence-proportional random selection.

2. run_single_cell_stochastic() -- Monte Carlo ensemble of single-cell CA
   trajectories. Each trial uses a distinct RNG stream derived from
   (seed + trial_index).

3. compute_ensemble_analytics() -- aggregate statistics over ensemble trials:
   attractor distributions, cliff crossing probability, per-variable terminal
   bin distributions, time-to-crisis statistics.

Mirrors ~/lemurs-simulator/ca_stochastic.py architecture.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ca_schema import BIN_SCHEMA, CA_VAR_ORDER
from ca_rules import get_applicable_rules, _apply_direction, RULE_TABLE


# ── Attractor classifier (inlined to avoid dependency on ca_analytics.py) ──

def _classify_attractor(state: dict[str, str]) -> str:
    """Classify a discrete state into one of four attractor basins.

    This matches the classifier in ca_simulator._compute_tissue_summary and
    is inlined here to avoid a circular import with ca_analytics.py (which
    may import from this module).

    Attractor basins:
        point_of_no_return -- all repair pathways failed
        cliff_approaching  -- deletion het near or past cliff
        slow_decline       -- gradual deterioration
        healthy_aging      -- maintaining function
    """
    n_del = state.get("N_deletion", "minimal")
    atp = state.get("ATP", "healthy")
    ros = state.get("ROS", "basal")
    sen = state.get("Senescent_fraction", "minimal")

    if (n_del == "past_cliff" and atp == "collapsed"
            and ros == "pathological" and sen == "severe"):
        return "point_of_no_return"
    if n_del in ("approaching_cliff", "past_cliff"):
        return "cliff_approaching"
    if atp in ("compromised", "crisis") or n_del == "growing":
        return "slow_decline"
    return "healthy_aging"


# ── Stochastic rule application ──────────────────────────────────────────


def apply_rules_stochastic(
    discrete_state: dict[str, str],
    context: dict,
    rng: np.random.Generator,
    rules: list[dict] | None = None,
) -> tuple[dict[str, str], list[dict]]:
    """Apply matching rules stochastically to produce the next discrete state.

    Rules are selected from those whose input and context conditions match.
    The point_of_no_return (absorbing state) is still deterministic -- if it
    matches, the state is frozen. For other rules:

    - Every rule fires with probability equal to its confidence. A rule
      with confidence 0.85 fires 85% of the time and is skipped 15% of
      the time. This generates genuine distributional spread across
      Monte Carlo trials.
    - When multiple surviving rules propose updates to the same variable,
      one is sampled proportional to its confidence weight.
    - When only one surviving rule proposes an update, it is applied
      directly (it already passed the probabilistic gate).

    Parameters
    ----------
    discrete_state : dict[str, str]
        Current discretized state {var_name: bin_label}.
    context : dict
        Age epoch, intervention levels, and other context.
    rng : np.random.Generator
        Random number generator for stochastic decisions.
    rules : list[dict] or None
        Rule table. Defaults to RULE_TABLE.

    Returns
    -------
    tuple[dict[str, str], list[dict]]
        (new_state, fired_rules) -- the updated discrete state and the list
        of rules that fired (passed the probabilistic gate).
    """
    applicable = get_applicable_rules(discrete_state, context, rules)

    # Point of no return is an absorbing state -- deterministic freeze
    for rule in applicable:
        if rule["name"] == "point_of_no_return":
            return dict(discrete_state), applicable

    # Probabilistic firing: each rule fires with probability = confidence
    surviving_rules = []
    for rule in applicable:
        if rng.random() < rule["confidence"]:
            surviving_rules.append(rule)

    if not surviving_rules:
        return dict(discrete_state), []

    # Collect proposals per variable: {var_name: [(direction, confidence, rule)]}
    proposals: dict[str, list[tuple[str, float, dict]]] = {}
    for rule in surviving_rules:
        for var_name, direction in rule["outputs"].items():
            if var_name not in proposals:
                proposals[var_name] = []
            proposals[var_name].append((direction, rule["confidence"], rule))

    # Resolve conflicts and apply updates
    new_state = dict(discrete_state)
    for var_name, candidates in proposals.items():
        if len(candidates) == 1:
            # Single proposal -- apply directly
            direction = candidates[0][0]
        else:
            # Multiple proposals -- sample proportional to confidence
            confidences = np.array([c[1] for c in candidates], dtype=np.float64)
            probs = confidences / confidences.sum()
            idx = rng.choice(len(candidates), p=probs)
            direction = candidates[idx][0]

        new_state[var_name] = _apply_direction(
            new_state[var_name], direction, var_name
        )

    return new_state, surviving_rules


# ── Monte Carlo ensemble runner ──────────────────────────────────────────


def run_single_cell_stochastic(
    patient: dict | None = None,
    intervention: dict | None = None,
    sim_years: float = 30.0,
    dt: float = 0.25,
    n_trials: int = 100,
    seed: int = 42,
    rules: list[dict] | None = None,
) -> dict:
    """Run a Monte Carlo ensemble of single-cell CA simulations.

    Each trial uses an independent RNG stream seeded with (seed + trial_index),
    producing a distribution of trajectories from identical initial conditions.
    The stochastic variation comes from apply_rules_stochastic.

    Parameters
    ----------
    patient : dict or None
        Patient parameters (merged with DEFAULT_PATIENT).
    intervention : dict or None
        Intervention parameters (merged with DEFAULT_INTERVENTION).
    sim_years : float
        Simulation horizon in years (default 30.0).
    dt : float
        Timestep in years (default 0.25 = quarterly).
    n_trials : int
        Number of Monte Carlo trials (default 100).
    seed : int
        Base random seed. Trial i uses seed + i.
    rules : list[dict] or None
        Rule table. Defaults to RULE_TABLE.

    Returns
    -------
    dict
        {
            "n_trials": int,
            "final_states": list[dict],
            "trajectories": list[list[dict]],
            "rule_logs": list[list[list[str]]],
            "patient": dict,
            "intervention": dict,
            "seed": int,
            "sim_years": float,
            "dt": float,
        }
    """
    from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT
    from simulator import initial_state
    from ca_schema import discretize_state
    from ca_simulator import _build_context

    pat = {**DEFAULT_PATIENT, **(patient or {})}
    intv = {**DEFAULT_INTERVENTION, **(intervention or {})}
    n_steps = int(sim_years / dt)

    # Initialize from ODE initial state (same for all trials)
    continuous_init = initial_state(pat)
    init_state = discretize_state(continuous_init)

    final_states = []
    trajectories = []
    rule_logs = []

    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial)

        state = dict(init_state)
        trajectory = [dict(state)]
        trial_rule_log = []
        prev_state = None

        for step in range(n_steps):
            ctx = _build_context(step, pat, intv, prev_state, state)
            new_state, fired = apply_rules_stochastic(state, ctx, rng, rules)
            trial_rule_log.append([r["name"] for r in fired])
            prev_state = state
            state = new_state
            trajectory.append(dict(state))

        final_states.append(dict(state))
        trajectories.append(trajectory)
        rule_logs.append(trial_rule_log)

    return {
        "n_trials": n_trials,
        "final_states": final_states,
        "trajectories": trajectories,
        "rule_logs": rule_logs,
        "patient": pat,
        "intervention": intv,
        "seed": seed,
        "sim_years": sim_years,
        "dt": dt,
    }


# ── Ensemble analytics ───────────────────────────────────────────────────


def compute_ensemble_analytics(ensemble_result: dict) -> dict:
    """Compute aggregate statistics over a stochastic CA ensemble.

    Takes the output of run_single_cell_stochastic and computes attractor
    distributions, cliff crossing probability, per-variable terminal bin
    distributions, and time-to-crisis statistics.

    Parameters
    ----------
    ensemble_result : dict
        Output from run_single_cell_stochastic().

    Returns
    -------
    dict
        {
            "attractor_probabilities": dict[str, float],
            "cliff_crossing_probability": float,
            "variable_distributions": dict[str, dict[str, float]],
            "time_to_crisis": dict,
        }
    """
    n_trials = ensemble_result["n_trials"]
    final_states = ensemble_result["final_states"]
    trajectories = ensemble_result["trajectories"]

    # ── Attractor probabilities ──
    attractor_counts = Counter(_classify_attractor(s) for s in final_states)
    attractor_probs = {k: round(v / n_trials, 3) for k, v in attractor_counts.items()}

    # Ensure all 4 attractors have entries
    for att in ("healthy_aging", "slow_decline", "cliff_approaching", "point_of_no_return"):
        attractor_probs.setdefault(att, 0.0)

    # ── Cliff crossing probability ──
    # Fraction of trials where N_deletion ever reaches past_cliff
    cliff_crossings = 0
    for traj in trajectories:
        for state in traj:
            if state.get("N_deletion") == "past_cliff":
                cliff_crossings += 1
                break
    cliff_crossing_prob = round(cliff_crossings / n_trials, 3)

    # ── Variable distributions at final step ──
    variable_distributions: dict[str, dict[str, float]] = {}
    for var_name in CA_VAR_ORDER:
        bin_labels = BIN_SCHEMA[var_name]["labels"]
        counts: dict[str, int] = {label: 0 for label in bin_labels}
        for state in final_states:
            label = state.get(var_name, bin_labels[0])
            if label in counts:
                counts[label] += 1
            else:
                counts[label] = 1
        variable_distributions[var_name] = {
            label: round(count / n_trials, 3) for label, count in counts.items()
        }

    # ── Time-to-crisis distribution ──
    # Step where ATP first hits "collapsed" or "crisis"
    crisis_times = []
    for traj in trajectories:
        for i, state in enumerate(traj):
            if state.get("ATP") in ("collapsed", "crisis"):
                crisis_times.append(i)
                break

    if crisis_times:
        crisis_arr = np.array(crisis_times, dtype=np.float64)
        crisis_stats = {
            "mean_step": round(float(crisis_arr.mean()), 1),
            "std_step": round(float(crisis_arr.std()), 1),
            "min_step": int(crisis_arr.min()),
            "max_step": int(crisis_arr.max()),
            "fraction_reaching_crisis": round(len(crisis_times) / n_trials, 3),
        }
    else:
        crisis_stats = {
            "mean_step": None,
            "std_step": None,
            "min_step": None,
            "max_step": None,
            "fraction_reaching_crisis": 0.0,
        }

    return {
        "attractor_probabilities": attractor_probs,
        "cliff_crossing_probability": cliff_crossing_prob,
        "variable_distributions": variable_distributions,
        "time_to_crisis": crisis_stats,
    }
