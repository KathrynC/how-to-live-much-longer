"""Analytics for the mitochondrial semantic cellular automaton.

Computes five sections of metrics from a CA simulation result:

1. **Rule stats**: Which rules fired most often, mean firings per step.
2. **Cascade stats**: Multi-step chain reactions where >= 3 rules fire
   on consecutive steps.
3. **Attractor stats**: Terminal attractor basin classification (healthy_aging,
   slow_decline, cliff_approaching, point_of_no_return) with transition tracking.
4. **Fidelity stats**: How well the CA trajectory agrees with the ODE
   trajectory at the bin level (requires an ODE result for comparison).
5. **Epoch diagnostic**: Pre/post age-65 transition comparison (analog of
   LEMURS spring break diagnostic).

Also provides tissue-level analytics for the 4-tissue grid.

Mirrors ~/lemurs-simulator/ca_analytics.py architecture.
"""
from __future__ import annotations

from collections import Counter

from ca_schema import (
    BIN_SCHEMA, CA_VAR_ORDER, CA_N_VARS,
    bin_index, bin_count, discretize_state,
)


# ── Variables where higher bin index means worse outcome ─────────────────
# N_deletion: minimal(0) < growing(1) < approaching_cliff(2) < past_cliff(3)
# ROS: basal(0) < elevated(1) < pathological(2)
# Senescent_fraction: minimal(0) < emerging(1) < severe(2)
# N_point: low(0) < moderate(1) < high(2)
_HIGHER_IS_WORSE = {"N_deletion", "ROS", "Senescent_fraction", "N_point"}

# Variables where lower bin index means worse outcome:
# N_healthy: depleted(0) < reduced(1) < adequate(2)
# ATP: collapsed(0) < crisis(1) < compromised(2) < healthy(3)
# NAD: depleted(0) < declining(1) < robust(2)
# Membrane_potential: collapsed(0) < impaired(1) < intact(2)
_LOWER_IS_WORSE = {"N_healthy", "ATP", "NAD", "Membrane_potential"}


# ── 1. Rule stats ────────────────────────────────────────────────────────


def _rule_stats(rule_log: list[list[str]]) -> dict:
    """Compute rule firing statistics from the rule log.

    Parameters
    ----------
    rule_log : list[list[str]]
        For each CA step, the list of rule names that fired.

    Returns
    -------
    dict
        total_firings, unique_rules, top_10, mean_rules_per_step.
    """
    all_firings = []
    for step_rules in rule_log:
        all_firings.extend(step_rules)

    counter = Counter(all_firings)
    total = len(all_firings)
    unique = len(counter)
    top_10 = counter.most_common(10)
    mean_per_step = total / max(len(rule_log), 1)

    return {
        "total_firings": total,
        "unique_rules": unique,
        "top_10": [{"name": n, "count": c} for n, c in top_10],
        "mean_rules_per_step": round(mean_per_step, 2),
    }


# ── 2. Cascade stats ────────────────────────────────────────────────────


def _cascade_stats(
    trajectory: list[dict[str, str]],
    rule_log: list[list[str]],
) -> dict:
    """Detect multi-step chain reactions (cascades).

    A cascade is a run of consecutive steps where >= 3 rules fire,
    indicating a chain reaction where one rule's state change triggers
    additional rules in the next step.

    Parameters
    ----------
    trajectory : list[dict[str, str]]
        Discrete state at each timestep (length n_steps + 1).
    rule_log : list[list[str]]
        Rules fired per step (length n_steps).

    Returns
    -------
    dict
        n_cascades, max_cascade_length, cascades (top 5).
    """
    cascades = []
    current_cascade = []

    for i, step_rules in enumerate(rule_log):
        if len(step_rules) >= 3:
            current_cascade.append(i)
        else:
            if len(current_cascade) >= 2:
                cascades.append({
                    "start_step": current_cascade[0],
                    "length": len(current_cascade),
                })
            current_cascade = []

    # Handle cascade at end of log
    if len(current_cascade) >= 2:
        cascades.append({
            "start_step": current_cascade[0],
            "length": len(current_cascade),
        })

    max_length = max((c["length"] for c in cascades), default=0)

    return {
        "n_cascades": len(cascades),
        "max_cascade_length": max_length,
        "cascades": cascades[:5],  # top 5 only
    }


# ── 3. Attractor classification and stats ────────────────────────────────


def _classify_attractor(state: dict[str, str]) -> str:
    """Classify a discrete state into one of 4 attractor basins.

    Attractor hierarchy (worst to best):
        point_of_no_return  -- all 4 critical conditions met
        cliff_approaching   -- deletion het near or past cliff
        slow_decline        -- ATP compromised/crisis or deletions growing
        healthy_aging       -- none of the above

    Parameters
    ----------
    state : dict[str, str]
        Discrete state {var_name: bin_label}.

    Returns
    -------
    str
        One of "point_of_no_return", "cliff_approaching", "slow_decline",
        "healthy_aging".
    """
    n_del = state.get("N_deletion", "minimal")
    atp = state.get("ATP", "healthy")
    ros = state.get("ROS", "basal")
    sen = state.get("Senescent_fraction", "minimal")

    # Point of no return: all 4 conditions met
    if (n_del == "past_cliff" and atp == "collapsed"
            and ros == "pathological" and sen == "severe"):
        return "point_of_no_return"

    # Cliff approaching: near or past cliff (but not full collapse)
    if n_del in ("approaching_cliff", "past_cliff"):
        return "cliff_approaching"

    # Slow decline: ATP compromised/crisis or deletions growing
    if atp in ("compromised", "crisis") or n_del == "growing":
        return "slow_decline"

    # Healthy aging: none of the above
    return "healthy_aging"


def _attractor_stats(trajectory: list[dict[str, str]]) -> dict:
    """Compute attractor statistics from the full trajectory.

    Classifies each timestep, tracks transitions between attractor basins,
    and reports time spent in each basin.

    Parameters
    ----------
    trajectory : list[dict[str, str]]
        Discrete state at each timestep (length n_steps + 1).

    Returns
    -------
    dict
        final_attractor, attractor_transitions, time_in_attractor,
        first_reached.
    """
    # Classify each step
    attractor_sequence = [_classify_attractor(s) for s in trajectory]
    final = attractor_sequence[-1]

    # Count transitions between attractor basins
    transitions = 0
    for i in range(1, len(attractor_sequence)):
        if attractor_sequence[i] != attractor_sequence[i - 1]:
            transitions += 1

    # Time in each attractor (as fraction of total)
    counts = Counter(attractor_sequence)
    total = len(attractor_sequence)
    fractions = {k: round(v / total, 3) for k, v in counts.items()}

    # First step reaching each attractor
    first_step = {}
    for i, att in enumerate(attractor_sequence):
        if att not in first_step:
            first_step[att] = i

    return {
        "final_attractor": final,
        "attractor_transitions": transitions,
        "time_in_attractor": fractions,
        "first_reached": first_step,
    }


# ── 4. Fidelity stats (CA vs ODE) ───────────────────────────────────────


def _fidelity_stats(
    trajectory: list[dict[str, str]],
    ode_result: dict,
    patient: dict | None = None,
) -> dict | None:
    """Compute CA vs ODE bin agreement.

    The ODE runs at dt=0.01yr (3000 steps for 30yr). The CA runs at
    dt=0.25yr (120 steps). We subsample the ODE at CA timesteps.

    Parameters
    ----------
    trajectory : list[dict[str, str]]
        CA discrete states (length n_steps + 1).
    ode_result : dict
        Output from simulator.simulate() with "states" (array) and
        "time" (array).
    patient : dict or None
        Patient parameters (unused currently, reserved for future use).

    Returns
    -------
    dict or None
        per_variable_agreement, overall_agreement. Returns None if
        ode_result is None.
    """
    if ode_result is None:
        return None

    ode_states = ode_result["states"]  # shape (n_ode_steps+1, 8)
    ode_dt = 0.01  # ODE timestep from constants
    ca_dt = 0.25   # CA timestep

    ca_n_steps = len(trajectory) - 1  # exclude initial state

    # Per-variable agreement
    var_agreement = {}
    total_agree = 0
    total_compared = 0

    for var_name in CA_VAR_ORDER:
        agree = 0
        compared = 0

        for ca_step in range(ca_n_steps + 1):
            # Find closest ODE timestep
            ca_time = ca_step * ca_dt
            ode_idx = min(int(round(ca_time / ode_dt)), len(ode_states) - 1)

            # Discretize the ODE state at this time
            ode_discrete = discretize_state(ode_states[ode_idx])
            ca_bin = trajectory[ca_step].get(var_name)
            ode_bin = ode_discrete.get(var_name)

            if ca_bin == ode_bin:
                agree += 1
            compared += 1

        var_agreement[var_name] = round(agree / max(compared, 1), 3)
        total_agree += agree
        total_compared += compared

    return {
        "per_variable_agreement": var_agreement,
        "overall_agreement": round(total_agree / max(total_compared, 1), 3),
    }


# ── 5. Epoch diagnostic (age-65 transition) ─────────────────────────────


def _epoch_diagnostic(
    trajectory: list[dict[str, str]],
    patient: dict,
) -> dict:
    """Compare pre/post age-65 transition states.

    Analog of the LEMURS spring break diagnostic. Compares the modal bin
    in a 1-year window (4 steps at dt=0.25) before and after age 65.

    Parameters
    ----------
    trajectory : list[dict[str, str]]
        CA discrete states (length n_steps + 1).
    patient : dict
        Patient parameters (needs baseline_age).

    Returns
    -------
    dict
        transition_step, transition_age, n_variables_changed, changes.
    """
    baseline_age = patient.get("baseline_age", 70.0)
    dt = 0.25

    # Find the CA step where age crosses 65
    transition_step = None
    for i in range(len(trajectory)):
        age = baseline_age + i * dt
        if age >= 65.0:
            transition_step = i
            break

    if (transition_step is None
            or transition_step < 4
            or transition_step >= len(trajectory) - 4):
        return {
            "transition_step": None,
            "message": "Age 65 not within simulation window",
        }

    # Compare 4 steps before vs 4 steps after (1 year window each side)
    pre_states = trajectory[max(0, transition_step - 4):transition_step]
    post_states = trajectory[transition_step:min(len(trajectory), transition_step + 4)]

    changes = {}
    for var_name in CA_VAR_ORDER:
        pre_bins = [s.get(var_name) for s in pre_states]
        post_bins = [s.get(var_name) for s in post_states]

        # Most common bin before/after
        pre_mode = Counter(pre_bins).most_common(1)[0][0] if pre_bins else None
        post_mode = Counter(post_bins).most_common(1)[0][0] if post_bins else None

        if pre_mode != post_mode:
            pre_idx = bin_index(var_name, pre_mode) if pre_mode else 0
            post_idx = bin_index(var_name, post_mode) if post_mode else 0

            # Determine direction based on variable polarity
            if var_name in _HIGHER_IS_WORSE:
                # Higher index = worse (more damage/ROS/senescence/mutations)
                direction = "worsened" if post_idx > pre_idx else "improved"
            else:
                # Lower index = worse (less ATP/NAD/membrane/healthy copies)
                direction = "worsened" if post_idx < pre_idx else "improved"

            changes[var_name] = {
                "pre": pre_mode,
                "post": post_mode,
                "direction": direction,
            }

    return {
        "transition_step": transition_step,
        "transition_age": baseline_age + transition_step * dt,
        "n_variables_changed": len(changes),
        "changes": changes,
    }


# ── Main analytics function ──────────────────────────────────────────────


def compute_ca_analytics(
    ca_result: dict,
    ode_result: dict | None = None,
) -> dict:
    """Compute all CA analytics from a single-cell result.

    Parameters
    ----------
    ca_result : dict
        Output from ca_simulator.run_single_cell() with "trajectory",
        "rule_log", "patient", etc.
    ode_result : dict or None
        Output from simulator.simulate() for fidelity comparison.

    Returns
    -------
    dict
        rule_stats, cascade_stats, attractor_stats, fidelity_stats,
        epoch_diagnostic.
    """
    trajectory = ca_result["trajectory"]
    rule_log = ca_result["rule_log"]
    patient = ca_result.get("patient", {})

    return {
        "rule_stats": _rule_stats(rule_log),
        "cascade_stats": _cascade_stats(trajectory, rule_log),
        "attractor_stats": _attractor_stats(trajectory),
        "fidelity_stats": _fidelity_stats(trajectory, ode_result, patient),
        "epoch_diagnostic": _epoch_diagnostic(trajectory, patient),
    }


# ── Tissue analytics ─────────────────────────────────────────────────────

# Attractor severity ordering (worst first) for tissue vulnerability ranking
_ATTRACTOR_SEVERITY = [
    "point_of_no_return",
    "cliff_approaching",
    "slow_decline",
    "healthy_aging",
]


def compute_tissue_analytics(tissue_result: dict) -> dict:
    """Compute per-tissue analytics from a tissue grid result.

    Classifies each tissue's final state into an attractor basin and
    computes tissue divergence metrics.

    Parameters
    ----------
    tissue_result : dict
        Output from ca_simulator.run_tissue_grid() with "final_tissues".

    Returns
    -------
    dict
        tissue_attractors, n_distinct_attractors, attractor_distribution,
        most_vulnerable, most_resilient.
    """
    final_tissues = tissue_result["final_tissues"]

    tissue_attractors = {}
    for tissue, state in final_tissues.items():
        tissue_attractors[tissue] = _classify_attractor(state)

    # Tissue divergence: how different are the 4 tissues?
    attractor_counts = Counter(tissue_attractors.values())
    n_distinct = len(attractor_counts)

    # Most vulnerable: tissue with worst (lowest index) attractor
    most_vulnerable = min(
        tissue_attractors,
        key=lambda t: _ATTRACTOR_SEVERITY.index(tissue_attractors[t]),
    )
    # Most resilient: tissue with best (highest index) attractor
    most_resilient = max(
        tissue_attractors,
        key=lambda t: _ATTRACTOR_SEVERITY.index(tissue_attractors[t]),
    )

    return {
        "tissue_attractors": tissue_attractors,
        "n_distinct_attractors": n_distinct,
        "attractor_distribution": dict(attractor_counts),
        "most_vulnerable": most_vulnerable,
        "most_resilient": most_resilient,
    }
