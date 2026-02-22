"""Cellular automaton simulator for the mitochondrial semantic CA.

Provides single-cell stepping and a 4-tissue population grid (brain,
muscle, cardiac, skin) with inter-tissue coupling channels. Each cell
is a discrete state dict updated by the rule table from ca_rules.py.

Mirrors ~/lemurs-simulator/ca_simulator.py architecture.
"""
from __future__ import annotations

from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT, TISSUE_PROFILES
from simulator import initial_state
from ca_schema import (
    discretize_state, bin_index, bin_count, BIN_SCHEMA,
    CA_VAR_ORDER, CA_N_VARS,
)
from ca_rules import apply_rules, RULE_TABLE

# ── Constants ────────────────────────────────────────────────────────────

CA_SIM_YEARS = 30.0
CA_DT = 0.25          # quarterly timesteps
CA_N_STEPS = int(CA_SIM_YEARS / CA_DT)  # = 120

TISSUE_TYPES = ["brain", "muscle", "cardiac", "skin"]

# Skin is not in TISSUE_PROFILES, so map it to "default".
_TISSUE_PROFILE_MAP = {
    "brain": "brain",
    "muscle": "muscle",
    "cardiac": "cardiac",
    "skin": "default",
}


# ── Context builder ──────────────────────────────────────────────────────

def _build_context(step, patient, intervention, prev_state, curr_state):
    """Build context dict for rule evaluation at a given CA step.

    Parameters
    ----------
    step : int
        Current simulation step (0-indexed).
    patient : dict
        Patient parameter dict.
    intervention : dict
        Intervention parameter dict.
    prev_state : dict or None
        Previous discrete state (None at step 0).
    curr_state : dict or None
        Current discrete state.

    Returns
    -------
    dict
        Context dict with age, age_epoch, intervention levels, etc.
    """
    age = patient.get("baseline_age", 70.0) + step * CA_DT

    # Age epoch
    if age < 50.0:
        age_epoch = "young"
    elif age < 70.0:
        age_epoch = "transition"
    else:
        age_epoch = "old"

    near_transition = abs(age - 65.0) < 5.0

    # Map intervention floats to level strings
    # >0.5 = "high", >0.2 = "moderate" (for exercise) / "low" (for others), else "none"
    def _level(val, use_moderate=False):
        if val > 0.5:
            return "high"
        elif val > 0.2:
            return "moderate" if use_moderate else "low"
        else:
            return "none"

    ctx = {
        "age": age,
        "age_epoch": age_epoch,
        "near_transition": near_transition,
        "rapamycin": _level(intervention.get("rapamycin_dose", 0.0)),
        "nad_supplement": _level(intervention.get("nad_supplement", 0.0)),
        "senolytic": _level(intervention.get("senolytic_dose", 0.0)),
        "exercise": _level(intervention.get("exercise_level", 0.0), use_moderate=True),
        "yamanaka": _level(intervention.get("yamanaka_intensity", 0.0)),
        "transplant": _level(intervention.get("transplant_rate", 0.0)),
    }

    # Cliff proximity from current state
    if curr_state:
        del_idx = bin_index("N_deletion", curr_state.get("N_deletion", "minimal"))
        del_max = bin_count("N_deletion") - 1
        ctx["cliff_proximity"] = del_idx / del_max if del_max > 0 else 0.0

    return ctx


# ── Single-cell stepper ──────────────────────────────────────────────────

def step_cell(state, context, rules=None):
    """Apply one CA step to a single cell.

    Delegates directly to apply_rules from ca_rules.py.

    Parameters
    ----------
    state : dict[str, str]
        Current discrete state.
    context : dict
        Context dict from _build_context.
    rules : list[dict] or None
        Rule table (defaults to RULE_TABLE).

    Returns
    -------
    tuple[dict[str, str], list[dict]]
        (new_state, fired_rules)
    """
    return apply_rules(state, context, rules)


# ── Single-cell simulation ───────────────────────────────────────────────

def run_single_cell(patient=None, intervention=None, sim_years=30.0, dt=0.25):
    """Run a single-cell CA simulation over the specified time horizon.

    Parameters
    ----------
    patient : dict or None
        Patient parameters (merged with DEFAULT_PATIENT).
    intervention : dict or None
        Intervention parameters (merged with DEFAULT_INTERVENTION).
    sim_years : float
        Simulation horizon in years.
    dt : float
        Timestep in years (default 0.25 = quarterly).

    Returns
    -------
    dict
        Result dict with trajectory, rule_log, final_state, etc.
    """
    pat = {**DEFAULT_PATIENT, **(patient or {})}
    intv = {**DEFAULT_INTERVENTION, **(intervention or {})}

    n_steps = int(sim_years / dt)

    # Initialize from ODE initial state
    continuous_init = initial_state(pat)
    state = discretize_state(continuous_init)

    trajectory = [dict(state)]
    rule_log = []
    prev_state = None

    for step in range(n_steps):
        ctx = _build_context(step, pat, intv, prev_state, state)
        new_state, fired = step_cell(state, ctx)
        rule_log.append([r["name"] for r in fired])
        prev_state = state
        state = new_state
        trajectory.append(dict(state))

    return {
        "trajectory": trajectory,
        "rule_log": rule_log,
        "final_state": dict(state),
        "initial_continuous": continuous_init,
        "patient": pat,
        "intervention": intv,
        "sim_years": sim_years,
        "dt": dt,
        "n_steps": n_steps,
    }


# ── Tissue grid simulation ──────────────────────────────────────────────

def run_tissue_grid(patient=None, intervention=None, sim_years=30.0, dt=0.25,
                    tissue_coupling=0.5):
    """Run a 4-tissue CA simulation with inter-tissue coupling.

    Each tissue type gets modified patient parameters from TISSUE_PROFILES.
    Inter-tissue coupling has 3 channels:
      1. Systemic SASP inflammation (severe senescence -> ROS in all tissues)
      2. Circulating NAD equilibrium (NAD trends toward majority)
      3. Senolytic clearance is systemic (handled by rules)

    Parameters
    ----------
    patient : dict or None
        Base patient parameters (merged with DEFAULT_PATIENT).
    intervention : dict or None
        Intervention parameters (merged with DEFAULT_INTERVENTION).
    sim_years : float
        Simulation horizon in years.
    dt : float
        Timestep in years.
    tissue_coupling : float
        Coupling strength between 0.0 (independent) and 1.0 (strong).

    Returns
    -------
    dict
        Result dict with tissue_states, final_tissues, population_summary, etc.
    """
    pat = {**DEFAULT_PATIENT, **(patient or {})}
    intv = {**DEFAULT_INTERVENTION, **(intervention or {})}
    n_steps = int(sim_years / dt)

    # Initialize 4 tissue cells with tissue-specific patient modifications
    tissue_states = {}
    tissue_patients = {}
    tissue_trajectories = {}
    tissue_rule_logs = {}

    for tissue in TISSUE_TYPES:
        profile_key = _TISSUE_PROFILE_MAP.get(tissue, "default")
        profile = TISSUE_PROFILES.get(profile_key, TISSUE_PROFILES["default"])
        tissue_pat = dict(pat)
        tissue_pat["metabolic_demand"] = (
            pat.get("metabolic_demand", 1.0) * profile.get("metabolic_demand", 1.0)
        )
        tissue_patients[tissue] = tissue_pat

        continuous_init = initial_state(tissue_pat)
        tissue_states[tissue] = discretize_state(continuous_init)
        tissue_trajectories[tissue] = [dict(tissue_states[tissue])]
        tissue_rule_logs[tissue] = []

    prev_states = {t: None for t in TISSUE_TYPES}

    for step in range(n_steps):
        new_states = {}

        for tissue in TISSUE_TYPES:
            ctx = _build_context(
                step, tissue_patients[tissue], intv,
                prev_states[tissue], tissue_states[tissue],
            )
            ctx["tissue_type"] = tissue
            new_state, fired = step_cell(tissue_states[tissue], ctx)
            new_states[tissue] = new_state
            tissue_rule_logs[tissue].append([r["name"] for r in fired])

        # Apply inter-tissue coupling (3 channels)
        if tissue_coupling > 0:
            # Channel 1: Systemic SASP inflammation
            # If ANY tissue has Senescent_fraction=severe, ROS +1 in ALL tissues
            any_severe_sen = any(
                new_states[t].get("Senescent_fraction") == "severe"
                for t in TISSUE_TYPES
            )
            if any_severe_sen:
                for t in TISSUE_TYPES:
                    ros_labels = BIN_SCHEMA["ROS"]["labels"]
                    curr_ros = new_states[t].get("ROS", "basal")
                    curr_idx = ros_labels.index(curr_ros)
                    new_idx = min(curr_idx + 1, len(ros_labels) - 1)
                    new_states[t]["ROS"] = ros_labels[new_idx]

            # Channel 2: Circulating NAD equilibrium
            # NAD level trends toward the most common NAD bin
            nad_labels = BIN_SCHEMA["NAD"]["labels"]
            nad_counts = {}
            for t in TISSUE_TYPES:
                nad_bin = new_states[t].get("NAD", "declining")
                nad_counts[nad_bin] = nad_counts.get(nad_bin, 0) + 1
            majority_nad = max(nad_counts, key=nad_counts.get)
            majority_idx = nad_labels.index(majority_nad)

            for t in TISSUE_TYPES:
                curr_nad = new_states[t].get("NAD", "declining")
                curr_idx = nad_labels.index(curr_nad)
                if curr_idx < majority_idx:
                    # Trend up toward majority (coupling strength modulates)
                    if tissue_coupling > 0.3:
                        new_states[t]["NAD"] = nad_labels[
                            min(curr_idx + 1, len(nad_labels) - 1)
                        ]
                elif curr_idx > majority_idx:
                    if tissue_coupling > 0.3:
                        new_states[t]["NAD"] = nad_labels[
                            max(curr_idx - 1, 0)
                        ]

            # Channel 3: Senolytic clearance is systemic
            # (Already handled by rules -- senolytics work on all tissues equally)

        prev_states = tissue_states.copy()
        tissue_states = new_states
        for t in TISSUE_TYPES:
            tissue_trajectories[t].append(dict(tissue_states[t]))

    # Population summary
    summary = _compute_tissue_summary(tissue_states)

    return {
        "tissue_states": tissue_trajectories,
        "final_tissues": {t: dict(tissue_states[t]) for t in TISSUE_TYPES},
        "tissue_rule_logs": tissue_rule_logs,
        "population_summary": summary,
        "patient": pat,
        "intervention": intv,
        "sim_years": sim_years,
        "dt": dt,
        "tissue_coupling": tissue_coupling,
    }


# ── Summary helper ───────────────────────────────────────────────────────

def _compute_tissue_summary(tissue_states):
    """Compute summary over 4 tissue final states.

    Parameters
    ----------
    tissue_states : dict[str, dict[str, str]]
        Mapping of tissue name to final discrete state.

    Returns
    -------
    dict
        Summary with total_tissues, tissue_attractors, attractor_distribution.
    """
    total = len(tissue_states)

    # Classify each tissue into an attractor basin
    attractors = {}
    for tissue, state in tissue_states.items():
        n_del = state.get("N_deletion", "minimal")
        atp = state.get("ATP", "healthy")
        ros = state.get("ROS", "basal")
        sen = state.get("Senescent_fraction", "minimal")

        if (n_del == "past_cliff" and atp == "collapsed"
                and ros == "pathological" and sen == "severe"):
            attractors[tissue] = "point_of_no_return"
        elif n_del in ("approaching_cliff", "past_cliff"):
            attractors[tissue] = "cliff_approaching"
        elif atp in ("compromised", "crisis") or n_del == "growing":
            attractors[tissue] = "slow_decline"
        else:
            attractors[tissue] = "healthy_aging"

    attractor_counts = {}
    for att in attractors.values():
        attractor_counts[att] = attractor_counts.get(att, 0) + 1

    return {
        "total_tissues": total,
        "tissue_attractors": attractors,
        "attractor_distribution": attractor_counts,
    }
