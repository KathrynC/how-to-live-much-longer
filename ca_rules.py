"""Tiered rule table for the mitochondrial semantic cellular automaton.

Rules are organized by ODE coupling tier, mirroring the coupled dynamics in
derivatives(). Each rule specifies input bin conditions, output bin updates,
a confidence weight, and the source citation. Cross-tier compound rules capture
emergent multi-variable dynamics (point of no return, vicious cycle lock).

The rule format follows the LEMURS project's rulebook pattern: JSON-serializable
dicts that can be inspected, edited, and versioned independently of code.

Input matching supports three modes:
  - "label"   -- exact match
  - "label+"  -- current bin index >= label's index (this level or worse)
  - "label-"  -- current bin index <= label's index (this level or better)

Context matching supports:
  - age_epoch: string or list of strings
  - intervention levels with optional "+" suffix for threshold comparison
"""
from __future__ import annotations

import json
from copy import deepcopy

from ca_schema import BIN_SCHEMA, _classify, bin_index


# ── Rule table ────────────────────────────────────────────────────────────
# Each rule is a dict with:
#   tier: int           -- which ODE coupling tier (1-6, 0 for cross-tier)
#   name: str           -- human-readable rule name
#   inputs: dict        -- {var_name: bin_label} conditions (all must match)
#   context: dict       -- {context_key: value} conditions on age/interventions
#   outputs: dict       -- {var_name: bin_label_or_direction} updates
#   confidence: float   -- rule strength [0, 1]
#   citation: str       -- source paper/chapter reference

RULE_TABLE: list[dict] = [
    # =================================================================
    # TIER 1: Energy-Damage Coupling (5 rules)
    # =================================================================
    {
        "tier": 1,
        "name": "deletion_expansion_young",
        "inputs": {"N_deletion": "growing+"},
        "context": {"age_epoch": "young"},
        "outputs": {"N_deletion": "+1", "N_healthy": "-1"},
        "confidence": 0.75,
        "citation": "Cramer Appendix 2: deletion doubling 11.8yr young",
    },
    {
        "tier": 1,
        "name": "deletion_expansion_old",
        "inputs": {"N_deletion": "growing+"},
        "context": {"age_epoch": "old"},
        "outputs": {"N_deletion": "+1", "N_healthy": "-1"},
        "confidence": 0.90,
        "citation": "Cramer Appendix 2: deletion doubling 3.06yr old",
    },
    {
        "tier": 1,
        "name": "cliff_atp_collapse",
        "inputs": {"N_deletion": "past_cliff"},
        "context": {},
        "outputs": {"ATP": "-2"},
        "confidence": 0.95,
        "citation": "HETEROPLASMY_CLIFF=0.50, sigmoid collapse",
    },
    {
        "tier": 1,
        "name": "cliff_approaching_warning",
        "inputs": {"N_deletion": "approaching_cliff", "ATP": "compromised+"},
        "context": {},
        "outputs": {"ATP": "-1"},
        "confidence": 0.80,
        "citation": "Pre-cliff ATP degradation, Cramer Ch VIII.A",
    },
    {
        "tier": 1,
        "name": "deletion_acceleration_energy_crisis",
        "inputs": {"N_deletion": "growing+", "ATP": "crisis-"},
        "context": {"rapamycin": "none"},
        "outputs": {"N_deletion": "+1", "N_healthy": "-1"},
        "confidence": 0.80,
        "citation": "C10: ATP collapse + poor mitophagy accelerates deletion expansion",
    },

    # =================================================================
    # TIER 2: ROS-Damage Vicious Cycle (5 rules)
    # =================================================================
    {
        "tier": 2,
        "name": "ros_from_deletions",
        "inputs": {"N_deletion": "growing+"},
        "context": {},
        "outputs": {"ROS": "+1"},
        "confidence": 0.85,
        "citation": "Cramer Ch II.H: damaged mitos produce excess ROS",
    },
    {
        "tier": 2,
        "name": "ros_from_points",
        "inputs": {"N_point": "moderate+"},
        "context": {},
        "outputs": {"ROS": "+1"},
        "confidence": 0.70,
        "citation": "C11: point mutations contribute ~33% ROS",
    },
    {
        "tier": 2,
        "name": "ros_drives_points",
        "inputs": {"ROS": "elevated+"},
        "context": {},
        "outputs": {"N_point": "+1"},
        "confidence": 0.75,
        "citation": "C11: ROS causes point mutations (not deletions)",
    },
    {
        "tier": 2,
        "name": "ros_membrane_damage",
        "inputs": {"ROS": "pathological"},
        "context": {},
        "outputs": {"Membrane_potential": "-1"},
        "confidence": 0.85,
        "citation": "Cramer Ch IV: ROS damages membrane potential",
    },
    {
        "tier": 2,
        "name": "point_mutation_pol_gamma_errors",
        "inputs": {"N_healthy": "adequate", "age_epoch": ["transition", "old"]},
        "context": {},
        "outputs": {"N_point": "+1"},
        "confidence": 0.60,
        "citation": "C11: Pol γ copying errors cause point mutations",
    },

    # =================================================================
    # TIER 3: Mitophagy Quality Control (4 rules)
    # =================================================================
    {
        "tier": 3,
        "name": "mitophagy_clears_deletions",
        "inputs": {"Membrane_potential": "impaired-", "ATP": "compromised+"},
        "context": {"rapamycin": "high"},
        "outputs": {"N_deletion": "-1"},
        "confidence": 0.80,
        "citation": "Cramer Ch VI.B: PINK1/Parkin + rapamycin",
    },
    {
        "tier": 3,
        "name": "mitophagy_atp_gated",
        "inputs": {"N_deletion": "growing+", "ATP": "collapsed"},
        "context": {},
        "outputs": {},
        "confidence": 0.90,
        "citation": "Fix C4: ATP collapse halts mitophagy",
    },
    {
        "tier": 3,
        "name": "mitophagy_weak_on_points",
        "inputs": {"N_point": "moderate+"},
        "context": {"rapamycin": "high"},
        "outputs": {"N_point": "-1"},
        "confidence": 0.55,
        "citation": "C11: mitophagy less effective on point mutations",
    },
    {
        "tier": 3,
        "name": "rapamycin_membrane_benefit",
        "inputs": {},
        "context": {"rapamycin": "high"},
        "outputs": {"Membrane_potential": "+1"},
        "confidence": 0.75,
        "citation": "Rapamycin enhances mitochondrial quality",
    },

    # =================================================================
    # TIER 4: Senescence & SASP (4 rules)
    # =================================================================
    {
        "tier": 4,
        "name": "ros_drives_senescence",
        "inputs": {"ROS": "pathological"},
        "context": {},
        "outputs": {"Senescent_fraction": "+1"},
        "confidence": 0.80,
        "citation": "Cramer Ch VII.A: ROS drives senescence",
    },
    {
        "tier": 4,
        "name": "senescent_energy_drain",
        "inputs": {"Senescent_fraction": "emerging+"},
        "context": {},
        "outputs": {"ATP": "-1"},
        "confidence": 0.85,
        "citation": "Cramer Ch VII.A: senescent cells drain energy",
    },
    {
        "tier": 4,
        "name": "senescent_ros_amplification",
        "inputs": {"Senescent_fraction": "severe"},
        "context": {},
        "outputs": {"ROS": "+1"},
        "confidence": 0.80,
        "citation": "SASP amplifies ROS production",
    },
    {
        "tier": 4,
        "name": "senolytics_clear",
        "inputs": {"Senescent_fraction": "emerging+"},
        "context": {"senolytic": "high"},
        "outputs": {"Senescent_fraction": "-1"},
        "confidence": 0.85,
        "citation": "Dasatinib+quercetin clears senescent cells",
    },

    # =================================================================
    # TIER 5: NAD+ & Supplementation (4 rules)
    # =================================================================
    {
        "tier": 5,
        "name": "nad_age_decline",
        "inputs": {"NAD": "robust"},
        "context": {"age_epoch": ["transition", "old"]},
        "outputs": {"NAD": "-1"},
        "confidence": 0.85,
        "citation": "Cramer Ch VI.A.3: NAD declines with age",
    },
    {
        "tier": 5,
        "name": "nad_supplement_restores",
        "inputs": {"NAD": "depleted+"},
        "context": {"nad_supplement": "high"},
        "outputs": {"NAD": "+1"},
        "confidence": 0.75,
        "citation": "Cramer Ch VI.A.3: NMN/NR + CD38 suppression",
    },
    {
        "tier": 5,
        "name": "nad_low_dose_cd38_blocked",
        "inputs": {"NAD": "depleted"},
        "context": {"nad_supplement": "low"},
        "outputs": {},
        "confidence": 0.80,
        "citation": "C7: CD38 destroys NMN/NR at low dose",
    },
    {
        "tier": 5,
        "name": "nad_boosts_defense",
        "inputs": {"NAD": "robust"},
        "context": {},
        "outputs": {"ROS": "-1"},
        "confidence": 0.70,
        "citation": "NAD supports antioxidant defense pathways",
    },

    # =================================================================
    # TIER 6: Interventions & Transplant (6 rules)
    # =================================================================
    {
        "tier": 6,
        "name": "transplant_adds_healthy",
        "inputs": {},
        "context": {"transplant": "high"},
        "outputs": {"N_healthy": "+1", "N_deletion": "-1"},
        "confidence": 0.85,
        "citation": "Cramer Ch VIII.G: platelet-derived mitlets",
    },
    {
        "tier": 6,
        "name": "transplant_het_penalty",
        "inputs": {"N_deletion": "past_cliff"},
        "context": {"transplant": "high"},
        "outputs": {},
        "confidence": 0.90,
        "citation": "C8: hostile environment impairs engraftment",
    },
    {
        "tier": 6,
        "name": "exercise_biogenesis",
        "inputs": {},
        "context": {"exercise": "moderate+"},
        "outputs": {"N_healthy": "+1"},
        "confidence": 0.75,
        "citation": "Exercise promotes mitochondrial biogenesis",
    },
    {
        "tier": 6,
        "name": "exercise_hormesis",
        "inputs": {},
        "context": {"exercise": "moderate"},
        "outputs": {"ROS": "+1", "Membrane_potential": "+1"},
        "confidence": 0.70,
        "citation": "Moderate exercise: transient ROS -> adaptation",
    },
    {
        "tier": 6,
        "name": "yamanaka_repairs",
        "inputs": {"ATP": "compromised+"},
        "context": {"yamanaka": "high"},
        "outputs": {"N_deletion": "-1", "N_point": "-1"},
        "confidence": 0.65,
        "citation": "Partial reprogramming repairs mtDNA",
    },
    {
        "tier": 6,
        "name": "yamanaka_energy_cost",
        "inputs": {},
        "context": {"yamanaka": "high"},
        "outputs": {"ATP": "-1"},
        "confidence": 0.90,
        "citation": "Cramer Ch VIII.A: Yamanaka costs 3-5 MU/day",
    },

    # =================================================================
    # CROSS-TIER COMPOUND RULES (6 rules, tier=0)
    # =================================================================
    {
        "tier": 0,
        "name": "point_of_no_return",
        "inputs": {
            "N_deletion": "past_cliff",
            "ATP": "collapsed",
            "ROS": "pathological",
            "Senescent_fraction": "severe",
        },
        "context": {},
        "outputs": {},
        "confidence": 0.95,
        "citation": "Absorbing state: all repair pathways fail",
    },
    {
        "tier": 0,
        "name": "vicious_cycle_lock",
        "inputs": {"ROS": "pathological", "Membrane_potential": "collapsed", "ATP": "crisis-"},
        "context": {},
        "outputs": {"N_deletion": "+1", "ROS": "+1"},
        "confidence": 0.90,
        "citation": "ROS-membrane-ATP vicious cycle",
    },
    {
        "tier": 0,
        "name": "transplant_rescue",
        "inputs": {"N_deletion": "approaching_cliff", "ATP": "compromised"},
        "context": {"transplant": "high"},
        "outputs": {"N_deletion": "-1", "ATP": "+1"},
        "confidence": 0.80,
        "citation": "Transplant can rescue near-cliff patients",
    },
    {
        "tier": 0,
        "name": "cocktail_synergy",
        "inputs": {"NAD": "robust", "Senescent_fraction": "minimal"},
        "context": {"rapamycin": "high", "exercise": "moderate+"},
        "outputs": {"N_healthy": "+1", "Membrane_potential": "+1"},
        "confidence": 0.75,
        "citation": "Multi-intervention synergy",
    },
    {
        "tier": 0,
        "name": "age_transition_acceleration",
        "inputs": {},
        "context": {"age_epoch": "old"},
        "outputs": {"N_deletion": "+1", "NAD": "-1", "Senescent_fraction": "+1"},
        "confidence": 0.85,
        "citation": "Old age accelerates all damage pathways",
    },
    {
        "tier": 0,
        "name": "young_homeostasis",
        "inputs": {"N_healthy": "adequate", "ATP": "healthy"},
        "context": {"age_epoch": "young"},
        "outputs": {},
        "confidence": 0.80,
        "citation": "Young + healthy = homeostatic maintenance",
    },
]


def _evaluate_context(context_spec: dict, context: dict) -> bool:
    """Check whether all context conditions in a rule are satisfied.

    Supports:
      - age_epoch: string match, or list-of-strings ("any of these")
      - Intervention levels (rapamycin, nad_supplement, senolytic, exercise,
        yamanaka, transplant): exact string match, or "level+" for threshold
        (current >= level in the none/low/moderate/high ordering)
    """
    for key, expected in context_spec.items():
        if key == "age_epoch":
            actual = context.get("age_epoch", "young")
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            else:
                if actual != expected:
                    return False
        elif key in (
            "rapamycin", "nad_supplement", "senolytic",
            "exercise", "yamanaka", "transplant",
        ):
            actual = context.get(key, "none")
            if isinstance(expected, str) and expected.endswith("+"):
                # "moderate+" means moderate or high
                levels = ["none", "low", "moderate", "high"]
                base = expected[:-1]
                if levels.index(actual) < levels.index(base):
                    return False
            else:
                if actual != expected:
                    return False
    return True


def _evaluate_inputs(input_spec: dict, discrete_state: dict) -> bool:
    """Check whether all input bin conditions in a rule match.

    Supports three match modes:
      - "label"  -- exact match
      - "label+" -- current bin index >= label's index (this level or higher)
      - "label-" -- current bin index <= label's index (this level or lower)
    """
    for var_name, required in input_spec.items():
        current = discrete_state.get(var_name)
        if current is None:
            return False

        if required.endswith("+"):
            base_label = required[:-1]
            schema = BIN_SCHEMA[var_name]
            labels = schema["labels"]
            req_idx = labels.index(base_label)
            cur_idx = labels.index(current)
            if cur_idx < req_idx:
                return False
        elif required.endswith("-"):
            base_label = required[:-1]
            schema = BIN_SCHEMA[var_name]
            labels = schema["labels"]
            req_idx = labels.index(base_label)
            cur_idx = labels.index(current)
            if cur_idx > req_idx:
                return False
        else:
            if current != required:
                return False
    return True


def _apply_direction(
    current_label: str,
    direction: str,
    var_name: str,
) -> str:
    """Apply a directional update (+1, -1, +2, -2, 0) or absolute bin assignment."""
    schema = BIN_SCHEMA[var_name]
    labels = schema["labels"]

    # Absolute assignment: the direction IS the target bin label
    if direction in labels:
        return direction

    # Hold: "0" means no change
    if direction == "0":
        return current_label

    current_idx = labels.index(current_label)

    if direction == "+1":
        new_idx = min(current_idx + 1, len(labels) - 1)
    elif direction == "-1":
        new_idx = max(current_idx - 1, 0)
    elif direction == "+2":
        new_idx = min(current_idx + 2, len(labels) - 1)
    elif direction == "-2":
        new_idx = max(current_idx - 2, 0)
    else:
        return current_label

    return labels[new_idx]


def get_applicable_rules(
    discrete_state: dict[str, str],
    context: dict,
    rules: list[dict] | None = None,
) -> list[dict]:
    """Return the subset of rules whose conditions are satisfied.

    Parameters
    ----------
    discrete_state : dict[str, str]
        Current discretized state {var_name: bin_label}.
    context : dict
        Age epoch, intervention levels, and other context.
    rules : list[dict] or None
        Rule table to evaluate. Defaults to RULE_TABLE.

    Returns
    -------
    list[dict]
        Rules whose input and context conditions all match.
    """
    if rules is None:
        rules = RULE_TABLE
    applicable = []
    for rule in rules:
        if not _evaluate_inputs(rule["inputs"], discrete_state):
            continue
        if not _evaluate_context(rule["context"], context):
            continue
        applicable.append(rule)
    return applicable


def apply_rules(
    discrete_state: dict[str, str],
    context: dict,
    rules: list[dict] | None = None,
) -> tuple[dict[str, str], list[dict]]:
    """Apply all matching rules to produce the next discrete state.

    Rules are applied deterministically. If multiple rules try to update the
    same variable, the one with higher confidence wins. For the point_of_no_return
    (absorbing state), no outputs are applied -- the state is frozen.

    Parameters
    ----------
    discrete_state : dict[str, str]
        Current discretized state.
    context : dict
        Age epoch, intervention levels, and other context.
    rules : list[dict] or None
        Rule table. Defaults to RULE_TABLE.

    Returns
    -------
    tuple[dict[str, str], list[dict]]
        (new_state, fired_rules) -- the updated discrete state and the list
        of rules that fired.
    """
    applicable = get_applicable_rules(discrete_state, context, rules)

    # Check for point_of_no_return (absorbing state) -- freeze the state
    for rule in applicable:
        if rule["name"] == "point_of_no_return":
            return dict(discrete_state), applicable

    # Collect proposed updates, resolve conflicts by confidence
    # {var_name: (direction, confidence, rule_name)}
    proposals: dict[str, tuple[str, float, str]] = {}
    for rule in applicable:
        for var_name, direction in rule["outputs"].items():
            if var_name not in proposals or rule["confidence"] > proposals[var_name][1]:
                proposals[var_name] = (direction, rule["confidence"], rule["name"])

    # Apply winning proposals
    new_state = dict(discrete_state)
    for var_name, (direction, confidence, _) in proposals.items():
        new_state[var_name] = _apply_direction(
            new_state[var_name], direction, var_name
        )

    return new_state, applicable


def save_rules(path: str, rules: list[dict] | None = None) -> None:
    """Save rule table to JSON."""
    if rules is None:
        rules = RULE_TABLE
    with open(path, "w") as f:
        json.dump(rules, f, indent=2)


def load_rules(path: str) -> list[dict]:
    """Load rule table from JSON."""
    with open(path) as f:
        return json.load(f)
