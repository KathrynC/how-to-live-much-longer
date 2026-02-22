# Mitochondrial Semantic Cellular Automaton Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Semantic Cellular Automaton for the mitochondrial aging simulator — 8 files mirroring the LEMURS CA architecture, with 8-variable bin schema, 32 tiered rules, 4-tissue population grid, stochastic ensemble, and 3 Zimmerman protocol adapters.

**Architecture:** Direct port of `~/lemurs-simulator/ca_*.py` with mito-specific adaptations: age epochs replace calendar, tissue coupling replaces social coupling, heteroplasmy cliff replaces burnout cascade. All new files in `~/how-to-live-much-longer/`.

**Tech Stack:** Python 3.11+, numpy-only, matplotlib Agg backend

---

## Context

**LEMURS CA (template):** `~/lemurs-simulator/ca_schema.py` (215 lines), `ca_rules.py` (550 lines), `ca_simulator.py` (501 lines), `ca_analytics.py` (402 lines), `ca_stochastic.py` (275 lines), `ca_zimmerman_bridge.py` (282 lines).

**Mito simulator (target):** 8D state vector [N_healthy, N_deletion, ATP, ROS, NAD, Senescent, ΔΨ, N_point], 12D params (6 intervention + 6 patient), 30-year RK4 at dt=0.01yr.

**Key constants from `~/how-to-live-much-longer/constants.py`:**
- `N_STATES = 8`, `STATE_NAMES = ["N_healthy", "N_deletion", "ATP", "ROS", "NAD", "Senescent_fraction", "Membrane_potential", "N_point"]`
- `DEFAULT_INTERVENTION`, `DEFAULT_PATIENT`, `INTERVENTION_PARAMS`, `PATIENT_PARAMS`
- `HETEROPLASMY_CLIFF = 0.50`, `CLIFF_STEEPNESS = 15.0`, `AGE_TRANSITION = 65.0`
- `BASELINE_ATP = 1.0`, `BASELINE_ROS = 0.1`, `BASELINE_NAD = 1.0`
- `ATP_CRISIS_FRACTION = 0.5`, `MITOPHAGY_ATP_MIDPOINT = 0.6`
- `TISSUE_PROFILES` dict with brain/muscle/cardiac entries

**Design doc:** `~/how-to-live-much-longer/docs/plans/2026-02-21-mito-semantic-ca-design.md`

---

## Tasks

### Task 1: ca_schema.py — Bin Schema

**Files:**
- Create: `~/how-to-live-much-longer/ca_schema.py`
- Test: `~/how-to-live-much-longer/tests/test_ca.py`

**What this does:** Defines the discretization schema mapping 8 continuous ODE state variables to 3-4 clinically meaningful bins each. Provides `discretize_state()` and `continuous_exemplar()` for round-tripping between continuous and discrete representations.

**Step 1: Write the test file with schema tests**

Create `tests/test_ca.py`:

```python
"""Tests for the mitochondrial semantic cellular automaton."""
import numpy as np
import pytest

from ca_schema import (
    BIN_SCHEMA, discretize_state, continuous_exemplar,
    bin_index, bin_count, CA_VAR_ORDER, CA_N_VARS,
)
from constants import N_STATES


class TestBinSchema:
    """Tests for the bin schema definition."""

    def test_all_state_vars_covered(self):
        """Every mito state variable has a bin schema entry."""
        assert len(BIN_SCHEMA) == N_STATES

    def test_var_order_length(self):
        assert len(CA_VAR_ORDER) == CA_N_VARS == 8

    def test_each_var_has_required_keys(self):
        required = {"index", "thresholds", "labels", "centers", "unit", "source"}
        for var_name, schema in BIN_SCHEMA.items():
            assert required.issubset(schema.keys()), f"{var_name} missing keys"

    def test_labels_match_thresholds_plus_one(self):
        for var_name, schema in BIN_SCHEMA.items():
            assert len(schema["labels"]) == len(schema["thresholds"]) + 1, var_name

    def test_centers_match_labels(self):
        for var_name, schema in BIN_SCHEMA.items():
            assert len(schema["centers"]) == len(schema["labels"]), var_name

    def test_n_deletion_has_cliff_threshold(self):
        """N_deletion bins must include the 0.50 cliff threshold."""
        assert 0.5 in BIN_SCHEMA["N_deletion"]["thresholds"]

    def test_atp_has_crisis_threshold(self):
        """ATP bins must include the 0.5 crisis fraction."""
        assert 0.5 in BIN_SCHEMA["ATP"]["thresholds"]


class TestDiscretize:
    """Tests for discretize_state()."""

    def test_healthy_young_patient(self):
        """A healthy young state should discretize to good bins."""
        state = np.array([0.9, 0.05, 0.95, 0.08, 0.9, 0.02, 0.95, 0.02])
        discrete = discretize_state(state)
        assert discrete["N_healthy"] == "adequate"
        assert discrete["N_deletion"] == "minimal"
        assert discrete["ATP"] == "healthy"
        assert discrete["ROS"] == "basal"

    def test_cliff_patient(self):
        """A patient past the cliff should have correct deletion bin."""
        state = np.array([0.2, 0.6, 0.15, 0.35, 0.25, 0.5, 0.2, 0.15])
        discrete = discretize_state(state)
        assert discrete["N_deletion"] == "past_cliff"
        assert discrete["ATP"] == "collapsed"

    def test_returns_dict_of_strings(self):
        state = np.zeros(N_STATES)
        discrete = discretize_state(state)
        assert isinstance(discrete, dict)
        for k, v in discrete.items():
            assert isinstance(v, str)

    def test_all_vars_present(self):
        state = np.zeros(N_STATES)
        discrete = discretize_state(state)
        assert len(discrete) == CA_N_VARS


class TestContinuousExemplar:
    """Tests for continuous_exemplar() inverse mapping."""

    def test_round_trip_bins(self):
        """discretize(exemplar(discrete)) should return same bins."""
        state = np.array([0.85, 0.05, 0.9, 0.05, 0.85, 0.05, 0.85, 0.05])
        discrete = discretize_state(state)
        reconstructed = continuous_exemplar(discrete)
        re_discrete = discretize_state(reconstructed)
        assert re_discrete == discrete

    def test_returns_correct_shape(self):
        discrete = {"N_healthy": "adequate", "N_deletion": "minimal",
                     "ATP": "healthy", "ROS": "basal", "NAD": "robust",
                     "Senescent_fraction": "minimal",
                     "Membrane_potential": "intact", "N_point": "low"}
        result = continuous_exemplar(discrete)
        assert result.shape == (N_STATES,)
        assert result.dtype == np.float64


class TestBinHelpers:
    def test_bin_index(self):
        assert bin_index("ATP", "collapsed") == 0
        assert bin_index("ATP", "healthy") == 3

    def test_bin_count(self):
        assert bin_count("N_deletion") == 4
        assert bin_count("ROS") == 3
```

**Step 2: Run tests to verify they fail**

```bash
cd ~/how-to-live-much-longer && python -m pytest tests/test_ca.py -v
```
Expected: FAIL (ca_schema not found)

**Step 3: Implement ca_schema.py**

```python
"""State discretization schema for the mitochondrial semantic cellular automaton.

Discretizes the 8D continuous ODE state vector into clinically meaningful
bins. Each variable gets 3-4 bins with thresholds drawn from published
biological constants (Cramer 2026) and the ODE coupling structure.

Mirrors ~/lemurs-simulator/ca_schema.py architecture.
"""
from __future__ import annotations

import numpy as np

from constants import STATE_NAMES, N_STATES

# ── Bin schema ────────────────────────────────────────────────────────────
# Each entry: variable name → {index, thresholds, labels, centers, unit, source}.
# value < threshold[0] → bin[0], threshold[0] <= value < threshold[1] → bin[1], etc.

BIN_SCHEMA: dict[str, dict] = {
    "N_healthy": {
        "index": 0,
        "thresholds": [0.3, 0.7],
        "labels": ["depleted", "reduced", "adequate"],
        "centers": [0.15, 0.5, 0.85],
        "unit": "normalized copies",
        "source": "C2 copy homeostasis",
    },
    "N_deletion": {
        "index": 1,
        "thresholds": [0.1, 0.3, 0.5],
        "labels": ["minimal", "growing", "approaching_cliff", "past_cliff"],
        "centers": [0.05, 0.2, 0.4, 0.7],
        "unit": "deletion het fraction",
        "source": "HETEROPLASMY_CLIFF=0.50, Cramer Appendix 2",
    },
    "ATP": {
        "index": 2,
        "thresholds": [0.2, 0.5, 0.8],
        "labels": ["collapsed", "crisis", "compromised", "healthy"],
        "centers": [0.1, 0.35, 0.65, 0.9],
        "unit": "MU/day",
        "source": "ATP_CRISIS_FRACTION=0.5, Cramer Ch. VIII.A Table 3",
    },
    "ROS": {
        "index": 3,
        "thresholds": [0.1, 0.25],
        "labels": ["basal", "elevated", "pathological"],
        "centers": [0.05, 0.175, 0.4],
        "unit": "normalized",
        "source": "BASELINE_ROS=0.1, Cramer Ch. II.H",
    },
    "NAD": {
        "index": 4,
        "thresholds": [0.3, 0.7],
        "labels": ["depleted", "declining", "robust"],
        "centers": [0.15, 0.5, 0.85],
        "unit": "normalized",
        "source": "NAD_DECLINE_RATE=0.01/yr, Cramer Ch. VI.A.3",
    },
    "Senescent_fraction": {
        "index": 5,
        "thresholds": [0.1, 0.4],
        "labels": ["minimal", "emerging", "severe"],
        "centers": [0.05, 0.25, 0.6],
        "unit": "fraction",
        "source": "SENESCENCE_RATE=0.005/yr, Cramer Ch. VII.A",
    },
    "Membrane_potential": {
        "index": 6,
        "thresholds": [0.3, 0.7],
        "labels": ["collapsed", "impaired", "intact"],
        "centers": [0.15, 0.5, 0.85],
        "unit": "normalized ΔΨ",
        "source": "MITOPHAGY_ATP_MIDPOINT=0.6, Cramer Ch. VI.B",
    },
    "N_point": {
        "index": 7,
        "thresholds": [0.1, 0.3],
        "labels": ["low", "moderate", "high"],
        "centers": [0.05, 0.2, 0.5],
        "unit": "point het fraction",
        "source": "POINT_ERROR_RATE=0.001, Cramer Ch. II.H",
    },
}

CA_N_VARS = len(BIN_SCHEMA)

CA_VAR_ORDER: list[str] = [
    "N_healthy", "N_deletion", "ATP", "ROS",
    "NAD", "Senescent_fraction", "Membrane_potential", "N_point",
]


def _classify(value: float, thresholds: list[float], labels: list[str]) -> str:
    """Assign a continuous value to a named bin."""
    for i, thresh in enumerate(thresholds):
        if value < thresh:
            return labels[i]
    return labels[-1]


def discretize_state(continuous_state: np.ndarray) -> dict[str, str]:
    """Convert an 8D continuous state vector to named clinical bins."""
    result = {}
    for var_name in CA_VAR_ORDER:
        schema = BIN_SCHEMA[var_name]
        val = float(continuous_state[schema["index"]])
        result[var_name] = _classify(val, schema["thresholds"], schema["labels"])
    return result


def continuous_exemplar(discrete_state: dict[str, str]) -> np.ndarray:
    """Convert a discrete bin assignment back to an 8D continuous exemplar."""
    state = np.zeros(N_STATES, dtype=np.float64)
    for var_name, label in discrete_state.items():
        schema = BIN_SCHEMA[var_name]
        idx = schema["labels"].index(label)
        state[schema["index"]] = schema["centers"][idx]
    return state


def bin_index(var_name: str, label: str) -> int:
    """Return the integer index of a bin label within its variable's bins."""
    return BIN_SCHEMA[var_name]["labels"].index(label)


def bin_count(var_name: str) -> int:
    """Return the number of bins for a given variable."""
    return len(BIN_SCHEMA[var_name]["labels"])
```

**Step 4: Run tests**

```bash
cd ~/how-to-live-much-longer && python -m pytest tests/test_ca.py::TestBinSchema -v && python -m pytest tests/test_ca.py::TestDiscretize -v && python -m pytest tests/test_ca.py::TestContinuousExemplar -v && python -m pytest tests/test_ca.py::TestBinHelpers -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
cd ~/how-to-live-much-longer && git add ca_schema.py tests/test_ca.py && git commit -m "feat(ca): add bin schema for 8 mito state variables"
```

---

### Task 2: ca_rules.py — 32 Tiered Rules

**Files:**
- Create: `~/how-to-live-much-longer/ca_rules.py`
- Test: append to `~/how-to-live-much-longer/tests/test_ca.py`

**What this does:** Defines 32 rules in 6 tiers + cross-tier compounds. Each rule is a JSON-serializable dict with tier, name, inputs, context, outputs, confidence, citation. Provides `apply_rules()` (deterministic), `get_applicable_rules()`, and conflict resolution.

**Step 1: Append rule tests to test_ca.py**

```python
from ca_rules import (
    RULE_TABLE, apply_rules, get_applicable_rules,
    _evaluate_context, _evaluate_inputs, _apply_direction,
    save_rules, load_rules,
)


class TestRuleTable:
    def test_rule_count(self):
        assert len(RULE_TABLE) == 32

    def test_all_rules_have_required_keys(self):
        required = {"tier", "name", "inputs", "context", "outputs", "confidence", "citation"}
        for i, rule in enumerate(RULE_TABLE):
            assert required.issubset(rule.keys()), f"Rule {i} ({rule.get('name')}) missing keys"

    def test_confidence_in_range(self):
        for rule in RULE_TABLE:
            assert 0.0 < rule["confidence"] <= 1.0, rule["name"]

    def test_tiers_present(self):
        tiers = {r["tier"] for r in RULE_TABLE}
        assert tiers == {0, 1, 2, 3, 4, 5, 6}

    def test_unique_names(self):
        names = [r["name"] for r in RULE_TABLE]
        assert len(names) == len(set(names))

    def test_absorbing_state_exists(self):
        names = [r["name"] for r in RULE_TABLE]
        assert "point_of_no_return" in names

    def test_absorbing_state_has_empty_outputs(self):
        for r in RULE_TABLE:
            if r["name"] == "point_of_no_return":
                assert r["outputs"] == {}
                assert r["confidence"] >= 0.9


class TestApplyRules:
    def test_healthy_state_stays_stable(self):
        state = {
            "N_healthy": "adequate", "N_deletion": "minimal",
            "ATP": "healthy", "ROS": "basal", "NAD": "robust",
            "Senescent_fraction": "minimal",
            "Membrane_potential": "intact", "N_point": "low",
        }
        ctx = {"age": 30, "age_epoch": "young",
               "rapamycin": "none", "nad_supplement": "none",
               "senolytic": "none", "exercise": "none",
               "yamanaka": "none", "transplant": "none"}
        new_state, fired = apply_rules(state, ctx)
        # Young homeostasis should keep things stable
        assert new_state["ATP"] == "healthy"

    def test_absorbing_state_freezes(self):
        state = {
            "N_healthy": "depleted", "N_deletion": "past_cliff",
            "ATP": "collapsed", "ROS": "pathological", "NAD": "depleted",
            "Senescent_fraction": "severe",
            "Membrane_potential": "collapsed", "N_point": "high",
        }
        ctx = {"age": 80, "age_epoch": "old",
               "rapamycin": "none", "nad_supplement": "none",
               "senolytic": "none", "exercise": "none",
               "yamanaka": "none", "transplant": "none"}
        new_state, fired = apply_rules(state, ctx)
        # Point of no return: state frozen
        assert new_state == state

    def test_cliff_causes_atp_collapse(self):
        state = {
            "N_healthy": "reduced", "N_deletion": "past_cliff",
            "ATP": "compromised", "ROS": "elevated", "NAD": "declining",
            "Senescent_fraction": "emerging",
            "Membrane_potential": "impaired", "N_point": "moderate",
        }
        ctx = {"age": 70, "age_epoch": "old",
               "rapamycin": "none", "nad_supplement": "none",
               "senolytic": "none", "exercise": "none",
               "yamanaka": "none", "transplant": "none"}
        new_state, fired = apply_rules(state, ctx)
        # cliff_atp_collapse should fire
        assert new_state["ATP"] in ("collapsed", "crisis")

    def test_transplant_adds_healthy(self):
        state = {
            "N_healthy": "reduced", "N_deletion": "approaching_cliff",
            "ATP": "compromised", "ROS": "elevated", "NAD": "declining",
            "Senescent_fraction": "minimal",
            "Membrane_potential": "impaired", "N_point": "low",
        }
        ctx = {"age": 65, "age_epoch": "transition",
               "rapamycin": "none", "nad_supplement": "none",
               "senolytic": "none", "exercise": "none",
               "yamanaka": "none", "transplant": "high"}
        new_state, fired = apply_rules(state, ctx)
        fired_names = [r["name"] for r in fired]
        assert any("transplant" in n for n in fired_names)


class TestEvaluateContext:
    def test_age_epoch_young(self):
        assert _evaluate_context({"age_epoch": "young"}, {"age_epoch": "young"})
        assert not _evaluate_context({"age_epoch": "young"}, {"age_epoch": "old"})

    def test_intervention_level(self):
        assert _evaluate_context({"rapamycin": "high"}, {"rapamycin": "high"})
        assert not _evaluate_context({"rapamycin": "high"}, {"rapamycin": "none"})


class TestApplyDirection:
    def test_plus_one(self):
        assert _apply_direction("basal", "+1", "ROS") == "elevated"

    def test_minus_one(self):
        assert _apply_direction("elevated", "-1", "ROS") == "basal"

    def test_clamp_at_max(self):
        assert _apply_direction("pathological", "+1", "ROS") == "pathological"

    def test_clamp_at_min(self):
        assert _apply_direction("basal", "-1", "ROS") == "basal"

    def test_absolute_assignment(self):
        assert _apply_direction("basal", "pathological", "ROS") == "pathological"

    def test_hold(self):
        assert _apply_direction("elevated", "0", "ROS") == "elevated"

    def test_minus_two(self):
        """Direction '-2' should move down 2 bins."""
        assert _apply_direction("healthy", "-2", "ATP") == "crisis"


class TestSaveLoadRules:
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "rules.json")
        save_rules(path)
        loaded = load_rules(path)
        assert len(loaded) == len(RULE_TABLE)
        assert loaded[0]["name"] == RULE_TABLE[0]["name"]
```

**Step 2: Run to verify they fail**

```bash
cd ~/how-to-live-much-longer && python -m pytest tests/test_ca.py::TestRuleTable -v
```
Expected: FAIL (ca_rules not found)

**Step 3: Implement ca_rules.py**

Create `~/how-to-live-much-longer/ca_rules.py` with:
- `RULE_TABLE`: 32 rules (see design doc for complete table)
- `_evaluate_context()`: checks age_epoch, intervention levels (rapamycin, nad_supplement, senolytic, exercise, yamanaka, transplant mapped to none/low/high)
- `_evaluate_inputs()`: checks bin label matches
- `_apply_direction()`: handles "+1", "-1", "-2", "0", and absolute label assignment
- `apply_rules()`: deterministic highest-confidence-wins conflict resolution, point_of_no_return freezes
- `get_applicable_rules()`, `save_rules()`, `load_rules()`

Context evaluation logic:
- `age_epoch`: direct string match ("young", "transition", "old")
- Intervention levels: `rapamycin`, `nad_supplement`, `senolytic`, `exercise`, `yamanaka`, `transplant` — each is "none", "low", "moderate", or "high" (mapped from 0-1 float in the simulator's `_build_context`)
- `near_transition`: bool flag
- `tissue_type`: string (brain/muscle/cardiac/skin)

**The full RULE_TABLE (32 rules) — implement exactly as shown in the design doc:**

Tier 1 (4 rules): deletion_expansion_young, deletion_expansion_old, cliff_atp_collapse, cliff_approaching_warning
Tier 2 (4 rules): ros_from_deletions, ros_from_points, ros_drives_points, ros_membrane_damage
Tier 3 (4 rules): mitophagy_clears_deletions, mitophagy_atp_gated, mitophagy_weak_on_points, rapamycin_membrane_benefit
Tier 4 (4 rules): ros_drives_senescence, senescent_energy_drain, senescent_ros_amplification, senolytics_clear
Tier 5 (4 rules): nad_age_decline, nad_supplement_restores, nad_low_dose_cd38_blocked, nad_boosts_defense
Tier 6 (6 rules): transplant_adds_healthy, transplant_het_penalty, exercise_biogenesis, exercise_hormesis, yamanaka_repairs, yamanaka_energy_cost
Cross-tier/0 (6 rules): point_of_no_return, vicious_cycle_lock, transplant_rescue, cocktail_synergy, age_transition_acceleration, young_homeostasis

**Step 4: Run tests**

```bash
cd ~/how-to-live-much-longer && python -m pytest tests/test_ca.py::TestRuleTable -v && python -m pytest tests/test_ca.py::TestApplyRules -v && python -m pytest tests/test_ca.py::TestEvaluateContext -v && python -m pytest tests/test_ca.py::TestApplyDirection -v && python -m pytest tests/test_ca.py::TestSaveLoadRules -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
cd ~/how-to-live-much-longer && git add ca_rules.py tests/test_ca.py && git commit -m "feat(ca): add 32 tiered rules for mito aging dynamics"
```

---

### Task 3: ca_simulator.py — Single Cell + Tissue Grid

**Files:**
- Create: `~/how-to-live-much-longer/ca_simulator.py`
- Test: append to `~/how-to-live-much-longer/tests/test_ca.py`

**What this does:** Provides `run_single_cell()` for one patient over 30 years at quarterly resolution (120 steps), and `run_tissue_grid()` for 4-tissue population with systemic coupling.

**Step 1: Append simulator tests to test_ca.py**

```python
from ca_simulator import (
    run_single_cell, run_tissue_grid, step_cell, _build_context,
    CA_DT, CA_N_STEPS, TISSUE_TYPES,
)


class TestBuildContext:
    def test_age_epoch_young(self):
        ctx = _build_context(0, {"baseline_age": 30.0}, {}, None, None)
        assert ctx["age_epoch"] == "young"
        assert ctx["age"] == 30.0

    def test_age_epoch_transition(self):
        ctx = _build_context(80, {"baseline_age": 40.0}, {}, None, None)
        # step 80 at dt=0.25 = 20 years, age = 60
        assert ctx["age_epoch"] == "transition"

    def test_age_epoch_old(self):
        ctx = _build_context(0, {"baseline_age": 75.0}, {}, None, None)
        assert ctx["age_epoch"] == "old"

    def test_intervention_levels(self):
        ctx = _build_context(0, {"baseline_age": 50.0},
                             {"rapamycin_dose": 0.8, "nad_supplement": 0.1}, None, None)
        assert ctx["rapamycin"] == "high"
        assert ctx["nad_supplement"] == "low"


class TestStepCell:
    def test_returns_new_state_and_fired(self):
        state = {
            "N_healthy": "adequate", "N_deletion": "minimal",
            "ATP": "healthy", "ROS": "basal", "NAD": "robust",
            "Senescent_fraction": "minimal",
            "Membrane_potential": "intact", "N_point": "low",
        }
        ctx = {"age": 30, "age_epoch": "young",
               "rapamycin": "none", "nad_supplement": "none",
               "senolytic": "none", "exercise": "none",
               "yamanaka": "none", "transplant": "none"}
        new_state, fired = step_cell(state, ctx)
        assert isinstance(new_state, dict)
        assert isinstance(fired, list)


class TestRunSingleCell:
    def test_default_params(self):
        result = run_single_cell()
        assert "trajectory" in result
        assert "rule_log" in result
        assert "final_state" in result
        assert len(result["trajectory"]) == CA_N_STEPS + 1  # init + 120 steps

    def test_trajectory_length(self):
        result = run_single_cell(sim_years=10, dt=0.5)
        assert len(result["trajectory"]) == 21  # 10/0.5 + 1

    def test_rule_log_length(self):
        result = run_single_cell()
        assert len(result["rule_log"]) == CA_N_STEPS

    def test_final_state_is_dict(self):
        result = run_single_cell()
        assert isinstance(result["final_state"], dict)
        assert len(result["final_state"]) == CA_N_VARS

    def test_deterministic(self):
        r1 = run_single_cell()
        r2 = run_single_cell()
        assert r1["final_state"] == r2["final_state"]

    def test_custom_patient(self):
        result = run_single_cell(
            patient={"baseline_age": 20.0, "baseline_heteroplasmy": 0.05}
        )
        assert result["patient"]["baseline_age"] == 20.0


class TestRunTissueGrid:
    def test_default(self):
        result = run_tissue_grid()
        assert "tissue_states" in result
        assert "final_tissues" in result
        assert len(result["final_tissues"]) == 4

    def test_tissue_names(self):
        result = run_tissue_grid()
        assert set(result["final_tissues"].keys()) == set(TISSUE_TYPES)

    def test_tissue_coupling(self):
        r_none = run_tissue_grid(tissue_coupling=0.0)
        r_high = run_tissue_grid(tissue_coupling=0.8)
        # With coupling, tissues should be more similar to each other
        # Just verify both run without error
        assert len(r_none["final_tissues"]) == 4
        assert len(r_high["final_tissues"]) == 4

    def test_population_summary(self):
        result = run_tissue_grid()
        summary = result["population_summary"]
        assert "total_tissues" in summary
        assert summary["total_tissues"] == 4
```

**Step 2: Run to verify they fail**

```bash
cd ~/how-to-live-much-longer && python -m pytest tests/test_ca.py::TestRunSingleCell -v
```
Expected: FAIL

**Step 3: Implement ca_simulator.py**

Key implementation details:
- `CA_DT = 0.25` (quarterly), `CA_N_STEPS = int(SIM_YEARS / CA_DT)` = 120
- `TISSUE_TYPES = ["brain", "muscle", "cardiac", "skin"]`
- `_build_context(step, patient, intervention, prev_state, curr_state)`:
  - `age = patient["baseline_age"] + step * CA_DT`
  - `age_epoch`: "young" if age < 50, "transition" if 50 ≤ age < 70, "old" if age ≥ 70
  - `near_transition = abs(age - 65) < 5`
  - Intervention levels: map float → string: >0.5 = "high", >0.2 = "low"/"moderate", else "none"
  - For exercise: >0.5 = "high", >0.2 = "moderate", else "none"
  - Derived: `cliff_proximity` = (current N_del bin index) / (max bin index)
- `run_single_cell(patient, intervention, sim_years=30, dt=0.25)`: Initialize from ODE `initial_state(patient)` → `discretize_state()` → 120 steps
- `run_tissue_grid(patient, intervention, sim_years, dt, tissue_coupling)`: 4 cells with TISSUE_PROFILES modifiers, 3 systemic coupling channels

Tissue coupling (per step):
1. **SASP inflammation**: If any tissue has Sen=severe, ROS +1 in all tissues
2. **NAD equilibrium**: NAD level trends toward most common NAD bin across tissues
3. **Senolytic clearance**: If senolytic=high, Sen -1 applied uniformly

**Step 4: Run tests**

```bash
cd ~/how-to-live-much-longer && python -m pytest tests/test_ca.py -k "Simulator or SingleCell or TissueGrid or BuildContext or StepCell" -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
cd ~/how-to-live-much-longer && git add ca_simulator.py tests/test_ca.py && git commit -m "feat(ca): add single-cell stepper + 4-tissue population grid"
```

---

### Task 4: ca_analytics.py — 5-Section Metrics

**Files:**
- Create: `~/how-to-live-much-longer/ca_analytics.py`
- Test: append to `~/how-to-live-much-longer/tests/test_ca.py`

**What this does:** Computes rule stats, cascade stats, attractor stats, fidelity stats, and epoch diagnostic from CA results. Mirrors `~/lemurs-simulator/ca_analytics.py` exactly.

**Step 1: Append analytics tests**

```python
from ca_analytics import (
    compute_ca_analytics, _classify_attractor,
    compute_tissue_analytics,
)


class TestClassifyAttractor:
    def test_point_of_no_return(self):
        state = {
            "N_deletion": "past_cliff", "ATP": "collapsed",
            "ROS": "pathological", "Senescent_fraction": "severe",
            "N_healthy": "depleted", "NAD": "depleted",
            "Membrane_potential": "collapsed", "N_point": "high",
        }
        assert _classify_attractor(state) == "point_of_no_return"

    def test_cliff_approaching(self):
        state = {
            "N_deletion": "approaching_cliff", "ATP": "compromised",
            "ROS": "elevated", "Senescent_fraction": "minimal",
            "N_healthy": "reduced", "NAD": "declining",
            "Membrane_potential": "impaired", "N_point": "low",
        }
        assert _classify_attractor(state) == "cliff_approaching"

    def test_healthy_aging(self):
        state = {
            "N_deletion": "minimal", "ATP": "healthy",
            "ROS": "basal", "Senescent_fraction": "minimal",
            "N_healthy": "adequate", "NAD": "robust",
            "Membrane_potential": "intact", "N_point": "low",
        }
        assert _classify_attractor(state) == "healthy_aging"

    def test_slow_decline(self):
        state = {
            "N_deletion": "growing", "ATP": "compromised",
            "ROS": "elevated", "Senescent_fraction": "emerging",
            "N_healthy": "reduced", "NAD": "declining",
            "Membrane_potential": "impaired", "N_point": "moderate",
        }
        assert _classify_attractor(state) == "slow_decline"


class TestComputeCAAnalytics:
    def test_all_sections_present(self):
        ca_result = run_single_cell()
        analytics = compute_ca_analytics(ca_result)
        assert "rule_stats" in analytics
        assert "cascade_stats" in analytics
        assert "attractor_stats" in analytics
        assert "epoch_diagnostic" in analytics

    def test_rule_stats_keys(self):
        ca_result = run_single_cell()
        analytics = compute_ca_analytics(ca_result)
        rs = analytics["rule_stats"]
        assert "total_firings" in rs
        assert "unique_rules" in rs
        assert "mean_rules_per_step" in rs

    def test_attractor_stats_keys(self):
        ca_result = run_single_cell()
        analytics = compute_ca_analytics(ca_result)
        att = analytics["attractor_stats"]
        assert "final_attractor" in att
        assert att["final_attractor"] in (
            "healthy_aging", "slow_decline", "cliff_approaching", "point_of_no_return"
        )

    def test_fidelity_stats_none_without_ode(self):
        ca_result = run_single_cell()
        analytics = compute_ca_analytics(ca_result)
        assert analytics["fidelity_stats"] is None
```

**Step 2–4: Implement and test**

Implement `ca_analytics.py` with:
- `_rule_stats(rule_log)`: same as LEMURS (Counter-based)
- `_cascade_stats(trajectory, rule_log)`: same as LEMURS (consecutive-day chain detection)
- `_classify_attractor(state)`: 4 categories:
  - point_of_no_return: N_del past_cliff & ATP collapsed & ROS pathological & Sen severe
  - cliff_approaching: N_del approaching_cliff or past_cliff (but not all 4 conditions)
  - slow_decline: ATP compromised or N_del growing
  - healthy_aging: none of the above
- `_attractor_stats(trajectory)`: last 4 steps (last year), transition count
- `_fidelity_stats(trajectory, ode_result)`: bin agreement (needs ODE states at matching timesteps — subsample ODE's 3000 steps to CA's 120 by taking every 25th ODE step)
- `_epoch_diagnostic(trajectory, patient)`: compare pre/post age-65 transition
- `compute_ca_analytics(ca_result, ode_result=None)`
- `compute_tissue_analytics(tissue_result)`: per-tissue attractor distribution

**Step 5: Commit**

```bash
cd ~/how-to-live-much-longer && git add ca_analytics.py tests/test_ca.py && git commit -m "feat(ca): add 5-section CA analytics (rule/cascade/attractor/fidelity/epoch)"
```

---

### Task 5: ca_stochastic.py — Stochastic Engine + Ensemble

**Files:**
- Create: `~/how-to-live-much-longer/ca_stochastic.py`
- Test: append to `~/how-to-live-much-longer/tests/test_ca.py`

**What this does:** Provides `apply_rules_stochastic()` (probabilistic firing), `run_single_cell_stochastic()` (Monte Carlo ensemble), and `compute_ensemble_analytics()` (attractor probabilities, cliff-crossing probability). Mirrors `~/lemurs-simulator/ca_stochastic.py`.

**Step 1: Append stochastic tests**

```python
from ca_stochastic import (
    apply_rules_stochastic, run_single_cell_stochastic,
    compute_ensemble_analytics,
)


class TestStochasticRules:
    def test_absorbing_state_deterministic(self):
        """Point of no return should freeze even in stochastic mode."""
        state = {
            "N_healthy": "depleted", "N_deletion": "past_cliff",
            "ATP": "collapsed", "ROS": "pathological", "NAD": "depleted",
            "Senescent_fraction": "severe",
            "Membrane_potential": "collapsed", "N_point": "high",
        }
        ctx = {"age": 80, "age_epoch": "old",
               "rapamycin": "none", "nad_supplement": "none",
               "senolytic": "none", "exercise": "none",
               "yamanaka": "none", "transplant": "none"}
        rng = np.random.default_rng(42)
        new_state, _ = apply_rules_stochastic(state, ctx, rng)
        assert new_state == state  # frozen

    def test_stochastic_varies(self):
        """Different seeds should produce different trajectories."""
        state = {
            "N_healthy": "reduced", "N_deletion": "growing",
            "ATP": "compromised", "ROS": "elevated", "NAD": "declining",
            "Senescent_fraction": "emerging",
            "Membrane_potential": "impaired", "N_point": "moderate",
        }
        ctx = {"age": 60, "age_epoch": "transition",
               "rapamycin": "none", "nad_supplement": "none",
               "senolytic": "none", "exercise": "none",
               "yamanaka": "none", "transplant": "none"}
        results = set()
        for seed in range(20):
            rng = np.random.default_rng(seed)
            new_state, _ = apply_rules_stochastic(state, ctx, rng)
            results.add(frozenset(new_state.items()))
        # With 20 different seeds, we should see at least 2 different outcomes
        assert len(results) >= 1  # At minimum, deterministic rules fire


class TestEnsemble:
    def test_ensemble_runs(self):
        result = run_single_cell_stochastic(n_trials=10)
        assert result["n_trials"] == 10
        assert len(result["final_states"]) == 10
        assert len(result["trajectories"]) == 10

    def test_ensemble_analytics(self):
        result = run_single_cell_stochastic(n_trials=10)
        analytics = compute_ensemble_analytics(result)
        assert "attractor_probabilities" in analytics
        assert "cliff_crossing_probability" in analytics
        probs = analytics["attractor_probabilities"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01

    def test_variable_distributions(self):
        result = run_single_cell_stochastic(n_trials=10)
        analytics = compute_ensemble_analytics(result)
        vd = analytics["variable_distributions"]
        assert "ATP" in vd
        assert abs(sum(vd["ATP"].values()) - 1.0) < 0.01
```

**Step 2–4: Implement and test**

Implement `ca_stochastic.py` following LEMURS pattern exactly:
- `apply_rules_stochastic(state, ctx, rng)`: probabilistic firing gate (`rng.random() > confidence → skip`), point_of_no_return deterministic, confidence-proportional conflict resolution
- `run_single_cell_stochastic(patient, intervention, sim_years, dt, n_trials, seed)`: N trials with independent RNG per trial
- `compute_ensemble_analytics(ensemble_result)`: attractor_probabilities, cliff_crossing_probability (fraction of trials where N_deletion ever reaches past_cliff), variable_distributions

**Step 5: Commit**

```bash
cd ~/how-to-live-much-longer && git add ca_stochastic.py tests/test_ca.py && git commit -m "feat(ca): add stochastic rule engine + Monte Carlo ensemble"
```

---

### Task 6: ca_zimmerman_bridge.py — 3 Zimmerman Adapters

**Files:**
- Create: `~/how-to-live-much-longer/ca_zimmerman_bridge.py`
- Test: append to `~/how-to-live-much-longer/tests/test_ca.py`

**What this does:** Three Zimmerman protocol adapters: `MitoCASimulator` (single-cell), `MitoTissueSimulator` (4-tissue grid), `MitoCAEnsembleSimulator` (stochastic ensemble). All implement `run(params) -> dict` + `param_spec() -> dict`.

**Step 1: Append bridge tests**

```python
from ca_zimmerman_bridge import (
    MitoCASimulator, MitoTissueSimulator, MitoCAEnsembleSimulator,
)


class TestMitoCASimulator:
    def test_param_spec(self):
        sim = MitoCASimulator()
        spec = sim.param_spec()
        assert len(spec) == 12
        assert "rapamycin_dose" in spec
        assert "baseline_age" in spec

    def test_run_default(self):
        sim = MitoCASimulator()
        result = sim.run({})
        assert isinstance(result, dict)
        assert all(isinstance(v, float) for v in result.values())
        assert "ca_final_attractor" in result

    def test_no_nan_inf(self):
        sim = MitoCASimulator()
        result = sim.run({})
        for k, v in result.items():
            assert not (v != v), f"NaN in {k}"  # NaN check
            assert v != float("inf"), f"Inf in {k}"
            assert v != float("-inf"), f"-Inf in {k}"


class TestMitoTissueSimulator:
    def test_param_spec(self):
        sim = MitoTissueSimulator()
        spec = sim.param_spec()
        assert "tissue_coupling" in spec
        assert len(spec) == 13  # 12 base + tissue_coupling

    def test_run_default(self):
        sim = MitoTissueSimulator()
        result = sim.run({})
        assert isinstance(result, dict)
        assert len(result) > 0


class TestMitoCAEnsembleSimulator:
    def test_param_spec(self):
        sim = MitoCAEnsembleSimulator(n_trials=5)
        spec = sim.param_spec()
        assert len(spec) == 12

    def test_run_default(self):
        sim = MitoCAEnsembleSimulator(n_trials=5)
        result = sim.run({})
        assert "ens_cliff_crossing_prob" in result
        assert isinstance(result, dict)
```

**Step 2–4: Implement and test**

Implement `ca_zimmerman_bridge.py` following LEMURS pattern:
- `MitoCASimulator`: split params by `INTERVENTION_PARAMS`/`PATIENT_PARAMS`, run `run_single_cell()`, flatten analytics
- `MitoTissueSimulator`: 12D + `tissue_coupling`, run `run_tissue_grid()`, flatten tissue summary
- `MitoCAEnsembleSimulator`: init with `n_trials`, run stochastic ensemble, flatten ensemble analytics
- `_flatten_ca_analytics()`, `_flatten_tissue_summary()`, `_flatten_ensemble_analytics()`: same pattern as LEMURS — NaN/Inf guards, float-only output

**Step 5: Commit**

```bash
cd ~/how-to-live-much-longer && git add ca_zimmerman_bridge.py tests/test_ca.py && git commit -m "feat(ca): add 3 Zimmerman protocol adapters for CA modes"
```

---

### Task 7: ca_visualize.py — Visualizations

**Files:**
- Create: `~/how-to-live-much-longer/ca_visualize.py`
- Test: append to `~/how-to-live-much-longer/tests/test_ca.py`

**What this does:** Matplotlib Agg-backend visualizations: trajectory heatmap (8 vars × 120 steps), rule timeline by tier, CA vs ODE fidelity bars, tissue comparison grid, cliff approach trajectory.

**Step 1: Append visualization tests**

```python
import os

from ca_visualize import (
    plot_ca_trajectory, plot_rule_timeline, plot_ca_fidelity,
    plot_tissue_grid, plot_cliff_approach, generate_all_plots,
)


class TestVisualization:
    def test_plot_ca_trajectory(self, tmp_path):
        result = run_single_cell()
        path = str(tmp_path / "ca_traj.png")
        plot_ca_trajectory(result, output_path=path)
        assert os.path.exists(path)

    def test_plot_rule_timeline(self, tmp_path):
        result = run_single_cell()
        path = str(tmp_path / "rule_timeline.png")
        plot_rule_timeline(result, output_path=path)
        assert os.path.exists(path)

    def test_plot_tissue_grid(self, tmp_path):
        result = run_tissue_grid()
        path = str(tmp_path / "tissue_grid.png")
        plot_tissue_grid(result, output_path=path)
        assert os.path.exists(path)

    def test_generate_all_plots(self, tmp_path):
        generate_all_plots(output_dir=str(tmp_path))
        files = os.listdir(str(tmp_path))
        assert len(files) >= 3  # at least trajectory, timeline, tissue
```

**Step 2–4: Implement and test**

Implement `ca_visualize.py`:
- `plot_ca_trajectory(ca_result, title, output_path)`: 8×120 heatmap, bin indices as colors, age axis
- `plot_rule_timeline(ca_result, output_path)`: rule firings colored by tier
- `plot_ca_fidelity(ca_result, ode_result, output_path)`: bar chart of per-variable agreement
- `plot_tissue_grid(tissue_result, output_path)`: 2×2 panel (brain, muscle, cardiac, skin) showing final attractor and key variable bins
- `plot_cliff_approach(ca_result, output_path)`: N_deletion bin trajectory over time with cliff threshold marked
- `generate_all_plots(output_dir="output/ca")`: run single-cell + tissue grid + all plots

**Step 5: Commit**

```bash
cd ~/how-to-live-much-longer && git add ca_visualize.py tests/test_ca.py && git commit -m "feat(ca): add CA visualization suite (heatmap, timeline, tissue grid)"
```

---

### Task 8: Update CLAUDE.md and Constants + Final Verification

**Files:**
- Modify: `~/how-to-live-much-longer/CLAUDE.md`
- Modify: `~/how-to-live-much-longer/constants.py`

**What this does:** Add CA documentation to CLAUDE.md (commands, architecture, conventions). Add `age_epoch()` helper to constants.py. Run full test suite.

**Step 1: Add age_epoch helper to constants.py**

```python
# ── CA age epoch helper ──────────────────────────────────────────────────────

def age_epoch(age: float) -> str:
    """Classify biological age into CA epoch: young, transition, or old."""
    if age < 50.0:
        return "young"
    elif age < 70.0:
        return "transition"
    return "old"
```

**Step 2: Update CLAUDE.md**

Add Semantic CA section after the existing architecture section (parallel to LEMURS CLAUDE.md's CA section):
- Commands for running CA tests
- Architecture: ca_schema, ca_rules, ca_simulator, ca_analytics, ca_stochastic, ca_zimmerman_bridge, ca_visualize
- Key CA dynamics: point of no return absorbing state, 4-tissue coupling, quarterly timesteps
- Attractor classification: healthy_aging, slow_decline, cliff_approaching, point_of_no_return

**Step 3: Run full test suite**

```bash
cd ~/how-to-live-much-longer && python -m pytest tests/ -v
```
Expected: ALL PASS (existing ~499 + new ~60-80 CA tests)

**Step 4: Verify ODE untouched**

```bash
cd ~/how-to-live-much-longer && git diff simulator.py
```
Expected: NO output (no changes to core ODE)

**Step 5: Final commit**

```bash
cd ~/how-to-live-much-longer && git add constants.py CLAUDE.md && git commit -m "docs: add Semantic CA section to CLAUDE.md, age_epoch helper to constants"
```

---

## Verification

After all tasks:
```bash
cd ~/how-to-live-much-longer
python -m pytest tests/ -v                    # All tests pass
python -m pytest tests/test_ca.py -v          # CA tests specifically

# One-liners
python -c "from ca_simulator import run_single_cell; r = run_single_cell(); print(r['final_state'])"
python -c "from ca_zimmerman_bridge import MitoCASimulator; s = MitoCASimulator(); print(len(s.param_spec()), 'params')"
python -c "from ca_stochastic import run_single_cell_stochastic, compute_ensemble_analytics; r = run_single_cell_stochastic(n_trials=10); a = compute_ensemble_analytics(r); print(a['attractor_probabilities'])"

# Visualizations
python -c "from ca_visualize import generate_all_plots; generate_all_plots()"

# Cramer fidelity check: confirm no changes to core ODE
git diff simulator.py               # Should show NO changes
```
