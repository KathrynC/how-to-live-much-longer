"""Tests for the mitochondrial semantic cellular automaton."""
import numpy as np
import pytest

from ca_schema import (
    BIN_SCHEMA, discretize_state, continuous_exemplar,
    bin_index, bin_count, CA_VAR_ORDER, CA_N_VARS,
)
from ca_rules import (
    RULE_TABLE, apply_rules, get_applicable_rules,
    _evaluate_context, _evaluate_inputs, _apply_direction,
    save_rules, load_rules,
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


# ── Rule tests ────────────────────────────────────────────────────────────


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

    def test_age_epoch_list(self):
        spec = {"age_epoch": ["transition", "old"]}
        assert _evaluate_context(spec, {"age_epoch": "transition"})
        assert _evaluate_context(spec, {"age_epoch": "old"})
        assert not _evaluate_context(spec, {"age_epoch": "young"})

    def test_intervention_level(self):
        assert _evaluate_context({"rapamycin": "high"}, {"rapamycin": "high"})
        assert not _evaluate_context({"rapamycin": "high"}, {"rapamycin": "none"})

    def test_intervention_level_threshold(self):
        spec = {"exercise": "moderate+"}
        assert _evaluate_context(spec, {"exercise": "moderate"})
        assert _evaluate_context(spec, {"exercise": "high"})
        assert not _evaluate_context(spec, {"exercise": "low"})
        assert not _evaluate_context(spec, {"exercise": "none"})

    def test_defaults_to_none(self):
        """Missing intervention defaults to 'none'."""
        assert not _evaluate_context({"rapamycin": "high"}, {})


class TestEvaluateInputs:
    def test_exact_match(self):
        state = {"N_deletion": "growing", "ATP": "healthy"}
        assert _evaluate_inputs({"N_deletion": "growing"}, state)
        assert not _evaluate_inputs({"N_deletion": "minimal"}, state)

    def test_plus_suffix(self):
        state = {"N_deletion": "approaching_cliff"}
        # growing+ means index >= 1; approaching_cliff is index 2
        assert _evaluate_inputs({"N_deletion": "growing+"}, state)

    def test_plus_suffix_below(self):
        state = {"N_deletion": "minimal"}
        # growing+ means index >= 1; minimal is index 0
        assert not _evaluate_inputs({"N_deletion": "growing+"}, state)

    def test_minus_suffix(self):
        state = {"ATP": "crisis"}
        # crisis- means index <= 1; crisis is index 1
        assert _evaluate_inputs({"ATP": "crisis-"}, state)

    def test_minus_suffix_above(self):
        state = {"ATP": "compromised"}
        # crisis- means index <= 1; compromised is index 2
        assert not _evaluate_inputs({"ATP": "crisis-"}, state)

    def test_missing_var(self):
        assert not _evaluate_inputs({"N_deletion": "growing"}, {})


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
        assert _apply_direction("healthy", "-2", "ATP") == "crisis"


class TestSaveLoadRules:
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "rules.json")
        save_rules(path)
        loaded = load_rules(path)
        assert len(loaded) == len(RULE_TABLE)
        assert loaded[0]["name"] == RULE_TABLE[0]["name"]
