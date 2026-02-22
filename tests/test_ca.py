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
