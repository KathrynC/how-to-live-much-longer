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
