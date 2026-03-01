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
        "thresholds": [0.3, 0.56],
        "labels": ["depleted", "reduced", "adequate"],
        "centers": [0.243, 0.27, 0.884],
        "unit": "normalized copies",
        "source": "C2 copy homeostasis",
    },
    "N_deletion": {
        "index": 1,
        "thresholds": [0.1, 0.3, 0.5],
        "labels": ["minimal", "growing", "approaching_cliff", "past_cliff"],
        "centers": [0.123, 0.203, 0.416, 0.37],
        "unit": "deletion het fraction",
        "source": "HETEROPLASMY_CLIFF=0.50, Cramer Appendix 2",
    },
    "ATP": {
        "index": 2,
        "thresholds": [0.2, 0.5, 0.79],
        "labels": ["collapsed", "crisis", "compromised", "healthy"],
        "centers": [0.63, 0.632, 0.638, 0.886],
        "unit": "MU/day",
        "source": "ATP_CRISIS_FRACTION=0.5, Cramer Ch. VIII.A Table 3",
    },
    "ROS": {
        "index": 3,
        "thresholds": [0.1, 0.25],
        "labels": ["basal", "elevated", "pathological"],
        "centers": [0.05, 0.211, 0.309],
        "unit": "normalized",
        "source": "BASELINE_ROS=0.1, Cramer Ch. II.H",
    },
    "NAD": {
        "index": 4,
        "thresholds": [0.3, 0.7],
        "labels": ["depleted", "declining", "robust"],
        "centers": [0.293, 0.537, 0.924],
        "unit": "normalized",
        "source": "NAD_DECLINE_RATE=0.01/yr, Cramer Ch. VI.A.3",
    },
    "Senescent_fraction": {
        "index": 5,
        "thresholds": [0.1, 0.4],
        "labels": ["minimal", "emerging", "severe"],
        "centers": [0.085, 0.228, 0.319],
        "unit": "fraction",
        "source": "SENESCENCE_RATE=0.005/yr, Cramer Ch. VII.A",
    },
    "Membrane_potential": {
        "index": 6,
        "thresholds": [0.3, 0.5],
        "labels": ["collapsed", "impaired", "intact"],
        "centers": [0.292, 0.454, 0.583],
        "unit": "normalized ΔΨ",
        "source": "MITOPHAGY_ATP_MIDPOINT=0.6, Cramer Ch. VI.B",
    },
    "N_point": {
        "index": 7,
        "thresholds": [0.1, 0.3],
        "labels": ["low", "moderate", "high"],
        "centers": [0.05, 0.255, 0.441],
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


def _in_bin_exemplar(thresholds: list[float], bin_idx: int) -> float:
    """Pick a value guaranteed to classify into bin_idx."""
    if bin_idx == 0:
        first = thresholds[0]
        return first * 0.5 if first > 0.0 else first - 1.0

    if bin_idx == len(thresholds):
        prev = thresholds[-1]
        return (prev + 1.0) * 0.5 if prev < 1.0 else prev + max(1e-3, 0.1 * abs(prev))

    lower = thresholds[bin_idx - 1]
    upper = thresholds[bin_idx]
    return 0.5 * (lower + upper)


def discretize_state(continuous_state: np.ndarray) -> dict[str, str]:
    """Convert an 8D continuous state vector to named clinical bins.
    
    For N_deletion and N_point, computes heteroplasmy fraction (N_var / total).
    Other variables use raw values.
    """
    # Extract copy counts
    n_healthy = float(continuous_state[BIN_SCHEMA["N_healthy"]["index"]])
    n_deletion = float(continuous_state[BIN_SCHEMA["N_deletion"]["index"]])
    n_point = float(continuous_state[BIN_SCHEMA["N_point"]["index"]])
    
    result = {}
    for var_name in CA_VAR_ORDER:
        schema = BIN_SCHEMA[var_name]
        idx = schema["index"]
        if var_name == "N_deletion":
            val = n_deletion / (n_healthy + n_deletion + n_point) if (n_healthy + n_deletion + n_point) > 1e-12 else 1.0
        elif var_name == "N_point":
            val = n_point / (n_healthy + n_deletion + n_point) if (n_healthy + n_deletion + n_point) > 1e-12 else 1.0
        else:
            val = float(continuous_state[idx])
        result[var_name] = _classify(val, schema["thresholds"], schema["labels"])
    return result


def continuous_exemplar(discrete_state: dict[str, str]) -> np.ndarray:
    """Convert a discrete bin assignment back to an 8D continuous exemplar."""
    state = np.zeros(N_STATES, dtype=np.float64)
    for var_name, label in discrete_state.items():
        # N_deletion/N_point are fraction-binned in discretize_state(), so defer
        # their raw copy-count reconstruction until we can solve jointly.
        if var_name in ("N_deletion", "N_point"):
            continue
        schema = BIN_SCHEMA[var_name]
        idx = schema["labels"].index(label)
        state[schema["index"]] = _in_bin_exemplar(schema["thresholds"], idx)

    if "N_deletion" in discrete_state and "N_point" in discrete_state:
        n_healthy_schema = BIN_SCHEMA["N_healthy"]
        n_deletion_schema = BIN_SCHEMA["N_deletion"]
        n_point_schema = BIN_SCHEMA["N_point"]

        n_healthy = state[n_healthy_schema["index"]]
        if "N_healthy" not in discrete_state:
            n_healthy = n_healthy_schema["centers"][-1]
            state[n_healthy_schema["index"]] = n_healthy

        n_del_label = discrete_state["N_deletion"]
        n_pt_label = discrete_state["N_point"]
        f_del = _in_bin_exemplar(
            n_deletion_schema["thresholds"], n_deletion_schema["labels"].index(n_del_label)
        )
        f_pt = _in_bin_exemplar(
            n_point_schema["thresholds"], n_point_schema["labels"].index(n_pt_label)
        )

        denom = 1.0 - (f_del + f_pt)
        if denom > 1e-12:
            total = n_healthy / denom
            state[n_deletion_schema["index"]] = f_del * total
            state[n_point_schema["index"]] = f_pt * total
        else:
            state[n_deletion_schema["index"]] = f_del
            state[n_point_schema["index"]] = f_pt
    else:
        for var_name in ("N_deletion", "N_point"):
            if var_name in discrete_state:
                schema = BIN_SCHEMA[var_name]
                idx = schema["labels"].index(discrete_state[var_name])
                state[schema["index"]] = _in_bin_exemplar(schema["thresholds"], idx)
    return state


def bin_index(var_name: str, label: str) -> int:
    """Return the integer index of a bin label within its variable's bins."""
    return BIN_SCHEMA[var_name]["labels"].index(label)


def bin_count(var_name: str) -> int:
    """Return the number of bins for a given variable."""
    return len(BIN_SCHEMA[var_name]["labels"])
