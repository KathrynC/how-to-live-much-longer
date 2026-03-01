"""Tests for CA image schema detectors."""
from __future__ import annotations

import warnings

from ca_image_schemas import CAImageSchemaDetector, CATrajectory
from ca_schema import BIN_SCHEMA


def _stable_state() -> dict[str, str]:
    return {
        "N_healthy": BIN_SCHEMA["N_healthy"]["labels"][2],
        "N_deletion": BIN_SCHEMA["N_deletion"]["labels"][0],
        "ATP": BIN_SCHEMA["ATP"]["labels"][3],
        "ROS": BIN_SCHEMA["ROS"]["labels"][0],
        "NAD": BIN_SCHEMA["NAD"]["labels"][2],
        "Senescent_fraction": BIN_SCHEMA["Senescent_fraction"]["labels"][0],
        "Membrane_potential": BIN_SCHEMA["Membrane_potential"]["labels"][2],
        "N_point": BIN_SCHEMA["N_point"]["labels"][0],
    }


def test_detect_balance_degenerate_traj_has_no_runtime_warning():
    detector = CAImageSchemaDetector()
    traj = CATrajectory([_stable_state() for _ in range(12)], timestep_years=0.25)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        schemas = detector.detect_all(traj)

    balance = schemas["BALANCE"].metrics
    assert "restoration_rate" in balance
    assert 0.0 <= balance["restoration_rate"] <= 1.0
