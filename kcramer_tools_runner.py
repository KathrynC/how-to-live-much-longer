#!/usr/bin/env python3
"""CLI runner for K-Cramer Toolkit integration workflows.

Provides a reproducible command-line interface around `kcramer_bridge.py`:
    - Full resilience analysis across protocol banks and stress scenarios
    - Vulnerability analysis for one protocol
    - Scenario-conditioned comparison of any supported scalar output

Usage:
    python kcramer_tools_runner.py --mode resilience
    python kcramer_tools_runner.py --mode vulnerability --protocol moderate
    python kcramer_tools_runner.py --mode compare --output-key final_atp
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from analytics import NumpyEncoder
from constants import DEFAULT_PATIENT
from kcramer_bridge import (
    MitoSimulator,
    PROTOCOLS,
    run_resilience_analysis,
    run_scenario_comparison,
    run_vulnerability_analysis,
)

ARTIFACTS_DIR = Path("artifacts")

# Profile seeds chosen to match existing experiment conventions in this repo.
PATIENT_PROFILES = {
    "default": DEFAULT_PATIENT,
    "near_cliff_80": {
        "baseline_age": 80.0, "baseline_heteroplasmy": 0.65,
        "baseline_nad_level": 0.4, "genetic_vulnerability": 1.0,
        "metabolic_demand": 1.0, "inflammation_level": 0.5,
    },
    "post_chemo_55": {
        "baseline_age": 55.0, "baseline_heteroplasmy": 0.40,
        "baseline_nad_level": 0.4, "genetic_vulnerability": 1.25,
        "metabolic_demand": 1.0, "inflammation_level": 0.5,
    },
}


def _scalar_analysis(sim: MitoSimulator, output_key: str = "final_atp") -> float:
    """Scalar function for `run_scenario_comparison`."""
    result = sim.run({**PROTOCOLS["moderate"], **DEFAULT_PATIENT})
    return float(result[output_key])


def run_mode(
    mode: str,
    patient_profile: str,
    output_key: str,
    protocol: str,
) -> dict:
    """Execute one K-Cramer toolkit mode and return artifact payload."""
    patient = PATIENT_PROFILES.get(patient_profile, DEFAULT_PATIENT)
    sim = MitoSimulator()

    # Cramer scenario banks perturb patient dimensions (age, NAD, inflammation,
    # vulnerability, etc.), so protocols must include patient keys in full 12D
    # mode. This keeps scenario effects active while fixing a patient baseline.
    merged_protocols = {
        name: {**iv, **patient} for name, iv in PROTOCOLS.items()
    }

    if mode == "resilience":
        result = run_resilience_analysis(
            sim=sim,
            protocols=merged_protocols,
            output_key=output_key,
        )
    elif mode == "vulnerability":
        chosen = merged_protocols.get(protocol)
        if chosen is None:
            raise ValueError(
                f"Unknown protocol '{protocol}'. "
                f"Choose from: {sorted(PROTOCOLS.keys())}"
            )
        result = run_vulnerability_analysis(
            sim=sim,
            protocol=chosen,
            output_key=output_key,
        )
    elif mode == "compare":
        def _profiled_scalar(local_sim: MitoSimulator, output_key: str = output_key) -> float:
            return float(local_sim.run({**PROTOCOLS["moderate"], **patient})[output_key])

        result = run_scenario_comparison(
            _profiled_scalar,
            sim=sim,
            scenarios=None,
            extract=lambda x: float(x),
            output_key=output_key,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "mode": mode,
        "patient_profile": patient_profile,
        "output_key": output_key,
        "protocol": protocol,
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Cramer Toolkit integration workflows."
    )
    parser.add_argument(
        "--mode",
        choices=["resilience", "vulnerability", "compare"],
        default="resilience",
        help="Workflow mode (default: resilience).",
    )
    parser.add_argument(
        "--patient-profile",
        choices=sorted(PATIENT_PROFILES.keys()),
        default="default",
        help="Patient profile seed (default: default).",
    )
    parser.add_argument(
        "--output-key",
        default="final_atp",
        help="Output metric key (default: final_atp).",
    )
    parser.add_argument(
        "--protocol",
        choices=sorted(PROTOCOLS.keys()),
        default="moderate",
        help="Protocol for vulnerability mode (default: moderate).",
    )
    args = parser.parse_args()

    payload = run_mode(
        mode=args.mode,
        patient_profile=args.patient_profile,
        output_key=args.output_key,
        protocol=args.protocol,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    out_path = ARTIFACTS_DIR / (
        f"kcramer_tools_{args.mode}_{args.patient_profile}_{stamp}.json"
    )
    out_path.write_text(json.dumps(payload, indent=2, cls=NumpyEncoder))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
