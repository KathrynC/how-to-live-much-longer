"""Grief-derived stress scenarios for cramer-toolkit resilience analysis.

Defines 16 grief stress scenarios (8 clinical seeds x 2 intervention levels)
and named grief intervention protocols, compatible with the cramer-toolkit's
scenario-based analysis framework.

Usage:
    from grief_mito_scenarios import (
        GRIEF_STRESS_SCENARIOS, GRIEF_PROTOCOLS,
        grief_scenario_disturbances,
    )
    from disturbances import simulate_with_disturbances

    # Run a specific grief scenario through the mito simulator
    for scenario in GRIEF_STRESS_SCENARIOS:
        result = simulate_with_disturbances(disturbances=[scenario["disturbance"]])
        print(f"{scenario['name']}: het={result['heteroplasmy'][-1]:.4f}")

Requires:
    grief-simulator (at ~/grief-simulator)
"""
from __future__ import annotations

from grief_bridge import (
    GriefDisturbance,
    GRIEF_CLINICAL_SEEDS,
    GRIEF_DEFAULT_INTERVENTION,
)

# -- Grief intervention protocols ----------------------------------------------
# Named profiles matching O'Connor's behavioral recommendations.

GRIEF_PROTOCOLS: dict[str, dict[str, float]] = {
    "no_grief_support": dict(GRIEF_DEFAULT_INTERVENTION),
    "minimal_support": {
        "slp_int": 0.3, "act_int": 0.2, "nut_int": 0.2,
        "alc_int": 0.5, "br_int": 0.0, "med_int": 0.0, "soc_int": 0.3,
    },
    "moderate_support": {
        "slp_int": 0.5, "act_int": 0.5, "nut_int": 0.5,
        "alc_int": 0.7, "br_int": 0.3, "med_int": 0.3, "soc_int": 0.5,
    },
    "full_grief_support": {
        "slp_int": 0.8, "act_int": 0.7, "nut_int": 0.6,
        "alc_int": 0.8, "br_int": 0.5, "med_int": 0.5, "soc_int": 0.7,
    },
}


# -- Build scenario bank -------------------------------------------------------

def _build_scenarios() -> list[dict]:
    """Build all grief stress scenarios from clinical seeds."""
    scenarios = []
    for seed in GRIEF_CLINICAL_SEEDS:
        # Without intervention
        scenarios.append({
            "name": f"{seed['name']}_no_support",
            "description": f"{seed['description']} -- no grief support",
            "seed": seed["name"],
            "intervention": "no_grief_support",
            "disturbance": GriefDisturbance(
                grief_patient=seed["patient"],
                grief_intervention=None,
                label=f"grief_{seed['name']}_no_support",
            ),
        })
        # With full support
        scenarios.append({
            "name": f"{seed['name']}_full_support",
            "description": f"{seed['description']} -- full grief support",
            "seed": seed["name"],
            "intervention": "full_grief_support",
            "disturbance": GriefDisturbance(
                grief_patient=seed["patient"],
                grief_intervention=GRIEF_PROTOCOLS["full_grief_support"],
                label=f"grief_{seed['name']}_full_support",
            ),
        })
    return scenarios


GRIEF_STRESS_SCENARIOS: list[dict] = _build_scenarios()


def grief_scenario_disturbances(seed_name: str) -> list[GriefDisturbance]:
    """Get the with/without-support disturbance pair for a clinical seed."""
    return [
        s["disturbance"] for s in GRIEF_STRESS_SCENARIOS
        if s["seed"] == seed_name
    ]
