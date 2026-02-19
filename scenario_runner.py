"""Scenario runner — execute scenarios through resolver + simulator + downstream.

Runs each Scenario through the full pipeline:
1. Build ParameterResolver from patient + intervention dicts
2. Run core ODE simulator with resolver
3. Compute downstream chain (neuroplasticity, Alzheimer's pathology)
4. Return bundled result dict
"""
from __future__ import annotations

from scenario_definitions import Scenario


# ── APOE genotype int→str mapping for downstream_chain ─────────────────────
# ParameterResolver/genetics_module uses int (0/1/2), but downstream_chain
# looks up string keys in GENOTYPE_MULTIPLIERS.
_APOE_INT_TO_STR = {
    1: 'apoe4_het',
    2: 'apoe4_hom',
}


def run_scenario(scenario: Scenario, years: float | None = None) -> dict:
    """Run a single scenario through the full pipeline.

    Args:
        scenario: A Scenario instance with patient_params and interventions.
        years: Override simulation duration (default: scenario.duration_years).

    Returns:
        Dict with keys:
            'core': dict from simulator.simulate() (time, states, heteroplasmy, ...)
            'downstream': list[dict] from downstream_chain.compute_downstream()
            'scenario_name': str
            'scenario': the Scenario object
    """
    from parameter_resolver import ParameterResolver
    from simulator import simulate
    from downstream_chain import compute_downstream

    years = years if years is not None else scenario.duration_years
    intervention_dict = scenario.interventions.to_dict()

    pr = ParameterResolver(
        patient_expanded=scenario.patient_params,
        intervention_expanded=intervention_dict,
        duration_years=years,
    )

    core = simulate(resolver=pr, sim_years=years)

    # Build the patient_expanded dict for downstream_chain, mapping apoe int→str
    patient_for_downstream = dict(scenario.patient_params)
    apoe_int = patient_for_downstream.get('apoe_genotype')
    if isinstance(apoe_int, int) and apoe_int in _APOE_INT_TO_STR:
        patient_for_downstream['apoe_genotype'] = _APOE_INT_TO_STR[apoe_int]

    downstream = compute_downstream(core, patient_for_downstream)

    return {
        'core': core,
        'downstream': downstream,
        'scenario_name': scenario.name,
        'scenario': scenario,
    }


def run_scenarios(scenarios: list[Scenario], years: float | None = None) -> list[dict]:
    """Run multiple scenarios and return list of results.

    Args:
        scenarios: List of Scenario instances.
        years: Override simulation duration for all scenarios.

    Returns:
        List of result dicts (one per scenario).
    """
    return [run_scenario(s, years) for s in scenarios]
