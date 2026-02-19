"""Scenario analysis — milestone extraction and cross-scenario comparison.

Extracts key milestones (final values, threshold-crossing ages) from
scenario results and compares across scenarios.
"""
from __future__ import annotations

import numpy as np


def extract_milestones(result: dict) -> dict:
    """Extract key milestones from a scenario result.

    Args:
        result: Dict returned by scenario_runner.run_scenario(), containing
            'core', 'downstream', 'scenario_name', 'scenario'.

    Returns:
        Dict with milestone values:
            - scenario_name: str
            - final_heteroplasmy, final_atp, final_memory_index,
              final_amyloid, final_tau, final_resilience: float
            - het_below_50_age: age when heteroplasmy first drops below 50%
            - atp_above_08_age: age when ATP first exceeds 0.8
            - dementia_age: age when memory_index first drops below 0.5
            - amyloid_pathology_age: age when amyloid_burden first exceeds 1.0
            (None if milestone never reached within simulation horizon)
    """
    core = result['core']
    downstream = result['downstream']
    time_arr = core['time']
    het = core['heteroplasmy']
    atp = core['states'][:, 2]

    milestones = {
        'scenario_name': result['scenario_name'],
        'final_heteroplasmy': float(het[-1]),
        'final_atp': float(atp[-1]),
        'final_memory_index': downstream[-1]['memory_index'],
        'final_amyloid': downstream[-1]['amyloid_burden'],
        'final_tau': downstream[-1]['tau_pathology'],
        'final_resilience': downstream[-1]['resilience'],
    }

    base_age = result['scenario'].patient_params.get('baseline_age', 70.0)

    # Het below 50%
    het_below_50 = np.where(het < 0.50)[0]
    milestones['het_below_50_age'] = (
        float(base_age + time_arr[het_below_50[0]]) if len(het_below_50) > 0 else None
    )

    # ATP above 0.8
    atp_above_08 = np.where(atp > 0.8)[0]
    milestones['atp_above_08_age'] = (
        float(base_age + time_arr[atp_above_08[0]]) if len(atp_above_08) > 0 else None
    )

    # Memory below 0.5 (dementia threshold)
    memory_arr = [d['memory_index'] for d in downstream]
    mem_below_05 = [i for i, m in enumerate(memory_arr) if m < 0.5]
    milestones['dementia_age'] = (
        float(base_age + time_arr[mem_below_05[0]]) if mem_below_05 else None
    )

    # Amyloid above 1.0 (pathological threshold)
    amyloid_arr = [d['amyloid_burden'] for d in downstream]
    amy_above_1 = [i for i, a in enumerate(amyloid_arr) if a > 1.0]
    milestones['amyloid_pathology_age'] = (
        float(base_age + time_arr[amy_above_1[0]]) if amy_above_1 else None
    )

    return milestones


def compare_scenarios(results: list[dict]) -> list[dict]:
    """Compare milestones across multiple scenario results.

    Args:
        results: List of result dicts from scenario_runner.run_scenarios().

    Returns:
        List of milestone dicts (one per scenario).
    """
    return [extract_milestones(r) for r in results]


def summary_table(
    results: list[dict],
    ages: list[float] | None = None,
) -> list[dict]:
    """Pivot of metric values at specified ages for each scenario.

    For each scenario and each target age, finds the closest simulation
    timestep and records heteroplasmy, ATP, and memory_index.

    Args:
        results: List of result dicts from scenario_runner.run_scenarios().
        ages: Target ages to evaluate at. Default: [70, 80, 90, 95].

    Returns:
        List of dicts, one per scenario, each containing:
            'scenario_name': str
            'metrics': dict mapping age -> {'heteroplasmy': float,
                                            'atp': float,
                                            'memory_index': float}
    """
    if ages is None:
        ages = [70, 80, 90, 95]

    table = []
    for r in results:
        core = r['core']
        downstream = r['downstream']
        time_arr = core['time']
        het = core['heteroplasmy']
        atp = core['states'][:, 2]
        base_age = r['scenario'].patient_params.get('baseline_age', 70.0)

        age_arr = base_age + time_arr
        metrics_by_age = {}
        for target_age in ages:
            if target_age < age_arr[0] or target_age > age_arr[-1]:
                metrics_by_age[target_age] = None
                continue
            idx = int(np.argmin(np.abs(age_arr - target_age)))
            metrics_by_age[target_age] = {
                'heteroplasmy': float(het[idx]),
                'atp': float(atp[idx]),
                'memory_index': downstream[idx]['memory_index'],
            }

        table.append({
            'scenario_name': r['scenario_name'],
            'metrics': metrics_by_age,
        })

    return table
