"""Cramer Toolkit bridge for the mitochondrial aging simulator.

Creates biologically-meaningful stress scenarios using the cramer-toolkit,
targeting the MitoSimulator adapter from zimmerman_bridge.py.

Usage:
    from kcramer_bridge import (
        MitoSimulator,
        INFLAMMATION_SCENARIOS, NAD_SCENARIOS, VULNERABILITY_SCENARIOS,
        AGING_SCENARIOS, COMBINED_SCENARIOS, ALL_STRESS_SCENARIOS,
        run_resilience_analysis,
    )

    sim = MitoSimulator()
    report = run_resilience_analysis(sim)

Requires:
    cramer-toolkit (at ~/cramer-toolkit)
    zimmerman-toolkit (at ~/zimmerman-toolkit)
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────

PROJECT = Path(__file__).resolve().parent
CRAMER_PATH = PROJECT.parent / "cramer-toolkit"
ZIMMERMAN_PATH = PROJECT.parent / "zimmerman-toolkit"

for p in (CRAMER_PATH, ZIMMERMAN_PATH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Project imports
from zimmerman_bridge import MitoSimulator  # noqa: E402

# Cramer toolkit imports
from cramer import (  # noqa: E402
    Scenario,
    ScenarioSet,
    scale_param,
    shift_param,
    set_param,
    compose,
    run_scenarios,
    run_protocol_suite,
    robustness_score,
    resilience_summary,
    vulnerability_profile,
    scenario_regret,
    ScenarioSimulator,
    scenario_aware,
    scenario_compare,
    stress_test_suite,
    boundary_scenarios,
)


# ── Biological stress scenarios ──────────────────────────────────────────────

# --- Inflammation scenarios ---
# Cramer Ch. VII.A pp.89-92: chronic inflammation drives senescence and
# accelerates mtDNA damage. inflammation_level range is [0.0, 1.0].

INFLAMMATION_SCENARIOS = ScenarioSet(
    "inflammation",
    "Chronic inflammation stress conditions",
    scenarios=[
        scale_param("inflammation_level", 1.5, name="mild_inflammaging"),
        scale_param("inflammation_level", 2.0, name="moderate_inflammaging"),
        set_param("inflammation_level", 0.75, name="severe_inflammation"),
        set_param("inflammation_level", 1.0, name="inflammatory_storm"),
    ],
)

# --- NAD+ depletion scenarios ---
# Cramer Ch. VI.A.3 pp.72-73: NAD+ declines with age. CD38 enzyme destroys
# NMN/NR supplements (p.73). baseline_nad_level range is [0.2, 1.0].

NAD_SCENARIOS = ScenarioSet(
    "nad_depletion",
    "NAD+ availability stress conditions",
    scenarios=[
        scale_param("baseline_nad_level", 0.8, name="mild_nad_decline"),
        scale_param("baseline_nad_level", 0.6, name="moderate_nad_decline"),
        scale_param("baseline_nad_level", 0.4, name="severe_nad_depletion"),
        set_param("baseline_nad_level", 0.2, name="critical_nad_crisis"),
    ],
)

# --- Genetic vulnerability scenarios ---
# Cramer Ch. V.J p.65: haplogroup-dependent susceptibility to mtDNA damage.
# genetic_vulnerability range is [0.5, 2.0], multiplier on damage rate.

VULNERABILITY_SCENARIOS = ScenarioSet(
    "genetic_vulnerability",
    "Haplogroup-dependent damage susceptibility",
    scenarios=[
        scale_param("genetic_vulnerability", 1.25, name="elevated_vulnerability"),
        scale_param("genetic_vulnerability", 1.5, name="high_vulnerability"),
        set_param("genetic_vulnerability", 2.0, name="max_vulnerability"),
    ],
)

# --- Metabolic demand scenarios ---
# Cramer Ch. V.J p.65: brain/cardiac tissue has higher demand.
# metabolic_demand range is [0.5, 2.0].

DEMAND_SCENARIOS = ScenarioSet(
    "metabolic_demand",
    "Tissue-specific metabolic demand stress",
    scenarios=[
        set_param("metabolic_demand", 1.5, name="muscle_demand"),
        set_param("metabolic_demand", 1.8, name="cardiac_demand"),
        set_param("metabolic_demand", 2.0, name="brain_demand"),
    ],
)

# --- Aging acceleration scenarios ---
# Shift baseline_age forward or scale baseline_heteroplasmy up to simulate
# accelerated aging or pre-existing damage.

AGING_SCENARIOS = ScenarioSet(
    "aging_acceleration",
    "Accelerated aging and pre-existing damage",
    scenarios=[
        shift_param("baseline_age", 10.0, name="decade_older"),
        shift_param("baseline_age", 20.0, name="two_decades_older"),
        scale_param("baseline_heteroplasmy", 1.5, name="elevated_damage"),
        scale_param("baseline_heteroplasmy", 2.0, name="high_damage"),
        set_param("baseline_heteroplasmy", 0.65, name="near_cliff"),
        set_param("baseline_heteroplasmy", 0.75, name="past_cliff"),
    ],
)

# --- Combined stress scenarios ---
# Real patients often face multiple simultaneous stressors.
# Cramer Ch. VIII.F p.103: senescent cells + inflammation + NAD decline.

COMBINED_SCENARIOS = ScenarioSet(
    "combined_stress",
    "Multi-factor biological stress conditions",
    scenarios=[
        compose(
            scale_param("inflammation_level", 2.0),
            scale_param("baseline_nad_level", 0.6),
            name="inflamed_nad_depleted",
        ),
        compose(
            set_param("baseline_heteroplasmy", 0.65),
            scale_param("genetic_vulnerability", 1.5),
            name="near_cliff_vulnerable",
        ),
        compose(
            shift_param("baseline_age", 15.0),
            scale_param("inflammation_level", 1.5),
            scale_param("baseline_nad_level", 0.7),
            name="accelerated_aging_syndrome",
        ),
        compose(
            set_param("metabolic_demand", 2.0),
            scale_param("baseline_nad_level", 0.5),
            scale_param("inflammation_level", 2.0),
            name="brain_energy_crisis",
        ),
        compose(
            set_param("baseline_heteroplasmy", 0.75),
            set_param("genetic_vulnerability", 2.0),
            set_param("metabolic_demand", 2.0),
            set_param("inflammation_level", 0.8),
            name="worst_case_patient",
        ),
    ],
)

# --- Aggregated scenario bank ---

ALL_STRESS_SCENARIOS = (
    INFLAMMATION_SCENARIOS
    + NAD_SCENARIOS
    + VULNERABILITY_SCENARIOS
    + DEMAND_SCENARIOS
    + AGING_SCENARIOS
    + COMBINED_SCENARIOS
)


# ── Reference intervention protocols ────────────────────────────────────────

PROTOCOLS = {
    "no_treatment": {
        "rapamycin_dose": 0.0,
        "nad_supplement": 0.0,
        "senolytic_dose": 0.0,
        "yamanaka_intensity": 0.0,
        "transplant_rate": 0.0,
        "exercise_level": 0.0,
    },
    "conservative": {
        "rapamycin_dose": 0.25,
        "nad_supplement": 0.25,
        "senolytic_dose": 0.1,
        "yamanaka_intensity": 0.0,
        "transplant_rate": 0.0,
        "exercise_level": 0.5,
    },
    "moderate": {
        "rapamycin_dose": 0.5,
        "nad_supplement": 0.5,
        "senolytic_dose": 0.25,
        "yamanaka_intensity": 0.0,
        "transplant_rate": 0.25,
        "exercise_level": 0.5,
    },
    "aggressive": {
        "rapamycin_dose": 0.75,
        "nad_supplement": 0.75,
        "senolytic_dose": 0.5,
        "yamanaka_intensity": 0.25,
        "transplant_rate": 0.5,
        "exercise_level": 0.75,
    },
    "transplant_focused": {
        "rapamycin_dose": 0.5,
        "nad_supplement": 0.5,
        "senolytic_dose": 0.25,
        "yamanaka_intensity": 0.0,
        "transplant_rate": 1.0,
        "exercise_level": 0.5,
    },
}


# ── Convenience analysis functions ──────────────────────────────────────────


def run_resilience_analysis(
    sim: MitoSimulator | None = None,
    protocols: dict | None = None,
    scenarios: ScenarioSet | list | None = None,
    output_key: str = "final_atp",
) -> dict:
    """Run a full resilience analysis on the mitochondrial simulator.

    Args:
        sim: MitoSimulator instance (defaults to full 12D mode).
        protocols: Dict of protocol_name → param_dict.
            Defaults to PROTOCOLS.
        scenarios: Scenarios to test. Defaults to ALL_STRESS_SCENARIOS.
        output_key: Output metric for scoring. Defaults to "final_atp".

    Returns:
        Full resilience summary from cramer.resilience_summary().
    """
    if sim is None:
        sim = MitoSimulator()
    if protocols is None:
        protocols = PROTOCOLS
    if scenarios is None:
        scenarios = ALL_STRESS_SCENARIOS

    results = run_protocol_suite(sim, protocols, scenarios)
    return resilience_summary(results, output_key)


def run_vulnerability_analysis(
    sim: MitoSimulator | None = None,
    protocol: dict | None = None,
    scenarios: ScenarioSet | list | None = None,
    output_key: str = "final_atp",
) -> list[dict]:
    """Identify which stress scenarios most damage a protocol.

    Args:
        sim: MitoSimulator instance.
        protocol: Intervention params. Defaults to "moderate" protocol.
        scenarios: Scenarios to test. Defaults to ALL_STRESS_SCENARIOS.
        output_key: Output metric for comparison.

    Returns:
        Sorted list of {scenario, impact, ...}, worst first.
    """
    if sim is None:
        sim = MitoSimulator()
    if protocol is None:
        protocol = PROTOCOLS["moderate"]
    if scenarios is None:
        scenarios = ALL_STRESS_SCENARIOS

    results = run_scenarios(sim, protocol, scenarios)
    return vulnerability_profile(results, output_key=output_key)


def run_scenario_comparison(
    analysis_fn,
    sim: MitoSimulator | None = None,
    scenarios: ScenarioSet | list | None = None,
    extract=None,
    **kwargs,
) -> dict:
    """Run any analysis function under multiple stress scenarios.

    Wraps the simulator in ScenarioSimulator for each scenario, so
    any Zimmerman or other analysis tool becomes scenario-conditioned.

    Args:
        analysis_fn: Function with signature fn(sim, **kwargs).
        sim: MitoSimulator instance.
        scenarios: Scenarios to apply.
        extract: Optional scalar extractor for delta computation.
        **kwargs: Passed to analysis_fn.

    Returns:
        Dict of {scenario_name: {result, value, baseline_value, delta}}.
    """
    if sim is None:
        sim = MitoSimulator()
    if scenarios is None:
        scenarios = ALL_STRESS_SCENARIOS

    return scenario_compare(analysis_fn, sim, scenarios, extract=extract, **kwargs)
