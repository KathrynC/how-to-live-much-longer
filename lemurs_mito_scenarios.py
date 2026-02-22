"""LEMURS-derived stress scenarios for kcramer resilience analysis.

Defines 14 college stress scenarios across 5 banks and named LEMURS
intervention protocols, compatible with the kcramer scenario-based
analysis framework.

Usage:
    from lemurs_mito_scenarios import (
        LEMURS_STRESS_SCENARIOS, LEMURS_PROTOCOLS,
        lemurs_scenario_disturbances,
    )
    from disturbances import simulate_with_disturbances

    # Run a specific LEMURS scenario through the mito simulator
    for scenario in LEMURS_STRESS_SCENARIOS:
        result = simulate_with_disturbances(disturbances=[scenario["disturbance"]])
        print(f"{scenario['name']}: het={result['heteroplasmy'][-1]:.4f}")

Requires:
    lemurs-simulator (at ~/lemurs-simulator)
"""
from __future__ import annotations

from lemurs_bridge import (
    LEMURSDisturbance,
    LEMURS_DEFAULT_INTERVENTION,
    LEMURS_DEFAULT_PATIENT,
)

# -- LEMURS intervention protocols --------------------------------------------
# Named profiles representing different levels of college well-being support.

LEMURS_PROTOCOLS: dict[str, dict[str, float]] = {
    "no_treatment": dict(LEMURS_DEFAULT_INTERVENTION),
    "nature_intervention": {
        **LEMURS_DEFAULT_INTERVENTION,
        "nature_rx": 0.8,
    },
    "exercise_intervention": {
        **LEMURS_DEFAULT_INTERVENTION,
        "exercise_rx": 0.8,
    },
    "full_support": {
        "nature_rx": 0.8,
        "exercise_rx": 0.8,
        "therapy_rx": 0.5,
        "sleep_hygiene": 0.8,
        "caffeine_reduction": 0.5,
        "academic_load": 0.3,
    },
    "academic_relief": {
        **LEMURS_DEFAULT_INTERVENTION,
        "academic_load": 0.2,
    },
}


# -- Scenario banks -----------------------------------------------------------

ACADEMIC_STRESS_SCENARIOS: list[dict] = [
    {
        "name": "mild_academic_stress",
        "description": "Mildly elevated academic load (0.6) -- typical mid-semester pressure",
        "disturbance": LEMURSDisturbance(
            lemurs_intervention={"academic_load": 0.6},
            label="lemurs_mild_academic_stress",
        ),
    },
    {
        "name": "moderate_academic_stress",
        "description": "Moderate academic overload (0.75) -- heavy course load with exams",
        "disturbance": LEMURSDisturbance(
            lemurs_intervention={"academic_load": 0.75},
            label="lemurs_moderate_academic_stress",
        ),
    },
    {
        "name": "severe_academic_stress",
        "description": "Severe academic overload (1.0) -- maximal course burden, constant deadlines",
        "disturbance": LEMURSDisturbance(
            lemurs_intervention={"academic_load": 1.0},
            label="lemurs_severe_academic_stress",
        ),
    },
]

SLEEP_DISRUPTION_SCENARIOS: list[dict] = [
    {
        "name": "mild_insomnia",
        "description": "Mild insomnia -- reduced sleep hygiene (0.7) with some caffeine (0.3)",
        "disturbance": LEMURSDisturbance(
            lemurs_intervention={"sleep_hygiene": 0.7, "caffeine_reduction": 0.3},
            label="lemurs_mild_insomnia",
        ),
    },
    {
        "name": "chronic_insomnia",
        "description": "Chronic insomnia -- poor sleep hygiene (0.4) and minimal caffeine control (0.1)",
        "disturbance": LEMURSDisturbance(
            lemurs_intervention={"sleep_hygiene": 0.4, "caffeine_reduction": 0.1},
            label="lemurs_chronic_insomnia",
        ),
    },
    {
        "name": "severe_deprivation",
        "description": "Severe sleep deprivation -- negligible hygiene (0.1), no caffeine control, high academic load (0.8)",
        "disturbance": LEMURSDisturbance(
            lemurs_intervention={"sleep_hygiene": 0.1, "caffeine_reduction": 0.0, "academic_load": 0.8},
            label="lemurs_severe_deprivation",
        ),
    },
]

ANXIETY_EXPOSURE_SCENARIOS: list[dict] = [
    {
        "name": "low_risk_anxiety",
        "description": "Low anxiety risk -- high emotional stability (5.0), default patient otherwise",
        "disturbance": LEMURSDisturbance(
            lemurs_patient={"emotional_stability": 5.0},
            label="lemurs_low_risk_anxiety",
        ),
    },
    {
        "name": "moderate_anxiety",
        "description": "Moderate anxiety risk -- reduced emotional stability (3.0) with trauma load (2.0)",
        "disturbance": LEMURSDisturbance(
            lemurs_patient={"emotional_stability": 3.0, "trauma_load": 2.0},
            label="lemurs_moderate_anxiety",
        ),
    },
    {
        "name": "vulnerable_anxiety",
        "description": "High anxiety vulnerability -- low stability (1.5), high trauma (4.0), prior MH diagnosis",
        "disturbance": LEMURSDisturbance(
            lemurs_patient={"emotional_stability": 1.5, "trauma_load": 4.0, "mh_diagnosis": 1.0},
            label="lemurs_vulnerable_anxiety",
        ),
    },
]

DIGITAL_BURNOUT_SCENARIOS: list[dict] = [
    {
        "name": "moderate_screen",
        "description": "Moderate digital immersion -- minimal nature (0.1) and exercise (0.1)",
        "disturbance": LEMURSDisturbance(
            lemurs_intervention={"nature_rx": 0.1, "exercise_rx": 0.1},
            label="lemurs_moderate_screen",
        ),
    },
    {
        "name": "full_digital_addiction",
        "description": "Full digital addiction -- zero nature and exercise, near-maximal academic load (0.9)",
        "disturbance": LEMURSDisturbance(
            lemurs_intervention={"nature_rx": 0.0, "exercise_rx": 0.0, "academic_load": 0.9},
            label="lemurs_full_digital_addiction",
        ),
    },
]

COMBINED_SEMESTER_SCENARIOS: list[dict] = [
    {
        "name": "finals_week_vulnerable",
        "description": "Finals week for a vulnerable student -- maximal academic load, negligible sleep, no caffeine control",
        "disturbance": LEMURSDisturbance(
            lemurs_intervention={"academic_load": 1.0, "sleep_hygiene": 0.1, "caffeine_reduction": 0.0},
            label="lemurs_finals_week_vulnerable",
        ),
    },
    {
        "name": "pandemic_isolation",
        "description": "Pandemic-era isolation -- zero nature, exercise, and therapy with poor sleep hygiene (0.3)",
        "disturbance": LEMURSDisturbance(
            lemurs_intervention={"nature_rx": 0.0, "exercise_rx": 0.0, "therapy_rx": 0.0, "sleep_hygiene": 0.3},
            label="lemurs_pandemic_isolation",
        ),
    },
    {
        "name": "burnout_cascade",
        "description": "Full burnout cascade -- maximal load, no nature/exercise, low stability, prior MH diagnosis",
        "disturbance": LEMURSDisturbance(
            lemurs_intervention={"academic_load": 1.0, "nature_rx": 0.0, "exercise_rx": 0.0},
            lemurs_patient={"emotional_stability": 2.0, "mh_diagnosis": 1.0},
            label="lemurs_burnout_cascade",
        ),
    },
]


# -- Aggregate all scenarios --------------------------------------------------

LEMURS_STRESS_SCENARIOS: list[dict] = (
    ACADEMIC_STRESS_SCENARIOS
    + SLEEP_DISRUPTION_SCENARIOS
    + ANXIETY_EXPOSURE_SCENARIOS
    + DIGITAL_BURNOUT_SCENARIOS
    + COMBINED_SEMESTER_SCENARIOS
)


def lemurs_scenario_disturbances(scenario_name: str) -> list[LEMURSDisturbance]:
    """Get disturbances matching a scenario name.

    Args:
        scenario_name: The name field of the scenario to look up.

    Returns:
        List of LEMURSDisturbance objects whose scenario name matches.
        Typically returns a single-element list.
    """
    return [
        s["disturbance"] for s in LEMURS_STRESS_SCENARIOS
        if s["name"] == scenario_name
    ]
