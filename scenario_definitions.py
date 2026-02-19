"""Scenario definitions — InterventionProfile, Scenario, and predefined A-D.

Defines the data structures for batch scenario comparison:
- InterventionProfile: all intervention parameters with defaults of 0.0
- Scenario: patient + interventions + duration + output metrics
- get_example_scenarios(): 4 predefined scenarios (A-D) for a 63-year-old
  APOE4 heterozygous female

Scenarios progress from sleep hygiene (A) through OTC supplements (B),
prescription drugs (C), and experimental therapies (D).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class InterventionProfile:
    """All intervention parameters with defaults of 0.0."""
    # Core 6
    rapamycin_dose: float = 0.0
    nad_supplement: float = 0.0
    senolytic_dose: float = 0.0
    yamanaka_intensity: float = 0.0
    transplant_rate: float = 0.0
    exercise_level: float = 0.0
    # Sleep
    sleep_intervention: float = 0.0
    # Dietary
    diet_type: str = 'standard'
    fasting_regimen: float = 0.0
    alcohol_intake: float = 0.0
    coffee_intake: float = 0.0
    coffee_type: str = 'filtered'
    # Probiotics
    probiotic_intensity: float = 0.0
    # Supplements (11)
    nr_dose: float = 0.0
    dha_dose: float = 0.0
    coq10_dose: float = 0.0
    resveratrol_dose: float = 0.0
    pqq_dose: float = 0.0
    ala_dose: float = 0.0
    vitamin_d_dose: float = 0.0
    b_complex_dose: float = 0.0
    magnesium_dose: float = 0.0
    zinc_dose: float = 0.0
    selenium_dose: float = 0.0
    # Therapy
    therapy_intensity: float = 0.0
    intellectual_engagement_intervention: float = 0.0

    def to_dict(self) -> dict:
        """Convert to flat dict for ParameterResolver."""
        return asdict(self)


@dataclass
class Scenario:
    """A named scenario combining patient state and intervention protocol."""
    name: str
    description: str
    patient_params: dict
    interventions: InterventionProfile
    duration_years: float = 30.0
    output_metrics: list = field(default_factory=lambda: ['heteroplasmy', 'atp', 'memory_index'])


# ── Base patient ────────────────────────────────────────────────────────────

BASE_PATIENT = {
    'baseline_age': 63.0,
    'baseline_heteroplasmy': 0.62,
    'baseline_nad_level': 0.45,
    'genetic_vulnerability': 1.0,
    'metabolic_demand': 1.0,
    'inflammation_level': 0.466,
    'apoe_genotype': 1,
    'foxo3_protective': 0,
    'cd38_risk': 0,
    'sex': 'F',
    'menopause_status': 'post',
    'estrogen_therapy': 0,
    'grief_intensity': 0.18,
    'grief_duration': 10,
    'therapy_intensity': 0.3,
    'social_support': 0.5,
    'love_presence': 0.3,
    'intellectual_engagement': 0.9,
    'education_level': 'doctoral',
    'occupational_complexity': 0.8,
}


# ── Predefined scenarios ────────────────────────────────────────────────────

def _scenario_a_interventions() -> InterventionProfile:
    """A: Sleep + Alcohol Cessation."""
    return InterventionProfile(
        sleep_intervention=0.8,
        alcohol_intake=0.0,
        coffee_intake=2,
        coffee_type='filtered',
    )


def _scenario_b_interventions() -> InterventionProfile:
    """B: A + OTC Supplements + Keto."""
    ip = _scenario_a_interventions()
    ip.nr_dose = 0.8
    ip.dha_dose = 0.8
    ip.coq10_dose = 0.7
    ip.resveratrol_dose = 0.7
    ip.pqq_dose = 0.7
    ip.ala_dose = 0.7
    ip.vitamin_d_dose = 0.8
    ip.b_complex_dose = 0.7
    ip.magnesium_dose = 0.8
    ip.zinc_dose = 0.7
    ip.selenium_dose = 0.7
    ip.diet_type = 'keto'
    ip.fasting_regimen = 0.5
    ip.probiotic_intensity = 0.8
    return ip


def _scenario_c_interventions() -> InterventionProfile:
    """C: B + Prescription."""
    ip = _scenario_b_interventions()
    ip.rapamycin_dose = 0.8
    ip.senolytic_dose = 0.8
    return ip


def _scenario_d_interventions() -> InterventionProfile:
    """D: C + Experimental."""
    ip = _scenario_c_interventions()
    ip.transplant_rate = 0.9
    ip.yamanaka_intensity = 0.5
    return ip


def get_example_scenarios() -> list[Scenario]:
    """Return the 4 predefined scenarios A-D for the base patient."""
    return [
        Scenario(
            name="A: Sleep + Alcohol Cessation",
            description="Foundational lifestyle: optimize sleep, eliminate alcohol, add filtered coffee.",
            patient_params=dict(BASE_PATIENT),
            interventions=_scenario_a_interventions(),
        ),
        Scenario(
            name="B: A + OTC Supplements + Keto",
            description="Add 11 OTC supplements (NR, DHA, CoQ10, resveratrol, PQQ, ALA, "
                        "vitamin D, B complex, magnesium, zinc, selenium), keto diet, "
                        "intermittent fasting, and probiotics.",
            patient_params=dict(BASE_PATIENT),
            interventions=_scenario_b_interventions(),
        ),
        Scenario(
            name="C: B + Prescription",
            description="Add prescription rapamycin (mTOR inhibition, enhanced mitophagy) "
                        "and senolytics (dasatinib + quercetin).",
            patient_params=dict(BASE_PATIENT),
            interventions=_scenario_c_interventions(),
        ),
        Scenario(
            name="D: C + Experimental",
            description="Add experimental therapies: mitochondrial transplant (platelet-derived "
                        "mitlets) and partial Yamanaka reprogramming (OSKM).",
            patient_params=dict(BASE_PATIENT),
            interventions=_scenario_d_interventions(),
        ),
    ]
