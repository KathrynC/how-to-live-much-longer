"""Supplement module — Hill-function dose-response for 11 nutraceuticals.

Each supplement has a max effect and half-max dose. Effects are categorized
into nad_boost, inflammation_reduction, mitophagy_boost, and demand_reduction
for mapping onto the core 12D params.

References: Hill-function parameters are simulation estimates (no single
    literature source). See constants.py for provenance note.
"""
from __future__ import annotations

from constants import (
    MAX_NR_EFFECT, NR_HALF_MAX,
    MAX_DHA_EFFECT, DHA_HALF_MAX,
    MAX_COQ10_EFFECT, COQ10_HALF_MAX,
    MAX_RESVERATROL_EFFECT, RESVERATROL_HALF_MAX,
    MAX_PQQ_EFFECT, PQQ_HALF_MAX,
    MAX_ALA_EFFECT, ALA_HALF_MAX,
    MAX_VITAMIN_D_EFFECT, VITAMIN_D_HALF_MAX,
    MAX_B_COMPLEX_EFFECT, B_COMPLEX_HALF_MAX,
    MAX_MAGNESIUM_EFFECT, MAGNESIUM_HALF_MAX,
    MAX_ZINC_EFFECT, ZINC_HALF_MAX,
    MAX_SELENIUM_EFFECT, SELENIUM_HALF_MAX,
    MIN_NAD_CONVERSION_EFFICIENCY, MAX_NAD_CONVERSION_EFFICIENCY,
)


def hill_effect(dose: float, max_effect: float, half_max: float) -> float:
    """Michaelis-Menten / Hill dose-response with diminishing returns.

    Returns: max_effect * dose / (dose + half_max). Zero at dose=0,
    asymptotes to max_effect at high dose.
    """
    if dose <= 0:
        return 0.0
    return max_effect * dose / (dose + half_max)


def nad_conversion_efficiency(gut_health: float) -> float:
    """Gut health modulates NAD+ precursor conversion efficiency.

    Returns a multiplier between MIN_NAD_CONVERSION_EFFICIENCY (0.7) and
    MAX_NAD_CONVERSION_EFFICIENCY (1.0).
    """
    return MIN_NAD_CONVERSION_EFFICIENCY + gut_health * (
        MAX_NAD_CONVERSION_EFFICIENCY - MIN_NAD_CONVERSION_EFFICIENCY
    )


def compute_supplement_effects(
    supplements: dict[str, float],
    gut_health: float = 0.5,
) -> dict[str, float]:
    """Aggregate supplement effects across all 11 nutraceuticals.

    Args:
        supplements: Dict of supplement_name -> dose (0-1).
            Keys: nr_dose, dha_dose, coq10_dose, resveratrol_dose,
            pqq_dose, ala_dose, vitamin_d_dose, b_complex_dose,
            magnesium_dose, zinc_dose, selenium_dose.
        gut_health: Current gut microbiome health (0-1), affects NAD conversion.

    Returns:
        Dict with:
            nad_boost: additive boost to effective nad_supplement
            inflammation_reduction: subtractive from inflammation_level
            mitophagy_boost: additive to effective rapamycin_dose
            demand_reduction: subtractive from metabolic_demand
            sleep_boost: additive to sleep quality
    """
    conversion = nad_conversion_efficiency(gut_health)

    nr = hill_effect(supplements.get('nr_dose', 0), MAX_NR_EFFECT, NR_HALF_MAX) * conversion
    b_complex = hill_effect(supplements.get('b_complex_dose', 0), MAX_B_COMPLEX_EFFECT, B_COMPLEX_HALF_MAX) * conversion
    nad_boost = nr * 0.25 + b_complex * 0.1

    dha = hill_effect(supplements.get('dha_dose', 0), MAX_DHA_EFFECT, DHA_HALF_MAX)
    ala = hill_effect(supplements.get('ala_dose', 0), MAX_ALA_EFFECT, ALA_HALF_MAX)
    vit_d = hill_effect(supplements.get('vitamin_d_dose', 0), MAX_VITAMIN_D_EFFECT, VITAMIN_D_HALF_MAX)
    zinc = hill_effect(supplements.get('zinc_dose', 0), MAX_ZINC_EFFECT, ZINC_HALF_MAX)
    selenium = hill_effect(supplements.get('selenium_dose', 0), MAX_SELENIUM_EFFECT, SELENIUM_HALF_MAX)
    inflammation_reduction = dha + ala + vit_d + zinc + selenium

    resveratrol = hill_effect(supplements.get('resveratrol_dose', 0), MAX_RESVERATROL_EFFECT, RESVERATROL_HALF_MAX)
    pqq = hill_effect(supplements.get('pqq_dose', 0), MAX_PQQ_EFFECT, PQQ_HALF_MAX)
    mitophagy_boost = resveratrol + pqq

    coq10 = hill_effect(supplements.get('coq10_dose', 0), MAX_COQ10_EFFECT, COQ10_HALF_MAX)
    demand_reduction = coq10

    magnesium = hill_effect(supplements.get('magnesium_dose', 0), MAX_MAGNESIUM_EFFECT, MAGNESIUM_HALF_MAX)
    sleep_boost = magnesium * 0.5

    return {
        'nad_boost': nad_boost,
        'inflammation_reduction': inflammation_reduction,
        'mitophagy_boost': mitophagy_boost,
        'demand_reduction': demand_reduction,
        'sleep_boost': sleep_boost,
    }
