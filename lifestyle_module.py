"""Lifestyle module — alcohol, coffee, diet, and fasting effects.

Maps lifestyle parameters to modifiers on the core 12D patient/intervention
params. Never imports from simulator.py.

References:
    Alcohol: Anttila et al. 2004, Downer et al. 2014
    Coffee: Nature Metabolism 2024 (trigonelline/NAD+ pathway)
    Diet/fasting: Ivanich et al. 2025
"""
from __future__ import annotations

from constants import (
    ALCOHOL_INFLAMMATION_FACTOR,
    ALCOHOL_NAD_FACTOR,
    ALCOHOL_SLEEP_DISRUPTION,
    COFFEE_TRIGONELLINE_NAD_EFFECT,
    COFFEE_CHLOROGENIC_ACID_ANTI_INFLAMMATORY,
    COFFEE_MAX_BENEFICIAL_CUPS,
    COFFEE_PREPARATION_MULTIPLIERS,
    COFFEE_APOE4_BENEFIT_MULTIPLIER,
    COFFEE_FEMALE_BENEFIT_MULTIPLIER,
    KETONE_ATP_FACTOR,
    IF_MITOPHAGY_FACTOR,
)


def compute_alcohol_effects(
    alcohol_intake: float = 0.0,
    apoe_sensitivity: float = 1.0,
) -> dict[str, float]:
    """Compute alcohol effects on inflammation and NAD+.

    Args:
        alcohol_intake: 0-1 normalized consumption.
        apoe_sensitivity: genotype alcohol sensitivity multiplier.

    Returns:
        Dict with inflammation_delta (additive), nad_multiplier (multiplicative),
        sleep_disruption (additive reduction to sleep quality).
    """
    return {
        'inflammation_delta': alcohol_intake * ALCOHOL_INFLAMMATION_FACTOR * apoe_sensitivity,
        'nad_multiplier': 1.0 - alcohol_intake * ALCOHOL_NAD_FACTOR * apoe_sensitivity,
        'sleep_disruption': alcohol_intake * ALCOHOL_SLEEP_DISRUPTION,
    }


def compute_coffee_effects(
    cups: float = 0.0,
    coffee_type: str = 'filtered',
    sex: str = 'M',
    apoe_genotype: int = 0,
) -> dict[str, float]:
    """Compute coffee effects on NAD+ and inflammation.

    Benefits cap at COFFEE_MAX_BENEFICIAL_CUPS. Preparation method scales
    bioavailability. APOE4 carriers and females get additional benefit.
    """
    effective_cups = min(cups, COFFEE_MAX_BENEFICIAL_CUPS)
    prep_mult = COFFEE_PREPARATION_MULTIPLIERS.get(coffee_type, 1.0)

    genotype_mult = COFFEE_APOE4_BENEFIT_MULTIPLIER if apoe_genotype > 0 else 1.0
    sex_mult = COFFEE_FEMALE_BENEFIT_MULTIPLIER if sex == 'F' else 1.0

    nad_boost = effective_cups * COFFEE_TRIGONELLINE_NAD_EFFECT * prep_mult * genotype_mult * sex_mult
    inflammation_reduction = effective_cups * COFFEE_CHLOROGENIC_ACID_ANTI_INFLAMMATORY * prep_mult

    return {
        'nad_boost': nad_boost,
        'inflammation_reduction': inflammation_reduction,
    }


def compute_diet_effects(
    diet_type: str = 'standard',
    fasting_regimen: float = 0.0,
) -> dict[str, float]:
    """Compute dietary effects on metabolic demand and mitophagy."""
    demand_mult = {
        'standard': 1.0,
        'mediterranean': 0.97,
        'keto': 1.0 - KETONE_ATP_FACTOR,
    }.get(diet_type, 1.0)

    gut_boost = {
        'standard': 0.0,
        'mediterranean': 0.03,
        'keto': 0.05,
    }.get(diet_type, 0.0)

    return {
        'demand_multiplier': demand_mult,
        'mitophagy_boost': fasting_regimen * IF_MITOPHAGY_FACTOR,
        'gut_health_boost': gut_boost,
    }
