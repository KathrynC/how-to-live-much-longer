"""Genetics module — maps genotype and sex to core patient parameter modifiers.

Consumes GENOTYPE_MULTIPLIERS from constants.py. Outputs modifier dicts that
the ParameterResolver applies to the core 6D patient params before the Cramer
ODE sees them. Never imports from or modifies simulator.py.

References:
    APOE4: O'Shea et al. 2024
    Sex differences: Ivanich et al. 2025
    FOXO3: longevity association studies
    CD38: Camacho-Pereira et al. 2016 (via Cramer Ch. VI.A.3)
"""
from __future__ import annotations

from constants import (
    GENOTYPE_MULTIPLIERS,
    FEMALE_APOE4_INFLAMMATION_BOOST,
    MENOPAUSE_HETEROPLASMY_ACCELERATION,
    ESTROGEN_PROTECTION_LOSS_FACTOR,
)


def compute_genetic_modifiers(
    apoe_genotype: int = 0,
    foxo3_protective: int = 0,
    cd38_risk: int = 0,
) -> dict[str, float]:
    """Compute multiplicative modifiers from genotype.

    Args:
        apoe_genotype: 0=non-carrier, 1=heterozygous, 2=homozygous APOE4.
        foxo3_protective: 1 if rs9486902 CC genotype, else 0.
        cd38_risk: 1 if rs6449197 risk variant, else 0.

    Returns:
        Dict with keys: vulnerability, inflammation, nad_efficiency,
        alcohol_sensitivity, grief_sensitivity, mef2_induction,
        amyloid_clearance. All default to 1.0 (neutral).
    """
    mods = {
        'vulnerability': 1.0,
        'inflammation': 1.0,
        'nad_efficiency': 1.0,
        'alcohol_sensitivity': 1.0,
        'grief_sensitivity': 1.0,
        'mef2_induction': 1.0,
        'amyloid_clearance': 1.0,
        'mitophagy_efficiency': 1.0,
    }

    apoe_key = {1: 'apoe4_het', 2: 'apoe4_hom'}.get(apoe_genotype)
    if apoe_key and apoe_key in GENOTYPE_MULTIPLIERS:
        gm = GENOTYPE_MULTIPLIERS[apoe_key]
        mods['vulnerability'] *= gm.get('vulnerability', 1.0)
        mods['inflammation'] *= gm.get('inflammation', 1.0)
        mods['alcohol_sensitivity'] *= gm.get('alcohol_sensitivity', 1.0)
        mods['grief_sensitivity'] *= gm.get('grief_sensitivity', 1.0)
        mods['mef2_induction'] *= gm.get('mef2_induction', 1.0)
        mods['amyloid_clearance'] *= gm.get('amyloid_clearance', 1.0)
        mods['mitophagy_efficiency'] *= gm.get('mitophagy_efficiency', 1.0)

    if foxo3_protective:
        fm = GENOTYPE_MULTIPLIERS.get('foxo3_protective', {})
        mods['vulnerability'] *= fm.get('vulnerability', 1.0)
        mods['inflammation'] *= fm.get('inflammation', 1.0)
        mods['mitophagy_efficiency'] *= fm.get('mitophagy_efficiency', 1.0)

    if cd38_risk:
        cm = GENOTYPE_MULTIPLIERS.get('cd38_risk', {})
        mods['nad_efficiency'] *= cm.get('nad_efficiency', 1.0)

    return mods


def compute_sex_modifiers(
    sex: str = 'M',
    menopause_status: str = 'pre',
    estrogen_therapy: int = 0,
) -> dict[str, float]:
    """Compute additive/multiplicative modifiers from biological sex.

    Args:
        sex: 'M' or 'F'.
        menopause_status: 'pre', 'peri', or 'post' (females only).
        estrogen_therapy: 1 if on HRT, else 0.

    Returns:
        Dict with keys: inflammation_delta (additive to inflammation_level),
        heteroplasmy_multiplier (multiplicative on baseline_heteroplasmy).
    """
    mods = {
        'inflammation_delta': 0.0,
        'heteroplasmy_multiplier': 1.0,
    }

    if sex == 'F' and menopause_status in ('peri', 'post'):
        base_inflammation = 0.1 if menopause_status == 'post' else 0.05
        base_het_mult = MENOPAUSE_HETEROPLASMY_ACCELERATION if menopause_status == 'post' else 1.02

        if estrogen_therapy:
            base_inflammation *= 0.5
            base_het_mult = 1.0 + (base_het_mult - 1.0) * 0.5

        mods['inflammation_delta'] = base_inflammation
        mods['heteroplasmy_multiplier'] = base_het_mult

    return mods


def apply_genetic_modifiers(
    patient_12d: dict[str, float],
    expanded_params: dict,
) -> dict[str, float]:
    """Apply genotype and sex modifiers to a core 12D patient dict.

    Returns a new dict (does not mutate input).
    """
    result = dict(patient_12d)

    genetic = compute_genetic_modifiers(
        apoe_genotype=expanded_params.get('apoe_genotype', 0),
        foxo3_protective=expanded_params.get('foxo3_protective', 0),
        cd38_risk=expanded_params.get('cd38_risk', 0),
    )

    sex = compute_sex_modifiers(
        sex=expanded_params.get('sex', 'M'),
        menopause_status=expanded_params.get('menopause_status', 'pre'),
        estrogen_therapy=expanded_params.get('estrogen_therapy', 0),
    )

    result['genetic_vulnerability'] *= genetic['vulnerability']
    result['inflammation_level'] = min(1.0, result['inflammation_level'] * genetic['inflammation'] + sex['inflammation_delta'])
    result['baseline_heteroplasmy'] *= sex['heteroplasmy_multiplier']
    result['baseline_nad_level'] *= genetic['nad_efficiency']

    return result
