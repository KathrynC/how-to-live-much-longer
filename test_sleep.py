import numpy as np
from parameter_resolver import ParameterResolver
from simulator import simulate
from analytics import compute_all

# Default patient (70yo)
patient = {'baseline_age': 70.0, 'baseline_heteroplasmy': 0.3, 'baseline_nad_level': 0.6,
           'genetic_vulnerability': 1.0, 'metabolic_demand': 1.0, 'inflammation_level': 0.25}
# Vary sleep intervention
for sleep_int in [0.1, 0.5, 0.9]:
    resolver = ParameterResolver(
        patient_params=patient,
        intervention_params={'sleep_intervention': sleep_int},
        # other interventions zero
        rapamycin_dose=0.0, nad_supplement=0.0, senolytic_dose=0.0,
        yamanaka_intensity=0.0, transplant_rate=0.0, exercise_level=0.0,
        alcohol_intake=0.0, coffee_intake=0.0, diet_type=0.5,
        probiotic_intensity=0.5, therapy_intensity=0.5,
        # genetics
        apoe_genotype=0, sex='M', menopause_status=0,
        grief_intensity=0.0, intellectual_engagement=0.5, education_level=0.5,
        # supplements zero
        nmn_dose=0.0, nr_dose=0.0, apigenin_dose=0.0, fisetin_dose=0.0,
        quercetin_dose=0.0, pterostilbene_dose=0.0, melatonin_dose=0.0,
        curcumin_dose=0.0, resveratrol_dose=0.0, caffeine_dose=0.0,
        theanine_dose=0.0,
        # time-varying trajectories
        alcohol_trajectory=None,
        lemurs_override=None,
        grief_trajectory=None,
    )
    # Simulate with resolver
    result = simulate(resolver=resolver)
    atp = result['states'][-1, 2]
    het = result['heteroplasmy'][-1]
    print(f'Sleep intervention {sleep_int:.1f}: ATP={atp:.4f}, Het={het:.4f}')
